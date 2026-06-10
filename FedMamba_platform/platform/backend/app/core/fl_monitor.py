"""
FL Output Monitor — parses CSV metrics and checkpoint files.

CSV column names are taken from the ACTUAL research script source files.
Never import research modules — read files only.
"""
import csv
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exact CSV column mappings from research script source files
# platform_key → csv_column_name (None = not available for this script)
# ---------------------------------------------------------------------------
CSV_COLUMN_MAPS: dict[str, dict[str, str | None]] = {
    "training_metrics.csv": {
        "round":   "epoch",
        "loss":    "loss",
        "val_acc": None,
        "val_auc": None,
        "lr":      "lr",
        "time_s":  "epoch_time_s",
        "enc_std": "student_std",
        "gpu_mb":  "gpu_mem_allocated_mb",
    },
    "federated_metrics.csv": {
        "round":   "round",
        "loss":    "avg_loss",
        "val_acc": None,
        "val_auc": None,
        "lr":      "lr",
        "time_s":  "round_time_s",
        "enc_std": "avg_enc_std",
        "gpu_mb":  "gpu_mb",
    },
    "fed_finetune_metrics.csv": {
        "round":   "round",
        "loss":    "val_loss_weighted",
        "val_acc": "val_acc",
        "val_auc": "auc",
        "lr":      "cls_lr",
        "time_s":  "round_time_s",
        "enc_std": None,
        "gpu_mb":  "gpu_mb",
    },
}

# Regex patterns for stdout line parsing (from actual research f-strings)
_CENTRALIZED_RE = re.compile(
    r"Epoch\s*\[\s*(?P<epoch>\d+)/\s*\d+\]"
    r".*?loss=(?P<loss>[\d.]+)"
    r".*?enc_std=(?P<enc_std>[\d.]+)"
    r".*?lr=(?P<lr>[\d.e+-]+)"
    r".*?time=(?P<time_s>[\d.]+)s"
    r"(?:.*?gpu=(?P<gpu_mb>\d+)MB)?",
    re.DOTALL,
)
_FEDAVG_RE = re.compile(
    r"Round\s*\[\s*(?P<round>\d+)/\s*\d+\]"
    r".*?loss=(?P<loss>[\d.]+)"
    r".*?enc_std=(?P<enc_std>[\d.]+)"
    r".*?lr=(?P<lr>[\d.e+-]+)"
    r".*?time=(?P<time_s>[\d.]+)s",
    re.DOTALL,
)
_FINETUNE_RE = re.compile(
    r"Round\s*\[\s*(?P<round>\d+)/\s*\d+\]"
    r".*?val=(?P<val_acc>[\d.]+)%?"
    r".*?auc=(?P<val_auc>[\d.]+)"
    r".*?cls_lr=(?P<lr>[\d.e+-]+)"
    r".*?time=(?P<time_s>[\d.]+)s",
    re.DOTALL,
)


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _apply_column_map(row: dict, csv_filename: str) -> dict:
    """Normalize a raw CSV row using the column map for its script."""
    col_map = CSV_COLUMN_MAPS.get(csv_filename, {})
    result: dict = {"raw": row}
    for platform_key, csv_col in col_map.items():
        if csv_col is None:
            result[platform_key] = None
        else:
            result[platform_key] = _safe_float(row.get(csv_col))
    return result


class FLOutputMonitor:
    """Parses CSV metrics and checkpoint files from research script output."""

    def __init__(self, research_root: str) -> None:
        self.research_root = Path(research_root).resolve()

    def find_metrics_csv(self, output_dir: str) -> tuple[str, str] | None:
        """Find the metrics CSV in output_dir.

        Checks in order (most specific first):
          fed_finetune_metrics.csv → federated_metrics.csv → training_metrics.csv

        Returns (absolute_path, csv_filename) or None.
        """
        dir_path = Path(output_dir)
        for csv_name in (
            "fed_finetune_metrics.csv",
            "federated_metrics.csv",
            "training_metrics.csv",
        ):
            candidate = dir_path / csv_name
            if candidate.exists():
                return str(candidate), csv_name
        return None

    def parse_latest_round(self, csv_path: str) -> dict | None:
        """Read only the last row of the CSV (memory-efficient).

        Returns a normalized dict:
          {round, loss, val_acc, val_auc, lr, time_s, enc_std, gpu_mb, raw}
        All numeric values cast to float. None for unmapped columns.
        Returns None if file is empty or unreadable.
        """
        csv_filename = Path(csv_path).name
        try:
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                last_row = None
                for last_row in reader:
                    pass
            if last_row is None:
                return None
            return _apply_column_map(last_row, csv_filename)
        except Exception as e:
            logger.warning("Could not parse CSV %s: %s", csv_path, e)
            return None

    def parse_all_rounds(self, csv_path: str) -> list[dict]:
        """Return all rows normalized. Used for initial chart load."""
        csv_filename = Path(csv_path).name
        results = []
        try:
            with open(csv_path, newline="") as f:
                for row in csv.DictReader(f):
                    results.append(_apply_column_map(row, csv_filename))
        except Exception as e:
            logger.warning("Could not parse CSV %s: %s", csv_path, e)
        return results

    def find_checkpoints(self, output_dir: str) -> list[dict]:
        """Scan for ckpt_*.pth files and classify them.

        Types:
          ckpt_latest.pth         → LATEST,   round_number=None
          ckpt_best.pth           → BEST,     round_number=None
          ckpt_best_finetune.pth  → BEST,     round_number=None
          ckpt_epoch_NNNN.pth     → PERIODIC, round_number=NNNN
          ckpt_round_NNNN.pth     → PERIODIC, round_number=NNNN
        """
        results = []
        dir_path = Path(output_dir)
        _periodic_re = re.compile(r"ckpt_(?:epoch|round)_(\d+)\.pth$")

        for pth_file in sorted(dir_path.glob("ckpt_*.pth")):
            name = pth_file.name
            stat = pth_file.stat()

            if name == "ckpt_latest.pth":
                ckpt_type, round_number = "LATEST", None
            elif name in ("ckpt_best.pth", "ckpt_best_finetune.pth"):
                ckpt_type, round_number = "BEST", None
            else:
                m = _periodic_re.match(name)
                if m:
                    ckpt_type = "PERIODIC"
                    round_number = int(m.group(1))
                else:
                    ckpt_type, round_number = "UNKNOWN", None

            results.append({
                "filename": name,
                "path": str(pth_file),
                "type": ckpt_type,
                "round_number": round_number,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified_at": stat.st_mtime,
            })
        return results

    def parse_stdout_line(self, line: str) -> dict | None:
        """Extract real-time metrics from a stdout log line.

        Handles all 3 research script formats (from actual f-strings):
          train_centralized.py:  "  Epoch [ 10/100]  loss=0.0423 ... gpu=1024MB"
          train_fedavg.py:       "  Round [ 10/200]  loss=0.0512 ..."
          train_fed_finetune.py: "  Round [ 10/ 50]  train_acc=72.45%  val=78.23% ..."

        Returns None if no format matches.
        Returns same keys as parse_latest_round() plus "source_script".
        The gpu_mb field captures real GPU memory on CUDA (not 0MB as on CPU).
        """
        # Try centralized format (has "Epoch" keyword)
        if "Epoch" in line:
            m = _CENTRALIZED_RE.search(line)
            if m:
                return {
                    "round":       _safe_float(m.group("epoch")),
                    "loss":        _safe_float(m.group("loss")),
                    "val_acc":     None,
                    "val_auc":     None,
                    "lr":          _safe_float(m.group("lr")),
                    "time_s":      _safe_float(m.group("time_s")),
                    "enc_std":     _safe_float(m.group("enc_std")),
                    "gpu_mb":      _safe_float(m.group("gpu_mb")),
                    "source_script": "train_centralized.py",
                    "raw": line.strip(),
                }

        # Try finetune format (has "val=" and "auc=")
        if "val=" in line and "auc=" in line:
            m = _FINETUNE_RE.search(line)
            if m:
                return {
                    "round":       _safe_float(m.group("round")),
                    "loss":        None,
                    "val_acc":     _safe_float(m.group("val_acc")),
                    "val_auc":     _safe_float(m.group("val_auc")),
                    "lr":          _safe_float(m.group("lr")),
                    "time_s":      _safe_float(m.group("time_s")),
                    "enc_std":     None,
                    "gpu_mb":      None,
                    "source_script": "train_fed_finetune.py",
                    "raw": line.strip(),
                }

        # Try fedavg format (has "Round" keyword)
        if "Round" in line:
            m = _FEDAVG_RE.search(line)
            if m:
                return {
                    "round":       _safe_float(m.group("round")),
                    "loss":        _safe_float(m.group("loss")),
                    "val_acc":     None,
                    "val_auc":     None,
                    "lr":          _safe_float(m.group("lr")),
                    "time_s":      _safe_float(m.group("time_s")),
                    "enc_std":     _safe_float(m.group("enc_std")),
                    "gpu_mb":      None,
                    "source_script": "train_fedavg.py",
                    "raw": line.strip(),
                }

        return None
