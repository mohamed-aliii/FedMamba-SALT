"""Helpers for SSL-FL split CSV discovery."""

from pathlib import Path


def discover_client_split_csvs(data_path: str, n_clients: int, split_type: str) -> list[str]:
    """
    Return federated split CSV paths relative to ``data_path``.

    Retina split folders use numbered files (client_1.csv, ...), while
    COVID-FL's real-world split uses site names (bimcv.csv, rsna-0.csv, ...).
    """
    split_dir = Path(data_path) / f"{n_clients}_clients" / split_type
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Client split directory not found: {split_dir}")

    numbered = [split_dir / f"client_{i}.csv" for i in range(1, n_clients + 1)]
    if all(path.is_file() for path in numbered):
        files = numbered
    else:
        files = sorted(split_dir.glob("*.csv"))
        if len(files) != n_clients:
            raise FileNotFoundError(
                f"Expected {n_clients} split CSVs in {split_dir}, found {len(files)}."
            )

    return [
        str(Path(f"{n_clients}_clients") / split_type / path.name)
        for path in files
    ]
