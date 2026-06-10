"""
SafePathValidator — prevents path traversal attacks.

Called before building any subprocess command or reading output files.
"""
from pathlib import Path


class SafePathValidator:
    """Validates that a path stays within an allowed base directory."""

    @staticmethod
    def validate(path: str | Path, base_dir: str | Path) -> Path:
        """Resolve path and assert it is under base_dir.

        Args:
            path: The path to validate.
            base_dir: The directory it must be contained within.

        Returns:
            The resolved absolute Path.

        Raises:
            ValueError: If path escapes base_dir (symlinks followed).
        """
        resolved_path = Path(path).resolve()
        resolved_base = Path(base_dir).resolve()

        try:
            resolved_path.relative_to(resolved_base)
        except ValueError:
            raise ValueError(
                f"Path traversal detected: '{resolved_path}' is not under "
                f"base directory '{resolved_base}'"
            )

        return resolved_path
