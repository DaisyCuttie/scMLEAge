"""Helpers for creating the r-square output path once.

This companion module keeps the notebook naming convention:
    ./R_Squareds/{organ}_r_squared_summary.csv

Unlike the notebook-local helper, it does not delete an existing file.
"""

from __future__ import annotations

from pathlib import Path


def ensure_r_square_file_path(
    organ: str,
    r_squareds_dir: str | Path = "./R_Squareds",
) -> str:
    """Create the output directory if needed and return the summary CSV path.

    If the CSV already exists, it is left untouched.
    """
    output_dir = Path(r_squareds_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{organ}_r_squared_summary.csv"
    return str(file_path)


def create_r_square_file_path(
    organ: str,
    r_squareds_dir: str | Path = "./R_Squareds",
) -> str:
    """Backwards-friendly alias for callers that prefer the notebook wording."""
    return ensure_r_square_file_path(organ, r_squareds_dir)
