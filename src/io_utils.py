"""File I/O helpers with consistent paths and serialisation."""

from __future__ import annotations
import json
import pickle
import time
from pathlib import Path
from typing import Any

import polars as pl


DATA_DIR = Path("data")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")


def get_data_dir() -> Path:
    return DATA_DIR


def get_models_dir() -> Path:
    return MODELS_DIR


def get_reports_dir() -> Path:
    return REPORTS_DIR


def load_parquet(path: str | Path, columns: list[str] | None = None) -> pl.DataFrame:
    return pl.read_parquet(path, columns=columns)


def save_parquet(df: pl.DataFrame, path: str | Path, compression: str = "zstd") -> Path:
    path = Path(path)
    df.write_parquet(path, compression=compression)
    return path


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def save_json(data: Any, path: str | Path, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=str)
    return path


def load_pickle(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(data: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return path


def report_file_size(path: Path) -> float:
    """Return file size in MB."""
    return path.stat().st_size / (1024 * 1024)


class timer:
    """Context manager / direct-use timer.

    Usage:
        with timer() as t:
            do_work()
        print(f"Elapsed: {t.elapsed:.1f}s")

    Or:
        t = timer()
        t.start()
        do_work()
        print(f"Elapsed: {t.elapsed:.1f}s")
    """

    def __init__(self):
        self._start: float | None = None

    def start(self) -> None:
        self._start = time.time()

    @property
    def elapsed(self) -> float:
        if self._start is None:
            return 0.0
        return time.time() - self._start

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        pass
