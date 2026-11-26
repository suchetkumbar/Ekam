"""DuckDB connections that spill into the repo's .cache (not ./.tmp) with a memory cap."""

from __future__ import annotations

from pathlib import Path

import duckdb


def connect(memory_limit: str = "4GB",
            temp_dir: str = ".cache/duckdb") -> duckdb.DuckDBPyConnection:
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    return duckdb.connect(config={"memory_limit": memory_limit, "temp_directory": temp_dir})
