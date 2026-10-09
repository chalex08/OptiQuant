import sqlite3
from importlib.resources import files
from pathlib import Path

DEFAULT_DB_PATH = Path("data/optiquant.db")


def connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open the results DB, creating the file and tables if they don't exist yet."""
    if str(db_path) != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    schema = files("optiquant").joinpath("schema.sql").read_text(encoding="utf-8")
    conn.executescript(schema)
    return conn


def table_columns(conn: sqlite3.Connection, table: str = "benchmark_runs") -> list[str]:
    return [row["name"] for row in conn.execute(f"PRAGMA table_info({table})")]
