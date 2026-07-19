"""SQLite persistence for viewer accounts and background jobs.

One database file at data/app.db. Connections are short-lived (opened per
call) with WAL mode so the request threads and the job worker can share it.
"""

import json
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "app.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    created_at    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id),
    kind        TEXT NOT NULL,
    params      TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'queued',
    result      TEXT,
    error       TEXT,
    created_at  REAL NOT NULL,
    started_at  REAL,
    finished_at REAL
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status, id);
"""


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=15)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    return con


def init_db() -> None:
    with connect() as con:
        con.executescript(SCHEMA)


# ------------------------------------------------------------------- users


def create_user(username: str, password_hash: str) -> int | None:
    """Insert a user; returns the new id, or None if the name is taken."""
    with connect() as con:
        try:
            cur = con.execute(
                "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                (username, password_hash, time.time()),
            )
        except sqlite3.IntegrityError:
            return None
        return cur.lastrowid


def get_user_by_name(username: str) -> sqlite3.Row | None:
    with connect() as con:
        return con.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()


def get_user(user_id: int) -> sqlite3.Row | None:
    with connect() as con:
        return con.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


# -------------------------------------------------------------------- jobs


def create_job(user_id: int, kind: str, params: dict) -> int:
    with connect() as con:
        cur = con.execute(
            "INSERT INTO jobs (user_id, kind, params, created_at) VALUES (?, ?, ?, ?)",
            (user_id, kind, json.dumps(params), time.time()),
        )
        return cur.lastrowid


def claim_next_job() -> sqlite3.Row | None:
    """Atomically mark the oldest queued job as running and return it."""
    with connect() as con:
        row = con.execute(
            "SELECT * FROM jobs WHERE status = 'queued' ORDER BY id LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        cur = con.execute(
            "UPDATE jobs SET status = 'running', started_at = ? "
            "WHERE id = ? AND status = 'queued'",
            (time.time(), row["id"]),
        )
        return row if cur.rowcount == 1 else None  # lost the race; retry next poll


def finish_job(job_id: int, result: dict | list | None = None,
               error: str | None = None) -> None:
    with connect() as con:
        con.execute(
            "UPDATE jobs SET status = ?, result = ?, error = ?, finished_at = ? "
            "WHERE id = ?",
            ("error" if error else "done",
             None if result is None else json.dumps(result),
             error, time.time(), job_id),
        )


def get_job(job_id: int, user_id: int) -> sqlite3.Row | None:
    """A user's own job, or None (other users' jobs are invisible)."""
    with connect() as con:
        return con.execute(
            "SELECT * FROM jobs WHERE id = ? AND user_id = ?", (job_id, user_id)
        ).fetchone()


def fail_stale_jobs() -> int:
    """Mark queued/running jobs from a previous process as failed (startup)."""
    with connect() as con:
        cur = con.execute(
            "UPDATE jobs SET status = 'error', error = 'server restarted', "
            "finished_at = ? WHERE status IN ('queued', 'running')",
            (time.time(),),
        )
        return cur.rowcount
