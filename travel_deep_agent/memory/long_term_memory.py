"""
memory/long_term_memory.py
---------------------------
This is the agent's LONG-TERM MEMORY: a tiny SQLite database that survives
between separate runs of the program (unlike the ContextManager, which only
lives for one conversation).

Two things are remembered per traveler (user_id):
  1. preferences   -> key/value facts, e.g. currency="INR", seat="window"
  2. trips          -> a short JSON summary of every trip that was planned

A beginner can open data/travel_memory.db with any SQLite viewer to see it.
"""
import json
import sqlite3
import time
from contextlib import contextmanager

from config import DB_PATH


class LongTermMemory:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self):
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS preferences (
                    user_id TEXT, key TEXT, value TEXT, updated_at REAL,
                    PRIMARY KEY (user_id, key)
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS trips (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT, summary TEXT, created_at REAL
                )"""
            )

    # ---- preferences -------------------------------------------------
    def set_preference(self, user_id: str, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "REPLACE INTO preferences (user_id, key, value, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (user_id, key, value, time.time()),
            )

    def get_preferences(self, user_id: str) -> dict:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT key, value FROM preferences WHERE user_id = ?", (user_id,)
            ).fetchall()
        return {k: v for k, v in rows}

    # ---- trip history --------------------------------------------------
    def save_trip(self, user_id: str, trip_summary: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO trips (user_id, summary, created_at) VALUES (?, ?, ?)",
                (user_id, json.dumps(trip_summary), time.time()),
            )

    def get_past_trips(self, user_id: str, limit: int = 3) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT summary FROM trips WHERE user_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def memory_briefing(self, user_id: str) -> str:
        """A short text block subagents can drop straight into a prompt."""
        prefs = self.get_preferences(user_id)
        trips = self.get_past_trips(user_id)
        lines = []
        if prefs:
            lines.append("Known traveler preferences: " + json.dumps(prefs))
        if trips:
            lines.append(f"Traveler has {len(trips)} past trip(s) on file. "
                          f"Most recent: {json.dumps(trips[0])}")
        return "\n".join(lines) if lines else "No prior memory for this traveler yet."
