"""
SQLite database helpers for the Attendance Tracker.
Stores face encodings (128-d vectors) as JSON – NO images are saved.
"""

import sqlite3
import json
import os
from datetime import datetime, date

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attendance.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            encoding TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Present',
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()


def add_user(name: str, encoding: list) -> int:
    """Register a new user with their face encoding. Returns the new user id."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO users (name, encoding, created_at) VALUES (?, ?, ?)",
        (name, json.dumps(encoding), datetime.now().isoformat()),
    )
    conn.commit()
    user_id = cursor.lastrowid
    conn.close()
    return user_id


def get_all_users() -> list[dict]:
    """Return all registered users with their encodings."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, encoding, created_at FROM users")
    rows = cursor.fetchall()
    conn.close()
    users = []
    for row in rows:
        users.append({
            "id": row["id"],
            "name": row["name"],
            "encoding": json.loads(row["encoding"]),
            "created_at": row["created_at"],
        })
    return users


def get_users_list() -> list[dict]:
    """Return users without encodings (for the frontend list)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, created_at FROM users")
    rows = cursor.fetchall()
    conn.close()
    return [{"id": r["id"], "name": r["name"], "created_at": r["created_at"]} for r in rows]


def delete_user(user_id: int) -> bool:
    """Delete a user by id. Returns True if a row was deleted."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
    deleted = cursor.rowcount > 0
    if deleted:
        cursor.execute("DELETE FROM attendance WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()
    return deleted


def record_attendance(user_id: int, name: str, status: str = "Present") -> int:
    """Record an attendance entry. Prevents duplicate entries within 60 seconds."""
    conn = get_connection()
    cursor = conn.cursor()
    # Check for recent duplicate (within last 60 seconds)
    cursor.execute(
        """SELECT id FROM attendance
           WHERE user_id = ? AND timestamp > datetime('now', '-60 seconds')""",
        (user_id,),
    )
    if cursor.fetchone():
        conn.close()
        return -1  # Already recorded recently
    cursor.execute(
        "INSERT INTO attendance (user_id, name, timestamp, status) VALUES (?, ?, ?, ?)",
        (user_id, name, datetime.now().isoformat(), status),
    )
    conn.commit()
    att_id = cursor.lastrowid
    conn.close()
    return att_id


def get_attendance(filter_date: str | None = None) -> list[dict]:
    """Retrieve attendance records, optionally filtered by date (YYYY-MM-DD)."""
    conn = get_connection()
    cursor = conn.cursor()
    if filter_date:
        cursor.execute(
            "SELECT id, user_id, name, timestamp, status FROM attendance WHERE date(timestamp) = ? ORDER BY timestamp DESC",
            (filter_date,),
        )
    else:
        cursor.execute(
            "SELECT id, user_id, name, timestamp, status FROM attendance ORDER BY timestamp DESC"
        )
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "id": r["id"],
            "user_id": r["user_id"],
            "name": r["name"],
            "timestamp": r["timestamp"],
            "status": r["status"],
        }
        for r in rows
    ]


def get_stats() -> dict:
    """Return dashboard statistics."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as cnt FROM users")
    total_users = cursor.fetchone()["cnt"]
    today = date.today().isoformat()
    cursor.execute(
        "SELECT COUNT(*) as cnt FROM attendance WHERE date(timestamp) = ?",
        (today,),
    )
    today_attendance = cursor.fetchone()["cnt"]
    cursor.execute("SELECT COUNT(*) as cnt FROM attendance")
    total_records = cursor.fetchone()["cnt"]
    conn.close()
    return {
        "total_users": total_users,
        "today_attendance": today_attendance,
        "total_records": total_records,
    }
