"""
SQLite database helpers for the Attendance Tracker.
Stores face encodings (embedding vectors) as JSON – NO images are saved.
Includes helpers for streaks, heatmaps, CSV export, and late-status tracking.
"""

import sqlite3
import json
import csv
import io
import os
from datetime import datetime, date, timedelta

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attendance.db")

# ─── Configurable check-in deadline (24-hour format) ─────────────────────────
CHECK_IN_DEADLINE = "09:30"  # Anyone marking attendance after this is considered "Late"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist, and migrate existing tables."""
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
            liveness_verified INTEGER NOT NULL DEFAULT 0,
            late_status TEXT NOT NULL DEFAULT 'On Time',
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    # Migrate: add columns if missing (for existing DBs)
    try:
        cursor.execute("ALTER TABLE attendance ADD COLUMN liveness_verified INTEGER NOT NULL DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # column already exists
    try:
        cursor.execute("ALTER TABLE attendance ADD COLUMN late_status TEXT NOT NULL DEFAULT 'On Time'")
    except sqlite3.OperationalError:
        pass
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


def _compute_late_status(timestamp_iso: str) -> str:
    """Determine if the attendance time is before or after the deadline."""
    try:
        ts = datetime.fromisoformat(timestamp_iso)
        deadline_h, deadline_m = map(int, CHECK_IN_DEADLINE.split(":"))
        deadline_time = ts.replace(hour=deadline_h, minute=deadline_m, second=0, microsecond=0)
        return "Late" if ts > deadline_time else "On Time"
    except Exception:
        return "On Time"


HOLIDAYS = [
    "2026-01-01",  # New Year
    "2026-01-26",  # Republic Day
    "2026-08-15",  # Independence Day
    "2026-10-02",  # Gandhi Jayanti
    "2026-12-25",  # Christmas
]


def record_attendance(user_id: int, name: str, status: str = "Present",
                      liveness_verified: bool = False) -> int:
    """Record an attendance entry. Prevents duplicate entries within 60 seconds.
    Auto-computes late_status based on CHECK_IN_DEADLINE.
    Returns -2 if today is Sunday or a holiday."""
    now = datetime.now()
    if now.weekday() == 6:  # Sunday
        return -2
    if now.strftime("%Y-%m-%d") in HOLIDAYS:
        return -2

    conn = get_connection()
    cursor = conn.cursor()
    # Check for recent duplicate (within last 60 seconds)
    # Use Python's datetime for comparison to avoid SQLite UTC vs local mismatch
    sixty_seconds_ago = (datetime.now() - timedelta(seconds=60)).isoformat()
    cursor.execute(
        """SELECT id FROM attendance
           WHERE user_id = ? AND timestamp > ?""",
        (user_id, sixty_seconds_ago),
    )
    if cursor.fetchone():
        conn.close()
        return -1  # Already recorded recently

    now_iso = datetime.now().isoformat()
    late = _compute_late_status(now_iso)
    cursor.execute(
        "INSERT INTO attendance (user_id, name, timestamp, status, liveness_verified, late_status) VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, name, now_iso, status, 1 if liveness_verified else 0, late),
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
            "SELECT id, user_id, name, timestamp, status, liveness_verified, late_status FROM attendance WHERE date(timestamp) = ? ORDER BY timestamp DESC",
            (filter_date,),
        )
    else:
        cursor.execute(
            "SELECT id, user_id, name, timestamp, status, liveness_verified, late_status FROM attendance ORDER BY timestamp DESC"
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
            "liveness_verified": bool(r["liveness_verified"]),
            "late_status": r["late_status"],
        }
        for r in rows
    ]


def get_user_streak(user_id: int) -> int:
    """Calculate consecutive attendance days ending today (or yesterday) for a user."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT DISTINCT date(timestamp) as d FROM attendance WHERE user_id = ? ORDER BY d DESC",
        (user_id,),
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return 0

    streak = 0
    check_date = date.today()
    # Allow starting from today or yesterday
    first_date = date.fromisoformat(rows[0]["d"])
    if first_date != check_date and first_date != check_date - timedelta(days=1):
        return 0
    check_date = first_date

    for row in rows:
        row_date = date.fromisoformat(row["d"])
        if row_date == check_date:
            streak += 1
            check_date -= timedelta(days=1)
        else:
            break
    return streak


def get_weekly_heatmap() -> list[dict]:
    """Return attendance count for each day of the current week (Mon–Sun)."""
    today = date.today()
    # Monday of this week
    monday = today - timedelta(days=today.weekday())
    result = []
    conn = get_connection()
    cursor = conn.cursor()
    for i in range(7):
        day = monday + timedelta(days=i)
        day_str = day.isoformat()
        cursor.execute(
            "SELECT COUNT(DISTINCT user_id) as cnt FROM attendance WHERE date(timestamp) = ?",
            (day_str,),
        )
        count = cursor.fetchone()["cnt"]
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        result.append({
            "day": day_names[i],
            "date": day_str,
            "count": count,
            "is_today": day == today,
        })
    conn.close()
    return result


def export_attendance_csv(filter_date: str | None = None) -> str:
    """Generate a CSV string for attendance records."""
    records = get_attendance(filter_date)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["#", "Name", "Date", "Time", "Status", "Late Status", "Liveness Verified"])
    for i, r in enumerate(records, 1):
        try:
            dt = datetime.fromisoformat(r["timestamp"])
            date_str = dt.strftime("%Y-%m-%d")
            time_str = dt.strftime("%H:%M:%S")
        except Exception:
            date_str = r["timestamp"]
            time_str = ""
        writer.writerow([
            i, r["name"], date_str, time_str, r["status"],
            r["late_status"], "Yes" if r["liveness_verified"] else "No"
        ])
    return output.getvalue()


def get_stats() -> dict:
    """Return dashboard statistics including top streak."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as cnt FROM users")
    total_users = cursor.fetchone()["cnt"]
    today_str = date.today().isoformat()
    cursor.execute(
        "SELECT COUNT(*) as cnt FROM attendance WHERE date(timestamp) = ?",
        (today_str,),
    )
    today_attendance = cursor.fetchone()["cnt"]
    cursor.execute("SELECT COUNT(*) as cnt FROM attendance")
    total_records = cursor.fetchone()["cnt"]

    # Best streak across all users
    cursor.execute("SELECT DISTINCT id FROM users")
    user_ids = [r["id"] for r in cursor.fetchall()]
    conn.close()

    best_streak = 0
    best_streak_name = ""
    for uid in user_ids:
        s = get_user_streak(uid)
        if s > best_streak:
            best_streak = s
            # get user name
            c2 = get_connection()
            cu2 = c2.cursor()
            cu2.execute("SELECT name FROM users WHERE id = ?", (uid,))
            row = cu2.fetchone()
            c2.close()
            best_streak_name = row["name"] if row else ""

    return {
        "total_users": total_users,
        "today_attendance": today_attendance,
        "total_records": total_records,
        "best_streak": best_streak,
        "best_streak_name": best_streak_name,
        "check_in_deadline": CHECK_IN_DEADLINE,
    }
