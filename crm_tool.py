"""
crm_tool.py
FreshMart Customer Relationship Management Tool

Stores and retrieves user information across sessions using SQLite.
The LLM calls this tool via the tool orchestrator.

Supported operations:
    get_user_info(user_id)                        -> dict
    update_user_info(user_id, field, value)       -> dict
    store_interaction(user_id, summary)           -> dict
    get_interaction_history(user_id, limit)       -> dict
"""

import sqlite3
import json
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

DB_PATH = "crm.db"


def _get_connection() -> sqlite3.Connection:
    """Get a SQLite connection with row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    """Create tables if they don't exist."""
    conn = _get_connection()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                user_id     TEXT PRIMARY KEY,
                name        TEXT,
                phone       TEXT,
                email       TEXT,
                address     TEXT,
                preferences TEXT DEFAULT '{}',
                created_at  TEXT,
                updated_at  TEXT
            );

            CREATE TABLE IF NOT EXISTS interactions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     TEXT NOT NULL,
                summary     TEXT NOT NULL,
                timestamp   TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            );
        """)
        conn.commit()
    finally:
        conn.close()


# Initialize DB on import
_init_db()


# ── Tool Functions ────────────────────────────────────────────────────────────

def get_user_info(user_id: str) -> dict:
    """
    Retrieve stored information about a user.

    Args:
        user_id: The session ID or unique user identifier.

    Returns:
        dict with user info, or empty profile if user is new.
    """
    if not user_id:
        return {"error": "user_id is required"}

    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()

        if row is None:
            return {
                "status": "new_user",
                "user_id": user_id,
                "name": None,
                "phone": None,
                "email": None,
                "address": None,
                "preferences": {},
                "message": "No existing profile found for this user."
            }

        prefs = {}
        try:
            prefs = json.loads(row["preferences"] or "{}")
        except Exception:
            pass

        return {
            "status": "returning_user",
            "user_id": user_id,
            "name": row["name"],
            "phone": row["phone"],
            "email": row["email"],
            "address": row["address"],
            "preferences": prefs,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "message": f"Welcome back{', ' + row['name'] if row['name'] else ''}!"
        }
    except Exception as e:
        logger.error(f"[CRM] get_user_info error: {e}")
        return {"error": str(e)}
    finally:
        conn.close()


def update_user_info(user_id: str, field: str, value: str) -> dict:
    """
    Update a specific field in the user's profile.

    Args:
        user_id: The session ID or unique user identifier.
        field:   Field to update — one of: name, phone, email, address, preferences
        value:   New value for the field.

    Returns:
        dict confirming the update.
    """
    ALLOWED_FIELDS = {"name", "phone", "email", "address", "preferences"}

    if not user_id:
        return {"error": "user_id is required"}
    if field not in ALLOWED_FIELDS:
        return {"error": f"Invalid field '{field}'. Allowed: {', '.join(ALLOWED_FIELDS)}"}

    conn = _get_connection()
    try:
        now = datetime.utcnow().isoformat()

        # Check if user exists
        existing = conn.execute(
            "SELECT user_id FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()

        if existing is None:
            # Create new user record
            conn.execute(
                """INSERT INTO users (user_id, name, phone, email, address, preferences, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, '{}', ?, ?)""",
                (user_id, None, None, None, None, now, now)
            )

        # Handle preferences as JSON
        if field == "preferences":
            row = conn.execute(
                "SELECT preferences FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
            current_prefs = {}
            try:
                current_prefs = json.loads(row["preferences"] or "{}")
            except Exception:
                pass

            # value can be a JSON string like '{"dietary": "vegan"}'
            try:
                new_prefs = json.loads(value) if isinstance(value, str) else value
                current_prefs.update(new_prefs)
                value = json.dumps(current_prefs)
            except Exception:
                # Treat as a simple string preference
                current_prefs["note"] = value
                value = json.dumps(current_prefs)

        conn.execute(
            f"UPDATE users SET {field} = ?, updated_at = ? WHERE user_id = ?",
            (value, now, user_id)
        )
        conn.commit()

        return {
            "status": "updated",
            "user_id": user_id,
            "field": field,
            "value": value,
            "message": f"Successfully updated {field} for user."
        }
    except Exception as e:
        logger.error(f"[CRM] update_user_info error: {e}")
        return {"error": str(e)}
    finally:
        conn.close()


def store_interaction(user_id: str, summary: str) -> dict:
    """
    Store a summary of the current interaction for future reference.

    Args:
        user_id: The session ID or unique user identifier.
        summary: A brief summary of what the user did/ordered/asked.

    Returns:
        dict confirming the interaction was stored.
    """
    if not user_id or not summary:
        return {"error": "user_id and summary are required"}

    conn = _get_connection()
    try:
        now = datetime.utcnow().isoformat()

        # Ensure user exists
        existing = conn.execute(
            "SELECT user_id FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
        if existing is None:
            conn.execute(
                """INSERT INTO users (user_id, name, phone, email, address, preferences, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, '{}', ?, ?)""",
                (user_id, None, None, None, None, now, now)
            )

        conn.execute(
            "INSERT INTO interactions (user_id, summary, timestamp) VALUES (?, ?, ?)",
            (user_id, summary, now)
        )
        conn.commit()

        return {
            "status": "stored",
            "user_id": user_id,
            "message": "Interaction recorded successfully."
        }
    except Exception as e:
        logger.error(f"[CRM] store_interaction error: {e}")
        return {"error": str(e)}
    finally:
        conn.close()


def get_interaction_history(user_id: str, limit: int = 5) -> dict:
    """
    Retrieve recent interaction history for a user.

    Args:
        user_id: The session ID or unique user identifier.
        limit:   Maximum number of recent interactions to return (default 5).

    Returns:
        dict with list of recent interactions.
    """
    if not user_id:
        return {"error": "user_id is required"}

    conn = _get_connection()
    try:
        rows = conn.execute(
            """SELECT summary, timestamp FROM interactions
               WHERE user_id = ?
               ORDER BY timestamp DESC
               LIMIT ?""",
            (user_id, limit)
        ).fetchall()

        interactions = [
            {"summary": row["summary"], "timestamp": row["timestamp"]}
            for row in rows
        ]

        return {
            "status": "ok",
            "user_id": user_id,
            "count": len(interactions),
            "interactions": interactions
        }
    except Exception as e:
        logger.error(f"[CRM] get_interaction_history error: {e}")
        return {"error": str(e)}
    finally:
        conn.close()


# ── Tool Schema (for LLM tool calling) ───────────────────────────────────────

CRM_TOOL_SCHEMAS = [
    {
        "name": "get_user_info",
        "description": "Retrieve stored profile information about a user. Call this when a returning user greets you or says they are back, to personalize the response.",
        "parameters": {
            "user_id": {"type": "string", "description": "The session or user ID"}
        },
        "required": ["user_id"]
    },
    {
        "name": "update_user_info",
        "description": "Update a user's profile field. Call this when the user provides their name, phone, email, address, or dietary preferences.",
        "parameters": {
            "user_id": {"type": "string", "description": "The session or user ID"},
            "field":   {"type": "string", "description": "Field to update: name, phone, email, address, or preferences"},
            "value":   {"type": "string", "description": "New value for the field"}
        },
        "required": ["user_id", "field", "value"]
    },
    {
        "name": "store_interaction",
        "description": "Save a summary of this conversation for future reference. Call this at the end of a session or after a completed order.",
        "parameters": {
            "user_id": {"type": "string", "description": "The session or user ID"},
            "summary": {"type": "string", "description": "Brief summary of what the user ordered or asked about"}
        },
        "required": ["user_id", "summary"]
    },
    {
        "name": "get_interaction_history",
        "description": "Retrieve a user's past interaction history. Call this when the user asks about previous orders or mentions being a repeat customer.",
        "parameters": {
            "user_id": {"type": "string",  "description": "The session or user ID"},
            "limit":   {"type": "integer", "description": "Number of recent interactions to retrieve (default 5)"}
        },
        "required": ["user_id"]
    }
]


if __name__ == "__main__":
    # Quick test
    print("Testing CRM tool...")
    test_id = "test_session_001"

    print("\n1. Get user info (new user):")
    print(get_user_info(test_id))

    print("\n2. Update name:")
    print(update_user_info(test_id, "name", "Afroz"))

    print("\n3. Update preferences:")
    print(update_user_info(test_id, "preferences", '{"dietary": "vegetarian"}'))

    print("\n4. Store interaction:")
    print(store_interaction(test_id, "User ordered 2 apples and 1 mango. Total $8.50."))

    print("\n5. Get user info (returning user):")
    print(get_user_info(test_id))

    print("\n6. Get interaction history:")
    print(get_interaction_history(test_id))

    print("\nCRM test complete!")
