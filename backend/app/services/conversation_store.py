import json
import sqlite3
from collections.abc import Sequence
from uuid import uuid4

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.core.config import BACKEND_DIR

DB_PATH = BACKEND_DIR / "chat_history.db"


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT 'Yeni sohbet',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                turn_id TEXT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources_json TEXT NOT NULL DEFAULT '[]',
                attachments_json TEXT NOT NULL DEFAULT '[]',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_tool_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                turn_id TEXT,
                name TEXT NOT NULL,
                input_json TEXT NOT NULL,
                output_json TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        _ensure_column(connection, "messages", "turn_id", "TEXT")
        _ensure_column(connection, "messages", "sources_json", "TEXT NOT NULL DEFAULT '[]'")
        _ensure_column(connection, "messages", "attachments_json", "TEXT NOT NULL DEFAULT '[]'")
        _ensure_column(connection, "conversation_tool_calls", "turn_id", "TEXT")
        connection.execute(
            """
            INSERT OR IGNORE INTO conversations (
                conversation_id,
                title,
                created_at,
                updated_at
            )
            SELECT
                conversation_id,
                COALESCE(
                    NULLIF(SUBSTR(MIN(CASE WHEN role IN ('user', 'human') THEN content END), 1, 60), ''),
                    'Eski sohbet'
                ),
                MIN(created_at),
                MAX(created_at)
            FROM messages
            GROUP BY conversation_id
            """
        )
        connection.execute(
            """
            UPDATE conversations
            SET title = COALESCE(
                (
                    SELECT SUBSTR(content, 1, 60)
                    FROM messages
                    WHERE messages.conversation_id = conversations.conversation_id
                      AND role IN ('user', 'human')
                    ORDER BY id ASC
                    LIMIT 1
                ),
                title
            )
            WHERE EXISTS (
                SELECT 1
                FROM messages
                WHERE messages.conversation_id = conversations.conversation_id
            )
            """
        )


def create_conversation() -> dict:
    init_db()
    conversation_id = f"chat_{uuid4().hex}"

    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute(
            "INSERT INTO conversations (conversation_id) VALUES (?)",
            (conversation_id,),
        )
        row = connection.execute(
            "SELECT * FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()

    return dict(row)


def ensure_conversation(conversation_id: str, title_candidate: str | None = None) -> None:
    init_db()
    title = _conversation_title(title_candidate) if title_candidate else "Yeni sohbet"

    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO conversations (conversation_id, title)
            VALUES (?, ?)
            """,
            (conversation_id, title),
        )
        if title_candidate:
            connection.execute(
                """
                UPDATE conversations
                SET title = CASE WHEN title = 'Yeni sohbet' THEN ? ELSE title END,
                    updated_at = CURRENT_TIMESTAMP
                WHERE conversation_id = ?
                """,
                (title, conversation_id),
            )


def list_conversations() -> list[dict]:
    init_db()

    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT conversation_id, title, created_at, updated_at
            FROM conversations
            ORDER BY updated_at DESC, rowid DESC
            """
        ).fetchall()

    return [dict(row) for row in rows]


def list_history_messages(conversation_id: str) -> list[dict]:
    init_db()

    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT role, content, created_at, turn_id, sources_json, attachments_json
            FROM messages
            WHERE conversation_id = ? AND role IN ('user', 'assistant')
            ORDER BY id ASC
            """,
            (conversation_id,),
        ).fetchall()

    return [
        {
            **dict(row),
            "sources": json.loads(row["sources_json"] or "[]"),
            "attachments": json.loads(row["attachments_json"] or "[]"),
        }
        for row in rows
    ]


def conversation_exists(conversation_id: str) -> bool:
    init_db()

    with sqlite3.connect(DB_PATH) as connection:
        row = connection.execute(
            "SELECT 1 FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()

    return row is not None


def delete_conversation(conversation_id: str) -> bool:
    init_db()

    with sqlite3.connect(DB_PATH) as connection:
        exists = connection.execute(
            "SELECT 1 FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        if exists is None:
            return False

        connection.execute(
            "DELETE FROM conversation_tool_calls WHERE conversation_id = ?",
            (conversation_id,),
        )
        connection.execute(
            "DELETE FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        )
        connection.execute(
            "DELETE FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        )

    return True


def save_tool_calls(
    conversation_id: str,
    tool_calls: Sequence[object],
    turn_id: str | None = None,
) -> None:
    if not tool_calls:
        return

    init_db()
    rows = [
        (
            conversation_id,
            getattr(tool_call, "turn_id", None) or turn_id,
            str(getattr(tool_call, "name")),
            json.dumps(getattr(tool_call, "input"), ensure_ascii=False),
            json.dumps(getattr(tool_call, "output"), ensure_ascii=False),
        )
        for tool_call in tool_calls
    ]

    with sqlite3.connect(DB_PATH) as connection:
        connection.executemany(
            """
            INSERT INTO conversation_tool_calls (
                conversation_id,
                turn_id,
                name,
                input_json,
                output_json
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )


def list_tool_calls(conversation_id: str) -> list[dict]:
    init_db()

    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, turn_id, name, input_json, output_json, created_at
            FROM conversation_tool_calls
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,),
        ).fetchall()

    return [
        {
            "id": row["id"],
            "turn_id": row["turn_id"],
            "name": row["name"],
            "input": json.loads(row["input_json"]),
            "output": json.loads(row["output_json"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def save_conversation_turn(
    conversation_id: str,
    turn_id: str,
    user_message: str,
    assistant_message: str,
    tool_calls: Sequence[object] | None = None,
    sources: Sequence[object] | None = None,
    user_attachments: Sequence[object] | None = None,
) -> None:
    init_db()
    sources_json = json.dumps(
        [_public_model_dump(source) for source in sources or []],
        ensure_ascii=False,
    )
    user_attachments_json = json.dumps(
        [_public_model_dump(attachment) for attachment in user_attachments or []],
        ensure_ascii=False,
    )

    with sqlite3.connect(DB_PATH) as connection:
        connection.executemany(
            """
            INSERT INTO messages (
                conversation_id,
                turn_id,
                role,
                content,
                sources_json,
                attachments_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (conversation_id, turn_id, "user", user_message, "[]", user_attachments_json),
                (conversation_id, turn_id, "assistant", assistant_message, sources_json, "[]"),
            ],
        )
        connection.execute(
            """
            UPDATE conversations
            SET updated_at = CURRENT_TIMESTAMP
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        )

    if tool_calls:
        save_tool_calls(conversation_id, tool_calls, turn_id=turn_id)


def _ensure_column(
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_type: str,
) -> None:
    columns = {
        row[1]
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if column_name not in columns:
        connection.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
        )


def _public_model_dump(item: object) -> dict:
    if hasattr(item, "model_dump"):
        return item.model_dump()
    if isinstance(item, dict):
        return item
    return dict(item)


def _conversation_title(message: str | None) -> str:
    compact = " ".join((message or "").split())
    return compact[:60] or "Yeni sohbet"


class SQLiteChatMessageHistory(BaseChatMessageHistory):
    """LangChain chat history backed by the application's SQLite database."""

    def __init__(self, conversation_id: str) -> None:
        self.conversation_id = conversation_id
        ensure_conversation(conversation_id)

    @property
    def messages(self) -> list[BaseMessage]:
        with sqlite3.connect(DB_PATH) as connection:
            rows = connection.execute(
                """
                SELECT role, content
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id ASC
                """,
                (self.conversation_id,),
            ).fetchall()

        message_types = {
            "user": HumanMessage,
            "human": HumanMessage,
            "assistant": AIMessage,
            "ai": AIMessage,
            "system": SystemMessage,
        }
        return [
            message_types[role](content=content)
            for role, content in rows
            if role in message_types
        ]

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        role_names = {
            "human": "user",
            "ai": "assistant",
            "system": "system",
        }
        rows = [
            (
                self.conversation_id,
                None,
                role_names.get(message.type, message.type),
                str(message.content),
                "[]",
                "[]",
            )
            for message in messages
        ]

        if not rows:
            return

        with sqlite3.connect(DB_PATH) as connection:
            connection.executemany(
                """
                INSERT INTO messages (
                    conversation_id,
                    turn_id,
                    role,
                    content,
                    sources_json,
                    attachments_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            connection.execute(
                """
                UPDATE conversations
                SET updated_at = CURRENT_TIMESTAMP
                WHERE conversation_id = ?
                """,
                (self.conversation_id,),
            )

    def clear(self) -> None:
        with sqlite3.connect(DB_PATH) as connection:
            connection.execute(
                "DELETE FROM messages WHERE conversation_id = ?",
                (self.conversation_id,),
            )


def get_chat_history(conversation_id: str) -> BaseChatMessageHistory:
    return SQLiteChatMessageHistory(conversation_id)
