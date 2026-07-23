"""
In-memory conversation memory, keyed by session id.

This is intentionally simple: a process-local dict guarded by a lock.
It resets if the server restarts and isn't shared across multiple
backend processes — fine for a single-instance prototype. For a real
multi-user deployment, swap this for Redis or a database keyed the
same way (session_id -> list of {role, content} turns).
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field


@dataclass
class ConversationStore:
    max_turns: int = 6  # a "turn" = one user message + one assistant reply
    _sessions: dict[str, list[dict]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def new_session_id(self) -> str:
        return uuid.uuid4().hex

    def get_history(self, session_id: str) -> list[dict]:
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def append_turn(self, session_id: str, user_text: str, assistant_text: str) -> None:
        with self._lock:
            history = self._sessions.setdefault(session_id, [])
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": assistant_text})
            max_messages = self.max_turns * 2
            if len(history) > max_messages:
                del history[: len(history) - max_messages]

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
