"""In-memory multi-turn history keyed by conversation_id."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class ConversationStore:
    """Process-local conversation buffer for ask_question follow-ups."""

    _threads: dict[str, list[dict[str, str]]] = field(default_factory=dict)

    def get_history(self, conversation_id: str | None) -> list[dict[str, str]]:
        if not conversation_id:
            return []
        return list(self._threads.get(conversation_id, []))

    def append_turn(
        self,
        conversation_id: str | None,
        *,
        user_message: str,
        assistant_message: str,
    ) -> str:
        cid = conversation_id or str(uuid.uuid4())
        thread = self._threads.setdefault(cid, [])
        thread.append({"role": "user", "content": user_message})
        thread.append({"role": "assistant", "content": assistant_message})
        return cid


conversation_store = ConversationStore()
