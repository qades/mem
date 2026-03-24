"""
Baseline context manager - sends all previous messages with current one.
"""

from typing import List, Dict, Any
from .base import BaseContextManager


class BaselineContextManager(BaseContextManager):
    """Baseline strategy: keep all conversation history."""

    def __init__(self):
        self.conversation_history: List[Dict[str, str]] = []

    def process_message(self, message: Dict[str, str]) -> None:
        """Add message to conversation history."""
        self.conversation_history.append(message)

    def get_context(self, current_message: Dict[str, str]) -> str:
        """Return all conversation history plus current message."""
        context_parts = []

        for msg in self.conversation_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            context_parts.append(f"{role.upper()}: {content}")

        current_role = current_message.get("role", "user")
        current_content = current_message.get("content", "")
        context_parts.append(f"{current_role.upper()}: {current_content}")

        return "\n".join(context_parts)

    def get_context_size(self) -> int:
        """Estimate token count (rough approximation: 1 token ≈ 4 chars)."""
        total_text = "\n".join(
            [
                f"{msg.get('role', 'user').upper()}: {msg.get('content', '')}"
                for msg in self.conversation_history
            ]
        )
        return len(total_text) // 4

    def reset(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
