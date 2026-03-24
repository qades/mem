"""
Base context manager interface.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseContextManager(ABC):
    """Interface for context management strategies."""

    @abstractmethod
    def process_message(self, message: Dict[str, str]) -> None:
        """Process a message and update internal state."""
        pass

    @abstractmethod
    def get_context(self, current_message: Dict[str, str]) -> str:
        """Get context for the current message."""
        pass

    @abstractmethod
    def get_context_size(self) -> int:
        """Get size of current context in tokens."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the context manager state."""
        pass
