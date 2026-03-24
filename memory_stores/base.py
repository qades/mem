"""
Base memory store interface.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseMemoryStore(ABC):
    """Interface for memory storage backends."""

    @abstractmethod
    def insert(
        self,
        entity: str,
        relation: str,
        value: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Insert a piece of information into memory."""
        pass

    @abstractmethod
    def retrieve(
        self, query: str, k: int = 5, use_embedding: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant information from memory."""
        pass

    @abstractmethod
    def update(
        self,
        id: str,
        entity: Optional[str] = None,
        relation: Optional[str] = None,
        value: Optional[str] = None,
    ) -> bool:
        """Update existing information."""
        pass

    @abstractmethod
    def delete(self, id: str) -> bool:
        """Delete information from memory."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all memory."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get memory store statistics."""
        pass
