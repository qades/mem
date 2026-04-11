"""
Memory-based context manager - parses messages, stores information, retrieves relevant context.
"""

from typing import List, Dict, Any, Set
from dataclasses import dataclass
from datetime import datetime

from membench.context_managers.base import BaseContextManager
from membench.memory_stores.base import BaseMemoryStore


@dataclass
class InformationBit:
    """A piece of information extracted from a message."""

    entity: str
    relation: str
    value: str
    timestamp: datetime
    confidence: float
    message_id: str


class MemoryBasedContextManager(BaseContextManager):
    """Memory-based strategy: parse, store, and retrieve relevant context."""

    def __init__(
        self,
        memory_store: BaseMemoryStore,
        use_embeddings: bool = True,
        k_retrieval: int = 5,
    ):
        self.memory_store = memory_store
        self.use_embeddings = use_embeddings
        self.k_retrieval = k_retrieval
        self.message_counter = 0
        self.stored_info: List[InformationBit] = []

    def process_message(self, message: Dict[str, str]) -> None:
        """Process message: parse, extract information, store in memory."""
        message_id = f"msg_{self.message_counter}"
        self.message_counter += 1

        content = message.get("content", "")
        role = message.get("role", "user")

        # Parse message and extract information
        info_bits = self._parse_message(content, role, message_id)

        # Store each piece of information
        for info in info_bits:
            self.memory_store.insert(
                entity=info.entity,
                relation=info.relation,
                value=info.value,
                metadata={
                    "timestamp": info.timestamp.isoformat(),
                    "confidence": info.confidence,
                    "message_id": info.message_id,
                },
            )
            self.stored_info.append(info)

        # Store message metadata
        self.memory_store.insert(
            entity="message",
            relation="content",
            value=content,
            metadata={
                "role": role,
                "message_id": message_id,
                "timestamp": datetime.now().isoformat(),
            },
        )

    def _parse_message(
        self, content: str, role: str, message_id: str
    ) -> List[InformationBit]:
        """Parse message and extract information bits (using cheap/fast LLM)."""
        # In real implementation, this would call an LLM to parse the message
        # For now, use simple heuristic extraction

        info_bits = []

        # Simple entity-relation-value extraction (placeholder for LLM parsing)
        entities = self._extract_entities(content)

        for entity in entities:
            info_bits.append(
                InformationBit(
                    entity=entity,
                    relation="mentioned_in",
                    value=message_id,
                    timestamp=datetime.now(),
                    confidence=0.8,
                    message_id=message_id,
                )
            )

        # Extract key facts (placeholder)
        if "I like" in content or "I love" in content:
            # Simulate extracting preference
            info_bits.append(
                InformationBit(
                    entity="user",
                    relation="has_preference",
                    value=content.split("I ")[-1][:50],
                    timestamp=datetime.now(),
                    confidence=0.9,
                    message_id=message_id,
                )
            )

        return info_bits

    def _extract_entities(self, content: str) -> List[str]:
        """Simple entity extraction (placeholder for LLM)."""
        # In real implementation, use NER or LLM
        entities = []

        # Simple heuristics
        words = content.split()
        for i, word in enumerate(words):
            if word.startswith("The") or word.startswith("A") or word.startswith("An"):
                if i < len(words) - 1:
                    entities.append(f"{word} {words[i + 1]}".rstrip(".,"))

        if not entities:
            entities.append("general_context")

        return entities

    def get_context(self, current_message: Dict[str, str]) -> str:
        """Get relevant context from memory for current message."""
        current_content = current_message.get("content", "")

        # Get relevant information from memory
        relevant_info = self.memory_store.retrieve(
            query=current_content, k=self.k_retrieval, use_embedding=self.use_embeddings
        )

        # Build context string
        context_parts = []

        for info in relevant_info:
            entity = info.get("entity", "unknown")
            relation = info.get("relation", "related_to")
            value = info.get("value", "")

            context_parts.append(f"{entity} {relation}: {value}")

        # Add recent conversation context if available
        recent_messages = self._get_recent_messages(3)
        if recent_messages:
            context_parts.append(f"Recent conversation:\n{recent_messages}")

        return "\n".join(context_parts) if context_parts else ""

    def _get_recent_messages(self, n: int) -> str:
        """Get recent message summaries."""
        # In real implementation, retrieve from memory store
        return ""

    def get_context_size(self) -> int:
        """Estimate size of retrieved context."""
        # This is a simplified estimate
        return self.k_retrieval * 100  # Assume ~100 tokens per retrieved item

    def reset(self) -> None:
        """Reset memory store and state."""
        self.memory_store.clear()
        self.stored_info = []
        self.message_counter = 0
