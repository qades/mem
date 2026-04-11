"""
Context managers for handling conversation context with memory.

These managers decide how to populate the context window:
- Baseline: Full conversation history
- OpenAI Parser: LLM-based extraction
- Memory-based: Use memory stores for retrieval
"""

from membench.context_managers.base import BaseContextManager
from membench.context_managers.baseline import BaselineContextManager
from membench.context_managers.openai_parser import OpenAICompatibleContextManager
from membench.context_managers.memory_based import MemoryBasedContextManager

__all__ = [
    "BaseContextManager",
    "BaselineContextManager",
    "OpenAICompatibleContextManager",
    "MemoryBasedContextManager",
]
