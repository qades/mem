"""
Memory stores for long-term agent memory benchmarking.

This module provides wrappers around various memory implementations:
- Reference libraries (Mem0, Zep, Graphiti, Letta, MemPalace)
- Custom implementations (VectorDB, Knowledge Graph)
- Hybrid approaches

Each store follows the BaseMemoryStore interface for fair benchmarking.
"""

from membench.memory_stores.base import BaseMemoryStore

# Reference library implementations (use actual libraries)
try:
    from membench.memory_stores.mem0_store import Mem0Store
except ImportError:
    Mem0Store = None

try:
    from membench.memory_stores.zep_store import ZepStore
except ImportError:
    ZepStore = None

try:
    from membench.memory_stores.graphiti_store import GraphitiStore
except ImportError:
    GraphitiStore = None

try:
    from membench.memory_stores.letta_store import LettaStore
except ImportError:
    LettaStore = None

try:
    from membench.memory_stores.mempalace_store import MemPalaceStore
except ImportError:
    MemPalaceStore = None

# Existing implementations
try:
    from membench.memory_stores.muninndb import MuninnDBStore
except ImportError:
    MuninnDBStore = None

try:
    from membench.memory_stores.knowledge_graph import KnowledgeGraphStore
except ImportError:
    KnowledgeGraphStore = None

try:
    from membench.memory_stores.vector_db import VectorDBStore
except ImportError:
    VectorDBStore = None

try:
    from membench.memory_stores.trustgraph import TrustGraphStore
except ImportError:
    TrustGraphStore = None

__all__ = [
    "BaseMemoryStore",
    # Reference library implementations
    "Mem0Store",
    "ZepStore", 
    "GraphitiStore",
    "LettaStore",
    "MemPalaceStore",
    # Existing implementations
    "MuninnDBStore",
    "KnowledgeGraphStore",
    "VectorDBStore",
    "TrustGraphStore",
]

# Registry of all available stores
MEMORY_STORES = {
    "mem0": Mem0Store,
    "zep": ZepStore,
    "graphiti": GraphitiStore,
    "letta": LettaStore,
    "mempalace": MemPalaceStore,
    "muninndb": MuninnDBStore,
    "knowledge_graph": KnowledgeGraphStore,
    "vector_db": VectorDBStore,
    "trustgraph": TrustGraphStore,
}


def get_available_stores():
    """Get list of available (installed) memory stores."""
    return {
        name: store for name, store in MEMORY_STORES.items()
        if store is not None
    }


def create_store(store_type: str, **kwargs):
    """Factory function to create a memory store by type.
    
    Args:
        store_type: Type of memory store (mem0, zep, graphiti, etc.)
        **kwargs: Configuration arguments for the store
        
    Returns:
        BaseMemoryStore: Configured memory store instance
        
    Raises:
        ValueError: If store_type is unknown or not installed
    """
    store_class = MEMORY_STORES.get(store_type)
    
    if store_class is None:
        available = list(get_available_stores().keys())
        raise ValueError(
            f"Unknown or unavailable memory store: {store_type}. "
            f"Available: {available}"
        )
    
    return store_class(**kwargs)
