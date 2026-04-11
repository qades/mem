"""
Memory Benchmark Suite for Long-Term Agent Memory Evaluation

This package provides tools and infrastructure for benchmarking
various long-term memory approaches for LLM-based agents.

Usage:
    from membench import create_store, get_available_stores
    from membench.benchmark.harness import BenchmarkHarness

    # Create a memory store
    store = create_store("mem0", api_key="...")
    
    # Run benchmark
    harness = BenchmarkHarness(config)
    results = harness.run_benchmark(messages)
"""

__version__ = "0.1.0"
__author__ = "Memory Benchmark Team"

# Core exports
from membench.memory_stores import (
    BaseMemoryStore,
    create_store,
    get_available_stores,
    MEMORY_STORES,
)

# Context managers
from membench.context_managers import (
    BaseContextManager,
    BaselineContextManager,
    MemoryBasedContextManager,
    OpenAICompatibleContextManager,
)

# Benchmark
from membench.benchmark.harness import (
    BenchmarkHarness,
    BenchmarkConfig,
    BenchmarkResult,
)

__all__ = [
    # Version
    "__version__",
    # Memory stores
    "BaseMemoryStore",
    "create_store",
    "get_available_stores",
    "MEMORY_STORES",
    # Context managers
    "BaseContextManager",
    "BaselineContextManager",
    "MemoryBasedContextManager",
    "OpenAICompatibleContextManager",
    # Benchmark
    "BenchmarkHarness",
    "BenchmarkConfig",
    "BenchmarkResult",
]
