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

# Suppress Pydantic deprecation warnings from external libraries (graphiti, letta)
# This must be at the very top before any imports that might trigger these warnings
import warnings
import sys

# Use simplefilter for stronger suppression
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", UserWarning)

# Also try to suppress specific Pydantic-related warnings
if sys.version_info >= (3, 0):
    warnings.filterwarnings("ignore", message=".*PydanticDeprecatedSince20.*")
    warnings.filterwarnings("ignore", message=".*json_encoders.*")
    warnings.filterwarnings("ignore", message=".*class-based.*config.*")

__version__ = "0.1.0"
__author__ = "Memory Benchmark Team"

# Core exports - these imports will trigger the external library loads
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
