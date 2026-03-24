"""
Context Management Benchmark Harness

Compares baseline (full conversation history) vs memory-based context management.
"""

import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context_managers.base import BaseContextManager
from context_managers.baseline import BaselineContextManager
from context_managers.memory_based import MemoryBasedContextManager
from memory_stores.base import BaseMemoryStore
from memory_stores.knowledge_graph import KnowledgeGraphStore
from memory_stores.vector_db import VectorDBStore
from datasets.loader import load_dataset
from eval.metrics import calculate_metrics

from config.manager import ContextManagerType


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    context_manager_type: str
    dataset_name: str
    total_messages: int
    context_size: int  # tokens in final context
    response_time_ms: float
    memory_usage_mb: float
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    context_manager_type: ContextManagerType
    dataset_name: str
    max_messages: Optional[int] = None
    use_embeddings: bool = True
    k_retrieval: int = 5
    enable_metrics: bool = True


class BenchmarkHarness:
    """Main benchmark harness for comparing context management strategies."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.memory_store = self._create_memory_store()
        self.context_manager = self._create_context_manager()

    def _create_memory_store(self) -> BaseMemoryStore:
        """Create memory store based on config."""
        if self.config.context_manager_type == ContextManagerType.KNOWLEDGE_GRAPH:
            return KnowledgeGraphStore()
        elif self.config.context_manager_type == ContextManagerType.VECTOR_DB:
            return VectorDBStore()
        return None

    def _create_context_manager(self) -> BaseContextManager:
        """Create context manager based on config."""
        if self.config.context_manager_type == ContextManagerType.BASELINE:
            return BaselineContextManager()
        else:
            return MemoryBasedContextManager(
                memory_store=self.memory_store,
                use_embeddings=self.config.use_embeddings,
                k_retrieval=self.config.k_retrieval,
            )

    def run_benchmark(
        self,
        messages: List[Dict[str, str]],
        reference_answers: Optional[List[str]] = None,
    ) -> BenchmarkResult:
        """Run a complete benchmark on a conversation."""
        start_time = time.time()

        # Process each message
        for i, message in enumerate(messages):
            if self.config.max_messages and i >= self.config.max_messages:
                break

            # Parse and store in memory
            self.context_manager.process_message(message)

            # Get context for response generation
            context = self.context_manager.get_context(message)

            # Simulate response generation (in real scenario, call LLM)
            # response = llm.generate(context, message)

        end_time = time.time()

        # Calculate metrics
        metrics = {}
        if self.config.enable_metrics and reference_answers:
            # In real scenario, generate responses and compare
            metrics = calculate_metrics([], reference_answers)

        result = BenchmarkResult(
            context_manager_type=self.config.context_manager_type.value,
            dataset_name=self.config.dataset_name,
            total_messages=len(messages),
            context_size=self.context_manager.get_context_size(),
            response_time_ms=(end_time - start_time) * 1000,
            memory_usage_mb=self._get_memory_usage(),
            metrics=metrics,
        )

        return result

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        # Placeholder - implement with psutil or similar
        return 0.0

    def run_comparison(self, datasets: List[str]) -> List[BenchmarkResult]:
        """Run benchmarks across multiple datasets."""
        results = []

        for dataset_name in datasets:
            messages, references = load_dataset(dataset_name)

            result = self.run_benchmark(messages, references)
            results.append(result)

        return results

    def save_results(self, results: List[BenchmarkResult], filepath: str):
        """Save benchmark results to file."""
        with open(filepath, "w") as f:
            json.dump([r.__dict__ for r in results], f, indent=2)


def run_full_benchmark():
    """Run complete benchmark suite."""
    configs = [
        BenchmarkConfig(
            context_manager_type=ContextManagerType.BASELINE,
            dataset_name="chatbot_conversations",
        ),
        BenchmarkConfig(
            context_manager_type=ContextManagerType.KNOWLEDGE_GRAPH,
            dataset_name="chatbot_conversations",
            use_embeddings=False,
        ),
        BenchmarkConfig(
            context_manager_type=ContextManagerType.VECTOR_DB,
            dataset_name="chatbot_conversations",
            use_embeddings=True,
        ),
    ]

    all_results = []

    for config in configs:
        harness = BenchmarkHarness(config)
        results = harness.run_comparison(["chatbot_conversations"])
        all_results.extend(results)

    return all_results


if __name__ == "__main__":
    # Example usage
    results = run_full_benchmark()

    for result in results:
        print(f"\n{result.context_manager_type} Results:")
        print(f"  Context size: {result.context_size} tokens")
        print(f"  Response time: {result.response_time_ms:.2f}ms")
        print(f"  Memory usage: {result.memory_usage_mb:.2f}MB")
