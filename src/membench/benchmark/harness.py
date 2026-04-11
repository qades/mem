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

from membench.context_managers.base import BaseContextManager
from membench.context_managers.baseline import BaselineContextManager
from membench.context_managers.memory_based import MemoryBasedContextManager
from membench.context_managers.openai_parser import (
    OpenAICompatibleContextManager,
    OpenAIContextParser,
)
from membench.memory_stores.base import BaseMemoryStore
from membench.memory_stores.knowledge_graph import KnowledgeGraphStore
from membench.memory_stores.vector_db import VectorDBStore
from membench.benchmark.dataset_loader import load_dataset
from membench.eval.metrics import calculate_metrics

from membench.config.manager import ContextManagerType, VectorStoreType


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
    params: Dict[str, Any] = field(default_factory=dict)


class BenchmarkHarness:
    """Main benchmark harness for comparing context management strategies."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.memory_store = self._create_memory_store()
        self.context_manager = self._create_context_manager()

    def _create_memory_store(self) -> Optional[BaseMemoryStore]:
        """Create memory store based on config."""
        store_type_str = self.config.params.get("store_type", "in_memory")
        store_type = VectorStoreType(store_type_str)

        if store_type == VectorStoreType.CHROMADB:
            return VectorDBStore(
                store_type="chromadb",
                collection_name=self.config.params.get("collection_name", "memory"),
                dimension=self.config.params.get("dimension", 768),
                api_url=self.config.params.get("api_url", "http://localhost:8000"),
                api_key=self.config.params.get("api_key"),
            )
        elif store_type == VectorStoreType.FAISS:
            return VectorDBStore(
                store_type="faiss",
                collection_name=self.config.params.get("collection_name", "memory"),
                dimension=self.config.params.get("dimension", 768),
            )
        elif self.config.context_manager_type == ContextManagerType.KNOWLEDGE_GRAPH:
            return KnowledgeGraphStore()
        elif self.config.context_manager_type == ContextManagerType.VECTOR_DB:
            return VectorDBStore(
                store_type="in_memory",
                collection_name=self.config.params.get("collection_name", "memory"),
                dimension=self.config.params.get("dimension", 768),
            )
        elif self.config.context_manager_type == ContextManagerType.MUNINNDB:
            return None  # MuninnDB handles its own storage
        elif self.config.context_manager_type == ContextManagerType.TRUSTGRAPH:
            return None  # TrustGraph handles its own storage
        return None

    def _create_context_manager(self) -> BaseContextManager:
        """Create context manager based on config."""
        if self.config.context_manager_type == ContextManagerType.BASELINE:
            return BaselineContextManager()
        elif self.config.context_manager_type == ContextManagerType.OPENAI_PARSER:
            return OpenAICompatibleContextManager(
                api_url=self.config.params.get("api_url", "http://localhost:58080/v1"),
                model=self.config.params.get(
                    "parser_model", "LFM2.5-1.2B-Instruct-Q8_0"
                ),
                k_retrieval=self.config.k_retrieval,
                enable_benchmarking=self.config.params.get("enable_benchmarking", True),
            )
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
        max_messages = self.config.max_messages or 10
        for i, message in enumerate(messages):
            if i >= max_messages:
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
            metadata={
                "k_retrieval": self.config.k_retrieval,
                "use_embeddings": self.config.use_embeddings,
            },
        )

        # Add parser-specific metadata if applicable
        if hasattr(self.context_manager, "get_benchmark_summary"):
            result.metadata["parser_summary"] = (
                self.context_manager.get_benchmark_summary()
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
