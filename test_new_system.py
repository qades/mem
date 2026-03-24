#!/usr/bin/env python3
"""
Test script for new memory/context management system.
Creates test dataset and runs benchmarks.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.test_dataset import create_test_dataset, load_test_dataset
from memory_stores.muninndb import MuninnDBStore
from memory_stores.trustgraph import TrustGraphStore
from context_managers.openai_parser import (
    OpenAIContextParser,
    OpenAICompatibleContextManager,
)
from benchmark.harness import BenchmarkHarness, BenchmarkConfig, ContextManagerType
from config.manager import ConfigManager


def test_muninndb():
    """Test MuninnDB memory store."""
    print("\n" + "=" * 60)
    print("Testing MuninnDB Memory Store")
    print("=" * 60)

    try:
        store = MuninnDBStore(api_url="http://localhost:8000", vault="test")

        # Test basic operations
        mem_id = store.insert(
            entity="Python",
            relation="programming_language",
            value="popular",
            metadata={"timestamp": "2026-03-24T00:00:00", "confidence": 0.9},
        )
        print(f"✓ Inserted memory: {mem_id}")

        results = store.retrieve("Python programming", k=3)
        print(f"✓ Retrieved {len(results)} results")

        stats = store.get_stats()
        print(f"✓ Stats: {stats}")

        store.clear()
        print("✓ Cleared store")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_trustgraph():
    """Test TrustGraph memory store."""
    print("\n" + "=" * 60)
    print("Testing TrustGraph Memory Store")
    print("=" * 60)

    try:
        store = TrustGraphStore(
            api_url="http://localhost:3000", vault="test", enable_benchmarking=True
        )

        mem_id = store.insert(
            entity="TrustGraph",
            relation="memory_system",
            value="benchmarkable",
            metadata={"timestamp": "2026-03-24T00:00:00", "confidence": 0.9},
        )
        print(f"✓ Inserted memory: {mem_id}")

        results = store.retrieve("TrustGraph memory", k=3)
        print(f"✓ Retrieved {len(results)} results")

        stats = store.get_stats()
        print(f"✓ Stats: {stats}")

        benchmark = store.get_benchmark_summary()
        print(f"✓ Benchmark: {benchmark}")

        store.clear()
        print("✓ Cleared store")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_openai_parser():
    """Test OpenAI context parser."""
    print("\n" + "=" * 60)
    print("Testing OpenAI Context Parser")
    print("=" * 60)

    try:
        parser = OpenAIContextParser(
            api_url="http://localhost:8000",
            model="gpt-3.5-turbo",
        )

        result = parser.parse_message("Hello, I love Python and data science!")
        print(f"✓ Parsed message: {result}")

        messages = [
            {"role": "user", "content": "I like machine learning"},
            {"role": "user", "content": "Python is great for AI"},
        ]

        batch_result = parser.batch_parse_messages(messages)
        print(f"✓ Batch parsed {len(batch_result)} messages")

        summary = parser.extract_context_summary(messages, k=2)
        print(f"✓ Summary: {summary}")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_context_manager():
    """Test OpenAI compatible context manager."""
    print("\n" + "=" * 60)
    print("Testing OpenAI Compatible Context Manager")
    print("=" * 60)

    try:
        manager = OpenAICompatibleContextManager(
            api_url="http://localhost:8000",
            model="gpt-3.5-turbo",
            k_retrieval=3,
        )

        messages = [
            {"role": "user", "content": "Hello, I'm learning Python"},
            {"role": "assistant", "content": "Python is a great language!"},
            {"role": "user", "content": "I want to learn about data science"},
        ]

        for msg in messages:
            manager.process_message(msg)

        context = manager.get_context({"role": "user", "content": "Python data"})
        print(f"✓ Context: {context[:100]}...")

        summary = manager.get_benchmark_summary()
        print(f"✓ Benchmark: {summary}")

        manager.reset()
        print("✓ Reset manager")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def run_full_benchmark():
    """Run full benchmark with test dataset."""
    print("\n" + "=" * 60)
    print("Running Full Benchmark")
    print("=" * 60)

    # Create test dataset
    create_test_dataset()
    messages, references = load_test_dataset()

    print(f"Loaded {len(messages)} messages")

    # Test with each strategy
    strategies = [
        ContextManagerType.BASELINE,
        ContextManagerType.VECTOR_DB,
        ContextManagerType.KNOWLEDGE_GRAPH,
    ]

    for strategy in strategies:
        print(f"\nTesting {strategy.value}...")

        config = BenchmarkConfig(
            context_manager_type=strategy,
            dataset_name="test_dataset",
            max_messages=20,
            use_embeddings=strategy != ContextManagerType.KNOWLEDGE_GRAPH,
            enable_metrics=False,
        )

        harness = BenchmarkHarness(config)
        result = harness.run_benchmark(messages[:20], references[:20])

        print(f"  Context size: {result.context_size} tokens")
        print(f"  Response time: {result.response_time_ms:.2f}ms")
        print(f"  Memory usage: {result.memory_usage_mb:.2f}MB")
        if result.metrics:
            print(f"  Metrics: {result.metrics}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("NEW MEMORY/CONTEXT MANAGEMENT SYSTEM - TEST SUITE")
    print("=" * 60)

    results = []

    # Test individual components
    results.append(("MuninnDB Store", test_muninndb()))
    results.append(("TrustGraph Store", test_trustgraph()))
    results.append(("OpenAI Parser", test_openai_parser()))
    results.append(("Context Manager", test_context_manager()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n✓ All tests passed!")
        run_full_benchmark()
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
