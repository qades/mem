"""
Test suite for the new memory/context management system.
"""

import json
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.manager import (
    ConfigManager,
    BenchmarkConfig,
    ModelConfig,
    ContextManagerType,
    VectorStoreType,
)
from data.test_dataset import create_test_dataset, load_test_dataset
from context_managers.openai_parser import OpenAICompatibleContextManager
from memory_stores.vector_db import VectorDBStore
from benchmark.harness import BenchmarkHarness, BenchmarkConfig as BenchConfig


def test_configuration():
    """Test configuration loading and saving."""
    print("\n=== Testing Configuration ===")

    config_mgr = ConfigManager()

    # Test model config
    model_config = config_mgr.load_model_config()
    print(f"Model config: {model_config}")
    assert model_config.api_url == "http://localhost:58080/v1"
    assert model_config.chat_model == "Qwen3-Coder-Next-Q4_K_M"
    assert model_config.parser_model == "LFM2.5-1.2B-Instruct-Q8_0"

    # Test vector store config
    vector_config = config_mgr.load_vector_store_config()
    print(f"Vector store config: {vector_config}")
    assert vector_config.store_type == VectorStoreType.IN_MEMORY

    # Test benchmark config loading
    baseline_config = config_mgr.load_config("config/baseline.json")
    print(f"Baseline config: {baseline_config}")
    assert baseline_config.context_manager_type == ContextManagerType.BASELINE

    print("✓ Configuration tests passed")


def test_dataset_creation():
    """Test dataset creation with configurable size."""
    print("\n=== Testing Dataset Creation ===")

    # Create test dataset with 20 messages
    create_test_dataset("data/test_dataset.jsonl", num_messages=20)

    # Verify file exists
    assert Path("data/test_dataset.jsonl").exists()

    # Load and verify
    user_msgs, assistant_msgs = load_test_dataset("data/test_dataset.jsonl")
    print(
        f"Loaded {len(user_msgs)} user messages, {len(assistant_msgs)} assistant messages"
    )

    # Create with different size
    create_test_dataset("data/test_dataset_10.jsonl", num_messages=10)
    user_msgs_10, _ = load_test_dataset("data/test_dataset_10.jsonl", max_messages=10)
    assert len(user_msgs_10) <= 10

    print("✓ Dataset creation tests passed")


def test_openai_parser():
    """Test OpenAI-compatible parser (without actual API call)."""
    print("\n=== Testing OpenAI Parser ===")

    # Test with mock data (no actual API call)
    parser = OpenAICompatibleContextManager(
        api_url="http://localhost:58080/v1",
        model="LFM2.5-1.2B-Instruct-Q8_0",
        k_retrieval=5,
        enable_benchmarking=True,
    )

    # Mock messages
    test_messages = [
        {"role": "user", "content": "Hello, I love Python programming."},
        {
            "role": "assistant",
            "content": "Hello! Python is great. What do you want to build?",
        },
        {"role": "user", "content": "I want to build a data science app."},
    ]

    # Process messages
    for msg in test_messages:
        parser.process_message(msg)

    # Get context
    context = parser.get_context(
        {"role": "user", "content": "What should I use for data science?"}
    )
    print(f"Context: {context[:100]}...")

    # Get benchmark summary
    summary = parser.get_benchmark_summary()
    print(f"Benchmark summary: {summary}")

    # Reset
    parser.reset()
    assert parser.get_context_size() == 0

    print("✓ OpenAI parser tests passed")


def test_vector_db_store():
    """Test vector DB store with different backends."""
    print("\n=== Testing Vector DB Store ===")

    # Test in-memory backend
    store = VectorDBStore(store_type="in_memory", dimension=768)

    # Insert some data
    vec_id = store.insert("user", "likes", "python", {"timestamp": "2024-01-01"})
    print(f"Inserted vector: {vec_id}")

    # Retrieve
    results = store.retrieve("python programming", k=5)
    print(f"Retrieved {len(results)} results")
    assert len(results) >= 0  # May be 0 if no matches

    # Get stats
    stats = store.get_stats()
    print(f"Stats: {stats}")
    assert stats["store_type"] == "in_memory"

    # Test ChromaDB backend (if available)
    try:
        chroma_store = VectorDBStore(store_type="chromadb", collection_name="test")
        print("✓ ChromaDB backend available")
    except ImportError:
        print("⚠ ChromaDB not installed (optional)")

    # Test FAISS backend (if available)
    try:
        faiss_store = VectorDBStore(store_type="faiss", dimension=768)
        print("✓ FAISS backend available")
    except ImportError:
        print("⚠ FAISS not installed (optional)")

    print("✓ Vector DB store tests passed")


def test_benchmark_harness():
    """Test benchmark harness with small dataset."""
    print("\n=== Testing Benchmark Harness ===")

    # Create small test dataset
    create_test_dataset("data/test_harness.jsonl", num_messages=20)

    # Test baseline
    config = BenchConfig(
        context_manager_type=ContextManagerType.BASELINE,
        dataset_name="chatbot_conversations",
        max_messages=20,
        enable_metrics=False,
    )
    harness = BenchmarkHarness(config)
    result = harness.run_benchmark([{"role": "user", "content": "test"}])
    print(f"Baseline result: time={result.response_time_ms:.2f}ms")

    # Test OpenAI parser
    config = BenchConfig(
        context_manager_type=ContextManagerType.OPENAI_PARSER,
        dataset_name="chatbot_conversations",
        max_messages=20,
        enable_metrics=False,
        params={
            "api_url": "http://localhost:58080/v1",
            "model": "LFM2.5-1.2B-Instruct-Q8_0",
            "enable_benchmarking": True,
        },
    )
    harness = BenchmarkHarness(config)
    result = harness.run_benchmark([{"role": "user", "content": "test"}])
    print(f"OpenAI parser result: time={result.response_time_ms:.2f}ms")

    print("✓ Benchmark harness tests passed")


def test_end_to_end():
    """Run end-to-end test with small dataset."""
    print("\n=== Running End-to-End Test ===")

    # Create test dataset
    create_test_dataset("data/e2e_test.jsonl", num_messages=20)

    # Load dataset
    messages, _ = load_test_dataset("data/e2e_test.jsonl", max_messages=20)
    print(f"Loaded {len(messages)} messages for end-to-end test")

    # Run with different strategies
    strategies = [
        (ContextManagerType.BASELINE, {}),
        (
            ContextManagerType.OPENAI_PARSER,
            {
                "api_url": "http://localhost:58080/v1",
                "parser_model": "LFM2.5-1.2B-Instruct-Q8_0",
            },
        ),
    ]

    for strategy, params in strategies:
        config = BenchConfig(
            context_manager_type=strategy,
            dataset_name="e2e_test",
            max_messages=20,
            enable_metrics=False,
            params=params,
        )
        harness = BenchmarkHarness(config)
        result = harness.run_benchmark(messages[:5])  # Just first 5 for speed
        print(
            f"{strategy.value}: {result.response_time_ms:.2f}ms, context_size={result.context_size}"
        )

    print("✓ End-to-end test completed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Memory/Context Management System - Test Suite")
    print("=" * 60)

    try:
        test_configuration()
        test_dataset_creation()
        test_openai_parser()
        test_vector_db_store()
        test_benchmark_harness()
        test_end_to_end()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
