#!/usr/bin/env python3
"""
Test the new memory management system with the test dataset.
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.test_dataset import load_test_dataset
from memory_stores.muninndb import MuninnDBStore
from memory_stores.trustgraph import TrustGraphStore
from memory_stores.vector_db import VectorDBStore
from memory_stores.knowledge_graph import KnowledgeGraphStore
from context_managers.openai_parser import OpenAICompatibleContextManager
from context_managers.memory_based import MemoryBasedContextManager


def test_memory_store(store_name: str, store):
    """Test a memory store."""
    print(f"\n{'=' * 60}")
    print(f"Testing {store_name}")
    print("=" * 60)

    # Store some test data
    messages = [
        {"role": "user", "content": "I love Python programming"},
        {"role": "user", "content": "Machine learning is fascinating"},
        {"role": "user", "content": "I work with data science"},
        {"role": "user", "content": "Python has great libraries like pandas and NumPy"},
        {"role": "user", "content": "I want to learn more about AI"},
    ]

    for i, msg in enumerate(messages):
        store.insert(
            entity=f"message_{i}",
            relation="content",
            value=msg["content"][:100],
            metadata={
                "timestamp": "2026-03-24T00:00:00",
                "confidence": 0.9,
                "message": msg,
            },
        )

    # Retrieve
    results = store.retrieve("Python machine learning", k=3)
    print(f"✓ Retrieved {len(results)} results for 'Python machine learning'")

    # Get stats
    stats = store.get_stats()
    print(f"✓ Stats: {json.dumps(stats, indent=2)}")

    # Test update
    if results:
        store.update(
            results[0].get("id", results[0].get("entity", "")), value="updated"
        )
        print("✓ Updated a memory")

    # Clear
    store.clear()
    print("✓ Cleared store")

    return True


def test_context_manager():
    """Test context manager with test dataset."""
    print(f"\n{'=' * 60}")
    print("Testing Context Manager")
    print("=" * 60)

    # Load test dataset
    messages, references = load_test_dataset()
    print(f"✓ Loaded {len(messages)} test messages")

    # Test with VectorDB
    print("\nTesting with VectorDB...")
    store = VectorDBStore()
    manager = MemoryBasedContextManager(
        memory_store=store,
        use_embeddings=True,
        k_retrieval=3,
    )

    start_time = time.time()
    for msg in messages[:20]:
        manager.process_message(msg)
        context = manager.get_context(msg)
    elapsed = time.time() - start_time

    print(f"✓ Processed 20 messages in {elapsed:.2f}s")
    print(f"✓ Context size: {manager.get_context_size()} tokens")

    manager.reset()
    print("✓ Reset manager")

    return True


def test_full_pipeline():
    """Test the full pipeline with test dataset."""
    print(f"\n{'=' * 60}")
    print("Testing Full Pipeline")
    print("=" * 60)

    # Load test dataset
    messages, references = load_test_dataset()
    print(f"✓ Loaded {len(messages)} test messages")

    # Test with all memory stores
    stores = [
        ("VectorDB", VectorDBStore()),
        ("KnowledgeGraph", KnowledgeGraphStore()),
    ]

    for name, store in stores:
        print(f"\nTesting {name}...")

        manager = MemoryBasedContextManager(
            memory_store=store,
            use_embeddings=True,
            k_retrieval=3,
        )

        start_time = time.time()
        for i, msg in enumerate(messages[:20]):
            manager.process_message(msg)
            context = manager.get_context(msg)

        elapsed = time.time() - start_time
        stats = store.get_stats()

        print(f"  ✓ Processed 20 messages in {elapsed:.2f}s")
        print(f"  ✓ Context size: {manager.get_context_size()} tokens")
        print(f"  ✓ Memory stats: {stats}")

        manager.reset()

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("NEW MEMORY MANAGEMENT SYSTEM - TEST SUITE")
    print("=" * 60)

    results = []

    # Test memory stores
    results.append(("VectorDB Store", test_memory_store("VectorDB", VectorDBStore())))
    results.append(
        (
            "KnowledgeGraph Store",
            test_memory_store("KnowledgeGraph", KnowledgeGraphStore()),
        )
    )

    # Test context manager
    results.append(("Context Manager", test_context_manager()))

    # Test full pipeline
    results.append(("Full Pipeline", test_full_pipeline()))

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
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
