"""
Tests for context management benchmark harness.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.harness import BenchmarkHarness, BenchmarkConfig, ContextManagerType
from context_managers.baseline import BaselineContextManager
from context_managers.memory_based import MemoryBasedContextManager
from memory_stores.knowledge_graph import KnowledgeGraphStore
from memory_stores.vector_db import VectorDBStore
from datasets import load_dataset


class TestContextManagers(unittest.TestCase):
    """Test context manager implementations."""

    def test_baseline_context_manager(self):
        """Test baseline context manager processes messages correctly."""
        cm = BaselineContextManager()

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]

        for msg in messages:
            cm.process_message(msg)

        context = cm.get_context(messages[-1])

        self.assertIn("user:", context.lower())
        self.assertIn("assistant:", context.lower())
        self.assertIn("how are you?", context.lower())
        self.assertGreater(cm.get_context_size(), 0)

    def test_baseline_reset(self):
        """Test baseline context manager reset works."""
        cm = BaselineContextManager()
        cm.process_message({"role": "user", "content": "Test"})
        cm.reset()

        self.assertEqual(len(cm.conversation_history), 0)
        self.assertEqual(cm.get_context_size(), 0)

    def test_memory_based_context_manager(self):
        """Test memory-based context manager."""
        store = KnowledgeGraphStore()
        cm = MemoryBasedContextManager(memory_store=store)

        messages = [
            {"role": "user", "content": "I like Python programming"},
            {"role": "assistant", "content": "Python is great!"},
        ]

        for msg in messages:
            cm.process_message(msg)

        context = cm.get_context(messages[-1])
        self.assertIsInstance(context, str)

    def test_memory_based_reset(self):
        """Test memory-based context manager reset."""
        store = KnowledgeGraphStore()
        cm = MemoryBasedContextManager(memory_store=store)
        cm.process_message({"role": "user", "content": "Test"})
        cm.reset()

        self.assertEqual(len(cm.stored_info), 0)


class TestMemoryStores(unittest.TestCase):
    """Test memory store implementations."""

    def test_knowledge_graph_store(self):
        """Test knowledge graph store operations."""
        store = KnowledgeGraphStore()

        store.insert("Python", "is_a", "programming language")
        store.insert("Python", "created_by", "Guido van Rossum")

        results = store.retrieve("Python")

        self.assertGreater(len(results), 0)

    def test_vector_db_store(self):
        """Test vector DB store operations."""
        store = VectorDBStore()

        store.insert("Python", "is_a", "programming language")
        store.insert("Java", "is_a", "programming language")

        results = store.retrieve("Python", use_embedding=True)

        self.assertGreater(len(results), 0)

    def test_vector_db_clear(self):
        """Test vector DB store clear."""
        store = VectorDBStore()
        store.insert("test", "rel", "value")
        store.clear()

        results = store.retrieve("test")
        self.assertEqual(len(results), 0)


class TestDatasetLoader(unittest.TestCase):
    """Test dataset loading."""

    def test_load_synthetic_dataset(self):
        """Test loading synthetic dataset."""
        messages, references = load_dataset("chatbot_conversations")

        self.assertIsInstance(messages, list)
        self.assertGreater(len(messages), 0)

        for msg in messages:
            self.assertIn("role", msg)
            self.assertIn("content", msg)


class TestBenchmarkHarness(unittest.TestCase):
    """Test benchmark harness."""

    def test_baseline_benchmark(self):
        """Test baseline benchmark run."""
        config = BenchmarkConfig(
            context_manager_type=ContextManagerType.BASELINE,
            dataset_name="chatbot_conversations",
        )
        harness = BenchmarkHarness(config)

        messages, _ = load_dataset("chatbot_conversations")
        result = harness.run_benchmark(messages[:3])

        self.assertEqual(result.context_manager_type, "baseline")
        self.assertGreater(result.total_messages, 0)
        self.assertGreater(result.context_size, 0)

    def test_knowledge_graph_benchmark(self):
        """Test knowledge graph benchmark run."""
        config = BenchmarkConfig(
            context_manager_type=ContextManagerType.KNOWLEDGE_GRAPH,
            dataset_name="chatbot_conversations",
            use_embeddings=False,
        )
        harness = BenchmarkHarness(config)

        messages, _ = load_dataset("chatbot_conversations")
        result = harness.run_benchmark(messages[:3])

        self.assertEqual(result.context_manager_type, "knowledge_graph")

    def test_vector_db_benchmark(self):
        """Test vector DB benchmark run."""
        config = BenchmarkConfig(
            context_manager_type=ContextManagerType.VECTOR_DB,
            dataset_name="chatbot_conversations",
            use_embeddings=True,
        )
        harness = BenchmarkHarness(config)

        messages, _ = load_dataset("chatbot_conversations")
        result = harness.run_benchmark(messages[:3])

        self.assertEqual(result.context_manager_type, "vector_db")

    def test_comparison_run(self):
        """Test comparison across multiple datasets."""
        config = BenchmarkConfig(
            context_manager_type=ContextManagerType.BASELINE,
            dataset_name="chatbot_conversations",
        )
        harness = BenchmarkHarness(config)

        results = harness.run_comparison(["chatbot_conversations"])

        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].context_manager_type, "baseline")


if __name__ == "__main__":
    unittest.main()
