"""
Main entry point for running benchmarks.
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark.harness import BenchmarkHarness, BenchmarkConfig, ContextManagerType
from config.manager import ConfigManager


def run_benchmark(config_path: str = None):
    """Run benchmark with given configuration."""
    config_manager = ConfigManager()

    if config_path:
        config = config_manager.load_config(config_path)
    else:
        config = BenchmarkConfig(
            context_manager_type=ContextManagerType.BASELINE,
            dataset_name="babilong",
        )

    harness = BenchmarkHarness(config)
    results = harness.run_comparison([config.dataset_name])

    output_path = os.path.join(
        config.output_dir, f"{config.context_manager_type.value}_results.json"
    )
    os.makedirs(config.output_dir, exist_ok=True)
    harness.save_results(results, output_path)

    return results


def compare_strategies():
    """Compare all context management strategies."""
    strategies = [
        ContextManagerType.BASELINE,
        ContextManagerType.KNOWLEDGE_GRAPH,
        ContextManagerType.VECTOR_DB,
    ]

    all_results = []

    for strategy in strategies:
        config = BenchmarkConfig(
            context_manager_type=strategy,
            dataset_name="babilong",
            use_embeddings=strategy != ContextManagerType.KNOWLEDGE_GRAPH,
        )

        harness = BenchmarkHarness(config)
        results = harness.run_comparison([config.dataset_name])
        all_results.extend(results)

        print(f"\n{'=' * 60}")
        print(f"Strategy: {strategy.value}")
        print(f"{'=' * 60}")
        for result in results:
            print(f"  Context size: {result.context_size} tokens")
            print(f"  Response time: {result.response_time_ms:.2f}ms")
            print(f"  Memory usage: {result.memory_usage_mb:.2f}MB")
            if result.metrics:
                print(f"  Metrics: {json.dumps(result.metrics, indent=2)}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Context Management Benchmark Harness")
    parser.add_argument("--config", "-c", help="Path to configuration file")
    parser.add_argument(
        "--compare", "-C", action="store_true", help="Compare all strategies"
    )
    parser.add_argument(
        "--list-datasets", "-l", action="store_true", help="List available datasets"
    )

    args = parser.parse_args()

    if args.list_datasets:
        from datasets.loader import list_available_datasets

        datasets = list_available_datasets()
        print("Available datasets:")
        for ds in datasets:
            print(f"  - {ds}")
        return

    if args.compare:
        results = compare_strategies()
    else:
        results = run_benchmark(args.config)

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    for result in results:
        print(f"\n{result.context_manager_type}:")
        print(f"  Context: {result.context_size} tokens")
        print(f"  Time: {result.response_time_ms:.2f}ms")


if __name__ == "__main__":
    main()
