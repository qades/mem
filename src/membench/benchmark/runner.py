"""
Benchmark Runner - Wires up coordinator to harness.

Runs benchmarks across multiple memory stores and datasets.
"""

import json
import time
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

warnings.filterwarnings("ignore")

from membench.benchmark.harness import BenchmarkHarness, BenchmarkConfig, BenchmarkResult
from membench.benchmark.dataset_loader import load_dataset
from membench.config.manager import ContextManagerType
from membench import create_store, get_available_stores


# Map store types to their required context manager type
STORE_TO_CONTEXT_MANAGER = {
    "mem0": ContextManagerType.VECTOR_DB,  # Use generic memory-based
    "zep": ContextManagerType.VECTOR_DB,
    "graphiti": ContextManagerType.KNOWLEDGE_GRAPH,
    "letta": ContextManagerType.VECTOR_DB,
    "mempalace": ContextManagerType.VECTOR_DB,
    "muninndb": ContextManagerType.MUNINNDB,
    "knowledge_graph": ContextManagerType.KNOWLEDGE_GRAPH,
    "vector_db": ContextManagerType.VECTOR_DB,
    "trustgraph": ContextManagerType.TRUSTGRAPH,
}


class BenchmarkRunner:
    """Runner that executes benchmarks across stores and datasets."""
    
    def __init__(
        self,
        api_url: str = "http://localhost:58080/v1",
        output_dir: str = "benchmark_results",
        max_messages: int = 20,
    ):
        self.api_url = api_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_messages = max_messages
        
    def run_store_vs_dataset(
        self,
        store_type: str,
        dataset_name: str,
    ) -> Optional[BenchmarkResult]:
        """Run a single store against a single dataset."""
        print(f"  Running {store_type} on {dataset_name}...", end=" ")
        
        try:
            # Determine context manager type
            context_type = STORE_TO_CONTEXT_MANAGER.get(
                store_type, 
                ContextManagerType.VECTOR_DB
            )
            
            # Create config
            config = BenchmarkConfig(
                context_manager_type=context_type,
                dataset_name=dataset_name,
                max_messages=self.max_messages,
                params={
                    "store_type": store_type,
                    "api_url": self.api_url,
                },
            )
            
            # Create harness
            harness = BenchmarkHarness(config)
            
            # Load dataset
            messages, references = load_dataset(dataset_name)
            
            # Run benchmark
            result = harness.run_benchmark(messages, references)
            
            print(f"✓ ({result.response_time_ms:.1f}ms)")
            return result
            
        except Exception as e:
            print(f"✗ ({e})")
            return None
    
    def run_all(
        self,
        stores: List[str],
        datasets: List[str],
    ) -> Dict[str, Any]:
        """Run all store/dataset combinations."""
        results = []
        errors = []
        
        total = len(stores) * len(datasets)
        current = 0
        
        print(f"\nRunning {total} benchmarks...")
        print("=" * 60)
        
        for store_type in stores:
            for dataset_name in datasets:
                current += 1
                print(f"[{current}/{total}]", end=" ")
                
                result = self.run_store_vs_dataset(store_type, dataset_name)
                
                if result:
                    results.append({
                        "store": store_type,
                        "dataset": dataset_name,
                        "context_size": result.context_size,
                        "response_time_ms": result.response_time_ms,
                        "total_messages": result.total_messages,
                    })
                else:
                    errors.append({
                        "store": store_type,
                        "dataset": dataset_name,
                    })
        
        # Compile results
        summary = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "api_url": self.api_url,
                "max_messages": self.max_messages,
                "stores": stores,
                "datasets": datasets,
            },
            "results": results,
            "errors": errors,
            "summary": {
                "total": total,
                "successful": len(results),
                "failed": len(errors),
            },
        }
        
        # Save results
        results_file = self.output_dir / f"benchmark_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(results_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        return summary


def run_benchmark(
    stores: List[str],
    datasets: List[str],
    api_url: str = "http://localhost:58080/v1",
    output_dir: str = "benchmark_results",
    max_messages: int = 20,
) -> Dict[str, Any]:
    """Convenience function to run benchmarks."""
    runner = BenchmarkRunner(
        api_url=api_url,
        output_dir=output_dir,
        max_messages=max_messages,
    )
    return runner.run_all(stores, datasets)
