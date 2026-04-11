#!/usr/bin/env python3
"""
Benchmark Coordinator Script

Runs benchmarks with available stores and datasets.

Usage:
    coordinator --available              # Show what's available
    coordinator --local                  # Run with all available resources
    coordinator --local --dry-run        # Show what would run
    coordinator --api-url http://...     # Use different API endpoint
"""

import argparse
import json
import sys
import os
import warnings
from pathlib import Path
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from membench import get_available_stores
from membench.benchmark.dataset_loader import get_available_datasets
from membench.benchmark.runner import BenchmarkRunner


def get_valid_datasets():
    """Get datasets that actually have content (not stubs)."""
    data_dir = Path(__file__).parent.parent / "data"
    datasets = {}
    
    # Look for actual data files
    for f in data_dir.glob("*.jsonl"):
        if f.stat().st_size > 10000:
            name = f.stem.replace("_unified", "")
            datasets[name] = str(f)
    
    datasets_dir = data_dir / "datasets"
    if datasets_dir.exists():
        for f in datasets_dir.glob("*.jsonl"):
            if f.stat().st_size > 10000:
                name = f.stem.replace("_unified", "")
                datasets[name] = str(f)
    
    return datasets  # Returns {name: path} dict


def get_valid_stores():
    """Get stores that can be instantiated without external services."""
    stores = get_available_stores()
    valid = []
    
    for name in stores.keys():
        # Skip stores requiring external services
        if name in ["graphiti", "zep"]:
            continue
        valid.append(name)
    
    return valid


def cmd_available(args):
    """Show what's available locally."""
    stores = get_available_stores()
    valid_stores = get_valid_stores()
    datasets = get_valid_datasets()
    
    print("\n" + "="*60)
    print("AVAILABLE RESOURCES")
    print("="*60)
    
    print(f"\nMemory Stores ({len(stores)} installed, {len(valid_stores)} ready):")
    for name in stores.keys():
        status = "✓ ready" if name in valid_stores else "✗ needs service"
        print(f"  {name:15} {status}")
    
    print(f"\nDatasets ({len(datasets)}):")
    for name in sorted(datasets.keys()):
        print(f"  ✓ {name}")
    
    print("\n" + "="*60)
    print("\nTo run benchmark:")
    print("  ./scripts/membench-coordinator.sh --local --dry-run  # Preview")
    print("  ./scripts/membench-coordinator.sh --local            # Execute")
    print("="*60 + "\n")
    
    return 0


def cmd_local(args):
    """Run benchmark with available resources."""
    stores = get_valid_stores()
    datasets_dict = get_valid_datasets()
    dataset_names = list(datasets_dict.keys())
    
    if not stores:
        print("Error: No memory stores available!")
        return 1
    
    if not dataset_names:
        print("Error: No datasets available!")
        return 1
    
    # Determine API URL
    api_url = args.api_url
    if not api_url:
        # Try environment variable
        api_url = os.environ.get("OPENAI_API_BASE", "http://localhost:58080/v1")
    
    print("\n" + "="*60)
    print("LOCAL BENCHMARK RUN")
    print("="*60)
    print(f"\nAPI URL: {api_url}")
    print(f"Stores: {', '.join(stores)}")
    print(f"Datasets: {', '.join(dataset_names)}")
    print(f"Max messages: {args.max_messages}")
    
    if args.dry_run:
        print("\n[DRY RUN - No execution]")
        print("="*60 + "\n")
        return 0
    
    # Run benchmarks
    print("\n" + "="*60)
    print("\nStarting benchmark execution...")
    print("="*60)
    
    try:
        runner = BenchmarkRunner(
            api_url=api_url,
            output_dir=args.output_dir,
            max_messages=args.max_messages,
        )
        
        results = runner.run_all(stores, dataset_names)
        
        # Print summary
        print("\n" + "="*60)
        print("BENCHMARK COMPLETE")
        print("="*60)
        print(f"\nTotal: {results['summary']['total']}")
        print(f"Successful: {results['summary']['successful']}")
        print(f"Failed: {results['summary']['failed']}")
        
        if results['results']:
            print("\nResults preview:")
            for r in results['results'][:5]:
                print(f"  {r['store']} + {r['dataset']}: {r['context_size']} tokens, {r['response_time_ms']:.1f}ms")
        
        if results['errors']:
            print("\nErrors:")
            for e in results['errors'][:5]:
                print(f"  ✗ {e['store']} + {e['dataset']}")
        
        print(f"\nResults saved to: {args.output_dir}/")
        print("="*60 + "\n")
        
        return 0 if results['summary']['failed'] == 0 else 1
        
    except Exception as e:
        print(f"\n✗ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmarks with available resources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--available",
        action="store_true",
        help="Show available stores and datasets",
    )
    
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run with locally available resources",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would run without executing",
    )
    
    parser.add_argument(
        "--api-url",
        help="LLM API URL (default: http://localhost:58080/v1 or $OPENAI_API_BASE)",
    )
    
    parser.add_argument(
        "--max-messages",
        type=int,
        default=20,
        help="Maximum messages to process (default: 20)",
    )
    
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Output directory (default: benchmark_results)",
    )
    
    args = parser.parse_args()
    
    if args.available:
        return cmd_available(args)
    elif args.local:
        return cmd_local(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
