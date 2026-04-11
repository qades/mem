#!/usr/bin/env python3
"""
Benchmark Coordinator Script

Runs benchmarks with available stores and datasets.

Usage:
    coordinator --available              # Show what's available
    coordinator --local                  # Run with all available resources
    coordinator --local --dry-run        # Show what would run without executing
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


def get_valid_datasets():
    """Get datasets that actually have content (not stubs)."""
    data_dir = Path(__file__).parent.parent / "data"
    datasets = []
    
    # Check for actual data files
    for f in data_dir.glob("*.jsonl"):
        if f.stat().st_size > 10000:  # Must be > 10KB to be real data
            datasets.append(f.stem.replace("_unified", ""))
    
    # Also check datasets/ subdirectory
    datasets_dir = data_dir / "datasets"
    if datasets_dir.exists():
        for f in datasets_dir.glob("*.jsonl"):
            if f.stat().st_size > 10000:
                datasets.append(f.stem.replace("_unified", ""))
    
    return sorted(set(datasets))


def get_valid_stores():
    """Get stores that can actually be instantiated."""
    stores = get_available_stores()
    valid = []
    
    for name in stores.keys():
        # Skip stores that require external services not running
        if name in ["graphiti", "zep", "letta"]:
            # These need Neo4j, Zep server, etc.
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
    for name in datasets:
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
    datasets = get_valid_datasets()
    
    if not stores:
        print("Error: No memory stores available!")
        return 1
    
    if not datasets:
        print("Error: No datasets available!")
        return 1
    
    print("\n" + "="*60)
    print("LOCAL BENCHMARK RUN")
    print("="*60)
    print(f"\nStores: {', '.join(stores)}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Max messages: 20")
    
    if args.dry_run:
        print("\n[DRY RUN - No execution]")
        print("="*60 + "\n")
        return 0
    
    print("\n" + "="*60)
    print("\n⚠️  ACTUAL BENCHMARK EXECUTION NOT YET IMPLEMENTED")
    print("\nThe coordinator currently shows what's available but doesn't")
    print("yet wire up to the actual benchmark harness.")
    print("\nTo run actual benchmarks, use the old run_benchmark.py:")
    print("  python scripts/run_benchmark.py --config config/baseline.json")
    print("\nNOTE: Requires LLM API running at localhost:58080")
    print("For Docker, use: http://host.containers.internal:58080")
    print("="*60 + "\n")
    
    # Save config for future use
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = {
        "timestamp": datetime.now().isoformat(),
        "stores": stores,
        "datasets": datasets,
        "max_messages": 20,
        "note": "Benchmark execution pending implementation",
    }
    
    config_file = output_dir / f"config_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to: {config_file}\n")
    
    return 0


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
