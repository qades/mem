#!/usr/bin/env python3
"""
Benchmark Coordinator Script

Orchestrates benchmark runs with configurable scope.
Uses only available stores and datasets by default.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from membench import get_available_stores
from membench.benchmark.dataset_loader import get_available_datasets


def get_local_config():
    """Generate config using only available stores and datasets."""
    stores = list(get_available_stores().keys())
    datasets = get_available_datasets()
    
    return {
        "name": "Local Benchmark (Auto-detected)",
        "description": f"Using {len(stores)} stores and {len(datasets)} datasets",
        "stores": stores,
        "datasets": datasets,
        "max_messages": 20,
        "k_retrieval": 5,
    }


def cmd_available(args):
    """Show what's available locally."""
    stores = get_available_stores()
    datasets = get_available_datasets()
    
    print("\n" + "="*60)
    print("AVAILABLE RESOURCES")
    print("="*60)
    
    print(f"\nMemory Stores ({len(stores)}):")
    for name, store_class in stores.items():
        print(f"  ✓ {name}")
    
    print(f"\nDatasets ({len(datasets)}):")
    for name in datasets:
        print(f"  ✓ {name}")
    
    print("\n" + "="*60)
    print(f"\nRun benchmark with: membench-coordinator --local")
    print("="*60 + "\n")
    
    return 0


def cmd_local(args):
    """Run benchmark with only available resources."""
    config = get_local_config()
    
    print("\n" + "="*60)
    print("LOCAL BENCHMARK RUN")
    print("="*60)
    print(f"\nConfig: {config['name']}")
    print(f"Stores: {', '.join(config['stores'])}")
    print(f"Datasets: {', '.join(config['datasets'])}")
    print(f"Max messages: {config['max_messages']}")
    print("\n" + "="*60)
    
    # TODO: Implement actual benchmark execution
    print("\n[PLACEHOLDER] Benchmark execution would run here:")
    print(f"  - Testing {len(config['stores'])} stores")
    print(f"  - On {len(config['datasets'])} datasets")
    print(f"  - Results saved to: {args.output_dir}")
    
    # Create output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_file = output_dir / f"config_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  - Config saved to: {config_file}")
    
    print("\n" + "="*60)
    print("Benchmark complete!")
    print("="*60 + "\n")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Coordinator - Run with available resources only",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show what's available
  coordinator --available
  
  # Run with all available stores and datasets
  coordinator --local
  
  # Run with specific output directory
  coordinator --local --output-dir my_results/
        """,
    )
    
    parser.add_argument(
        "--available",
        action="store_true",
        help="Show available stores and datasets",
    )
    
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run benchmark with only locally available resources",
    )
    
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Output directory for results (default: benchmark_results)",
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
