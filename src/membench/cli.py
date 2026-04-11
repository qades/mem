#!/usr/bin/env python3
"""
Command-line interface for membench.

Usage:
    membench --help
    membench list-stores
    membench list-datasets
    membench run --config quick
    membench run --stores mem0 zep --datasets locomo
"""

# Suppress Pydantic deprecation warnings from external libraries (graphiti, letta)
# Must be set BEFORE any imports that might trigger these warnings
import warnings
import os

# Use both filterwarnings and environment variable for maximum effectiveness
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*PydanticDeprecatedSince20.*")
warnings.filterwarnings("ignore", message=".*json_encoders.*")
warnings.filterwarnings("ignore", message=".*Support for class-based.*")

# Also suppress via env var for subprocesses
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning,ignore::UserWarning"

import argparse
import sys
from pathlib import Path

# Now import membench (warnings should be suppressed)
from membench import get_available_stores, __version__


def cmd_list_stores(args):
    """List available memory stores."""
    stores = get_available_stores()
    print("\nAvailable Memory Stores:")
    print("=" * 60)
    for name, store_class in stores.items():
        print(f"  • {name:15} - {store_class.__doc__.split(chr(10))[0] if store_class.__doc__ else 'No description'}")
    print()
    return 0


def cmd_list_datasets(args):
    """List available datasets."""
    data_dir = Path(__file__).parent.parent.parent / "data"
    
    print("\nAvailable Datasets:")
    print("=" * 60)
    
    dataset_files = {
        "locomo": ("LoCoMo", "Long Context Monitoring benchmark"),
        "babilong": ("BABILong", "Long-context QA benchmark"),
        "mutual": ("MuTual", "Multi-turn dialogue reasoning"),
        "agentbench": ("AgentBench", "Agent capability benchmark"),
        "prolong": ("ProLong", "Long conversation benchmark"),
    }
    
    for key, (name, desc) in dataset_files.items():
        exists = (data_dir / f"{key}.jsonl").exists() or (data_dir / f"{key}_unified.jsonl").exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {name:15} - {desc}")
    
    print()
    return 0


def cmd_run(args):
    """Run benchmarks."""
    print(f"\nMemBench v{__version__}")
    print("=" * 60)
    print(f"Configuration: {args.config}")
    
    if args.stores:
        print(f"Stores: {', '.join(args.stores)}")
    if args.datasets:
        print(f"Datasets: {', '.join(args.datasets)}")
    
    print("\nRun functionality is a placeholder. Implement full benchmark logic here.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="membench",
        description="Memory Benchmark Suite for Long-Term Agent Memory Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # list-stores command
    list_stores_parser = subparsers.add_parser(
        "list-stores",
        help="List available memory stores",
    )
    list_stores_parser.set_defaults(func=cmd_list_stores)
    
    # list-datasets command
    list_datasets_parser = subparsers.add_parser(
        "list-datasets",
        help="List available datasets",
    )
    list_datasets_parser.set_defaults(func=cmd_list_datasets)
    
    # run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run benchmarks",
    )
    run_parser.add_argument(
        "--config",
        default="quick",
        help="Configuration preset (quick, standard, full) or path to config file",
    )
    run_parser.add_argument(
        "--stores",
        nargs="+",
        help="Specific stores to benchmark",
    )
    run_parser.add_argument(
        "--datasets",
        nargs="+",
        help="Specific datasets to use",
    )
    run_parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for results",
    )
    run_parser.set_defaults(func=cmd_run)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
