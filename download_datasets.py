#!/usr/bin/env python3
"""
Download and unify multi-turn/long-context datasets for memory framework testing.

This script downloads, processes, and unifies datasets from HuggingFace into a
common format suitable for testing memory layers vs full-context baselines.

Available datasets:
- BABILong (long-context reasoning with needle-in-haystack)
- ProLong (long-context training data)
- AgentBench (agent interactions)
- MuTual (multi-turn dialogue reasoning)

ProLong Compression:
- ProLong is very large (~140GB). Use --compress to compress after download
- Compression typically achieves 3-4x size reduction (bzip2 -3)
- Compressed data is accessed seamlessly via data/compressed_dataset_loader.py
- See compress_prolong.py for standalone compression utilities
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor
import requests

# Install datasets if not available
try:
    import datasets
except ImportError:
    print("Installing datasets library...")
    os.system(f"{sys.executable} -m pip install datasets -q")
    import datasets

from datasets import load_dataset


def download_hf_dataset(
    dataset_name: str, output_dir: Path, subset: str = None
) -> Path:
    """Download a HuggingFace dataset."""
    output_path = output_dir / dataset_name.replace("/", "_")
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {dataset_name}...")

    try:
        if subset:
            dataset = load_dataset(dataset_name, subset)
        else:
            dataset = load_dataset(dataset_name)

        for split, data in dataset.items():
            split_path = output_path / f"{split}.jsonl"
            with open(split_path, "w") as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")

        print(f"✓ Downloaded {dataset_name} to {output_path}")
        return output_path

    except Exception as e:
        print(f"❌ Error downloading {dataset_name}: {e}")
        return None


def download_babilong(output_dir: Path, subset: str = None) -> Path:
    """Download BABILong dataset - long-context needle-in-haystack benchmark."""
    dataset_name = "RMT-team/babilong"

    if subset:
        return download_hf_dataset(dataset_name, output_dir, subset)
    else:
        # Download all subsets (different context lengths)
        subsets = [
            "0k",
            "1k",
            "2k",
            "4k",
            "8k",
            "16k",
            "32k",
            "64k",
            "128k",
            "256k",
            "512k",
            "1M",
        ]
        results = []
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(download_hf_dataset, dataset_name, output_dir, s)
                for s in subsets
            ]
            for future in futures:
                results.append(future.result())
        return output_dir / "babilong"


def download_mutual(output_dir: Path) -> Path:
    """Download MuTual dataset - multi-turn dialogue reasoning."""
    dataset_name = "lighteval/mutual_harness"
    return download_hf_dataset(dataset_name, output_dir, "mutual")


def download_agentbench(output_dir: Path) -> Path:
    """Download AgentBench dataset - agent interactions."""
    dataset_name = "eth-sri/agentbench"
    return download_hf_dataset(dataset_name, output_dir)


def download_prolong(output_dir: Path, version: str = "64K") -> Path:
    """Download ProLong dataset - long-context training data."""
    if version == "64K":
        dataset_name = "princeton-nlp/prolong-data-64K"
    elif version == "512K":
        dataset_name = "princeton-nlp/prolong-data-512K"
    else:
        raise ValueError(f"Unknown ProLong version: {version}")

    output_path = output_dir / "prolong"
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download

        print(f"Downloading ProLong {version}...")
        snapshot_download(
            repo_id=dataset_name,
            repo_type="dataset",
            local_dir=output_path,
            max_workers=4,
        )
        print(f"✓ Downloaded ProLong {version} to {output_path}")
        return output_path
    except ImportError:
        print(
            "❌ huggingface_hub not installed. Install with: pip install huggingface_hub"
        )
        return None
    except Exception as e:
        print(f"❌ Error downloading ProLong: {e}")
        return None


def check_compressed_prolong() -> Path:
    """
    Check for compressed ProLong dataset.
    
    Returns path to compressed dataset if available, None otherwise.
    Compressed version is ~4x smaller and uses bzip2 -3 compression.
    """
    compressed_path = Path("compressed/prolong")
    original_path = Path("data/prolong")
    
    if compressed_path.exists():
        compressed_shards = len(list(compressed_path.glob("**/*.mds.bz2")))
        if compressed_shards > 0:
            compressed_size = sum(
                f.stat().st_size for f in compressed_path.rglob("*.mds.bz2")
            ) / (1024**3)  # GB
            print(f"📦 Compressed ProLong found: {compressed_shards} shards ({compressed_size:.1f} GB)")
            print(f"   Location: {compressed_path.absolute()}")
            print(f"   Use data/compressed_dataset_loader.py for seamless access")
            return compressed_path
    
    return None


def get_prolong_path(prefer_compressed: bool = True) -> Path:
    """
    Get the path to ProLong dataset, with optional compression preference.
    
    Args:
        prefer_compressed: If True, prefer compressed version if available
        
    Returns:
        Path to ProLong dataset directory
    """
    original_path = Path("data/prolong")
    compressed_path = Path("compressed/prolong")
    
    if prefer_compressed and compressed_path.exists():
        return compressed_path
    elif original_path.exists():
        return original_path
    elif compressed_path.exists():
        return compressed_path
    else:
        raise FileNotFoundError(
            "ProLong dataset not found. Download with: python download_datasets.py --datasets prolong"
        )


def convert_to_common_format(
    input_path: Path, output_path: Path, dataset_type: str
) -> None:
    """Convert dataset to common format for memory framework testing."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    common_data = []

    if dataset_type == "babilong":
        # Convert BABILong to QA format
        for split_file in input_path.glob("*.jsonl"):
            if not split_file.exists():
                continue
            with open(split_file, "r") as f:
                for line in f:
                    item = json.loads(line)
                    # BABILong format: input (long context), target (answer), question
                    context = item.get("input", "")
                    question = item.get("question", "")
                    answer = item.get("target", "")

                    common_data.append(
                        {
                            "dataset": "babilong",
                            "type": "long_context_qa",
                            "turns": [
                                {
                                    "role": "user",
                                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                                },
                                {"role": "assistant", "content": answer},
                            ],
                            "metadata": {"source": split_file.stem},
                        }
                    )

    elif dataset_type == "mutual":
        # Convert MuTual to multi-turn dialogue format
        for split_file in input_path.glob("*.jsonl"):
            if not split_file.exists():
                continue
            with open(split_file, "r") as f:
                for line in f:
                    item = json.loads(line)
                    # MuTual format: dialogue history + question + options + answer
                    dialogue = item.get("dialogue", [])
                    question = item.get("question", "")
                    options = item.get("options", [])
                    answer = item.get("answer", "")

                    # Build multi-turn conversation
                    turns = []
                    for turn in dialogue:
                        if "speaker" in turn and "utterance" in turn:
                            turns.append(
                                {
                                    "role": turn["speaker"].lower(),
                                    "content": turn["utterance"],
                                }
                            )

                    turns.append(
                        {"role": "user", "content": f"{question}\nOptions: {options}"}
                    )
                    turns.append({"role": "assistant", "content": answer})

                    common_data.append(
                        {
                            "dataset": "mutual",
                            "type": "multi_turn_dialogue",
                            "turns": turns,
                            "metadata": {
                                "question_type": "qa",
                                "num_options": len(options),
                            },
                        }
                    )

    elif dataset_type == "agentbench":
        # Convert AgentBench to interaction format
        for split_file in input_path.glob("*.jsonl"):
            if not split_file.exists():
                continue
            with open(split_file, "r") as f:
                for line in f:
                    item = json.loads(line)
                    # AgentBench format: task description + interaction traces
                    problem = item.get("problem_description", "")
                    interactions = item.get("interactions", [])

                    turns = []
                    if problem:
                        turns.append({"role": "system", "content": problem})

                    for interaction in interactions:
                        if "action" in interaction:
                            turns.append(
                                {"role": "user", "content": interaction["action"]}
                            )
                        if "observation" in interaction:
                            turns.append(
                                {
                                    "role": "assistant",
                                    "content": interaction["observation"],
                                }
                            )

                    common_data.append(
                        {
                            "dataset": "agentbench",
                            "type": "agent_interaction",
                            "turns": turns,
                            "metadata": {"task_id": item.get("instance_id", "")},
                        }
                    )

    elif dataset_type == "prolong":
        # ProLong is pre-tokenized, extract text sequences
        # This requires special handling with mosaicml-streaming
        # Compressed version available in compressed/prolong/ with .mds.bz2 files
        common_data.append(
            {
                "dataset": "prolong",
                "type": "long_context_text",
                "turns": [
                    {"role": "user", "content": "[ProLong long-context training data]"},
                    {
                        "role": "assistant",
                        "content": "This dataset contains long-context training data with 65,536 token sequences.",
                    },
                ],
                "metadata": {
                    "note": "ProLong requires huggingface_hub and mosaicml-streaming for full extraction. "
                            "Compressed version available - see data/compressed_dataset_loader.py",
                    "compressed_path": "compressed/prolong",
                    "compression": "bzip2 -3 (4x size reduction)"
                },
            }
        )

    # Write unified format
    with open(output_path, "w") as f:
        for item in common_data:
            f.write(json.dumps(item) + "\n")

    print(f"✓ Converted {len(common_data)} items to common format at {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and unify multi-turn/long-context datasets"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/datasets",
        help="Output directory for downloaded datasets",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["babilong", "mutual", "agentbench", "prolong", "all"],
        default=["all"],
        help="Datasets to download",
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Convert downloaded datasets to common format",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Specific subset for BABILong (e.g., 128k, 1m)",
    )
    parser.add_argument(
        "--check-compressed",
        action="store_true",
        help="Check for compressed ProLong dataset availability",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Compress ProLong dataset after download (saves ~75%% disk space)",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=3,
        choices=range(1, 10),
        metavar="LEVEL",
        help="Compression level 1-9 (default: 3, recommended sweet spot)",
    )
    parser.add_argument(
        "--compression-threads",
        type=int,
        default=min(32, os.cpu_count() or 4),
        help=f"Parallel compression threads (default: {min(32, os.cpu_count() or 4)})",
    )

    args = parser.parse_args()
    
    # Check for compressed datasets if requested
    if args.check_compressed:
        print("Checking for compressed datasets...")
        print("-" * 60)
        check_compressed_prolong()
        print("-" * 60)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded_paths = {}

    if "all" in args.datasets or len(args.datasets) == 0:
        args.datasets = ["babilong", "mutual", "agentbench", "prolong"]

    if "babilong" in args.datasets:
        path = download_babilong(output_dir, args.subset)
        if path:
            downloaded_paths["babilong"] = path

    if "mutual" in args.datasets:
        path = download_mutual(output_dir)
        if path:
            downloaded_paths["mutual"] = path

    if "agentbench" in args.datasets:
        path = download_agentbench(output_dir)
        if path:
            downloaded_paths["agentbench"] = path

    if "prolong" in args.datasets:
        path = download_prolong(output_dir)
        if path:
            downloaded_paths["prolong"] = path
            
            # Compress ProLong if requested
            if args.compress:
                print("\n" + "=" * 60)
                print("Compressing ProLong dataset...")
                print("=" * 60)
                print(f"This will reduce disk usage by ~75% using bzip2 -{args.compression_level}")
                print(f"Using {args.compression_threads} parallel threads")
                print("")
                
                compress_path = Path("compressed/prolong")
                compress_path.mkdir(parents=True, exist_ok=True)
                
                # Run compression using compress_prolong.py if available
                compress_script = Path(__file__).parent / "compress_prolong.py"
                if compress_script.exists():
                    result = subprocess.run([
                        sys.executable, str(compress_script),
                        "--input", str(path),
                        "--output", str(compress_path),
                        "--threads", str(args.compression_threads),
                        "--level", str(args.compression_level)
                    ])
                    
                    if result.returncode == 0:
                        print("\n✓ ProLong compression complete!")
                        print(f"  Compressed data location: {compress_path}")
                        print(f"  Use data/compressed_dataset_loader.py for seamless access")
                    else:
                        print("\n⚠ Compression failed or incomplete", file=sys.stderr)
                else:
                    print(f"\n⚠ Compression script not found: {compress_script}")
                    print("  Run manually: python compress_prolong.py")

    if args.convert:
        print("\n" + "=" * 60)
        print("Converting to common format...")
        print("=" * 60)

        for dataset_type, path in downloaded_paths.items():
            if path and path.exists():
                convert_to_common_format(
                    path, output_dir / f"{dataset_type}_unified.jsonl", dataset_type
                )

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    print(f"\nDatasets downloaded to: {output_dir}")
    print("\nNext steps:")
    print("1. Review downloaded data in JSONL format")
    print("2. Use unified files for memory framework testing")
    print("3. Adjust context window sizes based on dataset requirements")
    print("\nDataset notes:")
    print("- BABILong: Good for testing long-context retrieval (up to 1M tokens)")
    print("- MuTual: Good for multi-turn dialogue with reasoning")
    print("- AgentBench: Good for agent-environment interactions")
    print("- ProLong: Pre-tokenized long-context training data (64K or 512K tokens)")
    print("  ├─ Large dataset (~140GB) - use --compress to save ~75% disk space")
    print("  ├─ Compressed: compressed/prolong/ (~4x smaller, ~35-45GB)")
    print("  └─ Access: data/compressed_dataset_loader.py (seamless, works with both)")
    print("\nCompression commands:")
    print("  python download_datasets.py --datasets prolong --compress")
    print("  python compress_prolong.py --threads 32 --level 3")


if __name__ == "__main__":
    main()
