#!/usr/bin/env python3
"""
Loader utilities for unified multi-turn/long-context datasets.

Provides consistent loading and preprocessing for datasets downloaded
by download_datasets.py, formatted for memory framework testing.

Compressed dataset support:
- ProLong dataset can be stored compressed (bzip2 -3, ~4x smaller)
- Use compressed_dataset_loader.py for seamless transparent access
- Compressed location: compressed/prolong/
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional
import random

# Add data directory to path for compressed loader
sys.path.insert(0, str(Path(__file__).parent / "data"))
try:
    from compressed_dataset_loader import (
        open_compressed,
        find_shard_file,
        load_prolong_dataset,
        get_dataset_path,
    )
    _HAS_COMPRESSED_LOADER = True
except ImportError:
    _HAS_COMPRESSED_LOADER = False


def load_unified_dataset(
    data_path: str, split: str = None, shuffle: bool = False, seed: int = 42
) -> List[Dict[str, Any]]:
    """Load unified dataset from JSONL file."""
    path = Path(data_path)

    if path.is_dir():
        # Load all JSONL files in directory
        files = list(path.glob("*.jsonl"))
        if not files:
            raise ValueError(f"No JSONL files found in {data_path}")

        data = []
        for f in files:
            data.extend(load_file(f))
    else:
        data = load_file(path)

    if shuffle:
        random.seed(seed)
        random.shuffle(data)

    return data


def load_file(file_path: Path) -> List[Dict[str, Any]]:
    """Load a single JSONL file."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_hf_dataset(dataset_name: str, split: str = None) -> List[Dict[str, Any]]:
    """Load directly from HuggingFace (bypass download script)."""
    try:
        from datasets import load_dataset

        dataset = load_dataset(dataset_name)

        if split:
            return list(dataset[split])
        else:
            all_data = []
            for s in dataset.keys():
                all_data.extend(list(dataset[s]))
            return all_data
    except ImportError:
        raise ImportError(
            "datasets library not installed. Install with: pip install datasets"
        )


def format_for_memory_testing(
    data: List[Dict[str, Any]],
    max_turns: int = None,
    max_context_tokens: int = None,
    context_window: int = 4096,
) -> List[Dict[str, Any]]:
    """
    Format data for memory framework testing.

    Args:
        data: Unified dataset items
        max_turns: Maximum turns to keep (None = all)
        max_context_tokens: Maximum context tokens (None = all)
        context_window: Model's context window size

    Returns:
        Formatted samples ready for memory framework
    """
    formatted = []

    for item in data:
        sample = {
            "dataset": item.get("dataset", "unknown"),
            "type": item.get("type", "unknown"),
            "turns": item.get("turns", []),
            "metadata": item.get("metadata", {}),
        }

        # Truncate turns if specified
        if max_turns and len(sample["turns"]) > max_turns:
            sample["turns"] = sample["turns"][-max_turns:]

        # Truncate context if specified (approximate token counting)
        if max_context_tokens:
            sample = truncate_context(sample, max_context_tokens)

        formatted.append(sample)

    return formatted


def truncate_context(sample: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
    """Truncate conversation context to fit within token limit."""
    total_tokens = 0
    truncated_turns = []

    # Process turns from most recent (reverse order)
    for turn in reversed(sample["turns"]):
        content_tokens = len(turn.get("content", "").split())

        if total_tokens + content_tokens <= max_tokens:
            truncated_turns.insert(0, turn)
            total_tokens += content_tokens
        else:
            # Keep recent turn but truncate content
            if not truncated_turns:  # First turn, must keep something
                words = turn["content"].split()
                kept_words = words[-max_tokens:]
                turn["content"] = "..." + " ".join(kept_words)
                truncated_turns.insert(0, turn)
            break

    sample["turns"] = truncated_turns
    return sample


def extract_qa_pairs(data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Extract question-answer pairs from multi-turn conversations."""
    qa_pairs = []

    for item in data:
        turns = item.get("turns", [])

        # Extract user questions and assistant answers
        for i in range(len(turns) - 1):
            if (
                turns[i].get("role") == "user"
                and turns[i + 1].get("role") == "assistant"
            ):
                qa_pairs.append(
                    {
                        "question": turns[i].get("content", ""),
                        "answer": turns[i + 1].get("content", ""),
                        "dataset": item.get("dataset", "unknown"),
                        "type": item.get("type", "unknown"),
                    }
                )

    return qa_pairs


def get_context_lengths(data: List[Dict[str, Any]]) -> List[int]:
    """Get approximate token counts for each sample."""
    lengths = []
    for item in data:
        total_length = 0
        for turn in item.get("turns", []):
            total_length += len(turn.get("content", "").split())
        lengths.append(total_length)
    return lengths


def split_by_length(
    data: List[Dict[str, Any]], buckets: List[int] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """Split data into buckets by approximate context length."""
    if buckets is None:
        buckets = [100, 500, 1000, 2000, 4000, 8000, 16000, 32000]

    buckets_dict = {f"<{buckets[0]}": [], "between": {}, f">={buckets[-1]}": []}

    for i, b in enumerate(buckets[:-1]):
        buckets_dict["between"][f"{b}-{buckets[i + 1]}"] = []

    for item in data:
        length = sum(len(t.get("content", "").split()) for t in item.get("turns", []))

        if length < buckets[0]:
            buckets_dict[f"<{buckets[0]}"].append(item)
        elif length >= buckets[-1]:
            buckets_dict[f">={buckets[-1]}"].append(item)
        else:
            for i in range(len(buckets) - 1):
                if buckets[i] <= length < buckets[i + 1]:
                    key = f"{buckets[i]}-{buckets[i + 1]}"
                    buckets_dict["between"][key].append(item)
                    break

    return buckets_dict


def create_memory_test_samples(
    data: List[Dict[str, Any]], test_type: str = "long_context", num_samples: int = None
) -> List[Dict[str, Any]]:
    """
    Create test samples for memory framework evaluation.

    Args:
        data: Unified dataset
        test_type: Type of test - 'long_context', 'multi_turn', 'retrieval'
        num_samples: Number of samples to select (None = all)

    Returns:
        Test samples with appropriate structure for memory testing
    """
    samples = []

    for item in data:
        sample = {
            "id": f"{item.get('dataset', 'unknown')}_{len(samples)}",
            "dataset": item.get("dataset", "unknown"),
            "type": item.get("type", "unknown"),
            "turns": item.get("turns", []),
            "metadata": item.get("metadata", {}),
        }

        if test_type == "long_context":
            # For long-context tests, ensure we have the full conversation
            sample["max_turns"] = len(sample["turns"])
        elif test_type == "multi_turn":
            # For multi-turn tests, sample recent turns
            sample["turns"] = sample["turns"][-10:]  # Last 10 turns
        elif test_type == "retrieval":
            # For retrieval tests, extract key facts
            sample["key_facts"] = extract_key_facts(sample["turns"])

        samples.append(sample)

        if num_samples and len(samples) >= num_samples:
            break

    return samples


def extract_key_facts(turns: List[Dict[str, str]]) -> List[str]:
    """Extract key factual statements from conversation."""
    facts = []
    for turn in turns:
        content = turn.get("content", "").lower()
        # Simple heuristic: look for factual statements
        if any(
            keyword in content
            for keyword in [
                "is",
                "was",
                "are",
                "were",
                "travelling",
                "journeyed",
                "moved",
                "went",
            ]
        ):
            # Extract sentence-like structures
            sentences = content.split(". ")
            for sentence in sentences:
                if len(sentence.split()) > 3:  # Filter very short phrases
                    facts.append(sentence.strip())

    return facts[:10]  # Return up to 10 key facts


def load_prolong(data_path: Optional[str] = None, **kwargs) -> Any:
    """
    Load ProLong dataset with automatic compression detection.
    
    Automatically detects and handles both compressed (.mds.bz2) and
    uncompressed (.mds) files. If a file exists in both formats, the
    uncompressed version is preferred.
    
    Args:
        data_path: Path to ProLong dataset (default: auto-detect from
                  data/prolong or compressed/prolong)
        **kwargs: Additional arguments passed to StreamingDatasetWrapper
        
    Returns:
        StreamingDatasetWrapper providing access to the dataset
        
    Example:
        >>> dataset = load_prolong()
        >>> print(f"Dataset size: {len(dataset)}")
        >>> for sample in dataset:
        ...     process(sample)
        
    Note:
        Requires mosaicml-streaming for full functionality. Without it,
        provides basic file access to shard contents.
    """
    if not _HAS_COMPRESSED_LOADER:
        raise ImportError(
            "Compressed dataset loader not available. "
            "Ensure data/compressed_dataset_loader.py exists."
        )
    
    if data_path is None:
        data_path = str(get_dataset_path())
    
    return load_prolong_dataset(data_path, **kwargs)


def get_prolong_info() -> Dict[str, Any]:
    """
    Get information about available ProLong datasets.
    
    Returns:
        Dictionary with paths and compression info
    """
    info = {
        "original_path": "data/prolong",
        "compressed_path": "compressed/prolong",
        "original_exists": Path("data/prolong").exists(),
        "compressed_exists": Path("compressed/prolong").exists(),
    }
    
    if info["original_exists"]:
        orig_files = list(Path("data/prolong").rglob("*.mds"))
        info["original_shards"] = len(orig_files)
        info["original_size_gb"] = sum(f.stat().st_size for f in orig_files) / (1024**3)
    
    if info["compressed_exists"]:
        comp_files = list(Path("compressed/prolong").rglob("*.mds.bz2"))
        info["compressed_shards"] = len(comp_files)
        if comp_files:
            info["compressed_size_gb"] = sum(f.stat().st_size for f in comp_files) / (1024**3)
            if info.get("original_size_gb"):
                info["compression_ratio"] = info["original_size_gb"] / info["compressed_size_gb"]
    
    return info


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and process unified datasets")
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to unified dataset JSONL file or directory",
    )
    parser.add_argument("--split", type=str, default=None, help="Dataset split to load")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle data")
    parser.add_argument(
        "--max-turns", type=int, default=None, help="Maximum turns per sample"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=4096, help="Maximum context tokens"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output file for processed data"
    )

    args = parser.parse_args()

    # Load data
    data = load_unified_dataset(args.data_path, split=args.split, shuffle=args.shuffle)

    # Process data
    processed = format_for_memory_testing(
        data, max_turns=args.max_turns, max_context_tokens=args.max_tokens
    )

    # Print statistics
    print(f"\nLoaded {len(data)} samples")
    lengths = get_context_lengths(data)
    print(f"Context length stats:")
    print(f"  Min: {min(lengths)} tokens")
    print(f"  Max: {max(lengths)} tokens")
    print(f"  Avg: {sum(lengths) / len(lengths):.1f} tokens")

    # Split by length
    buckets = split_by_length(data)
    print(f"\nSamples by length bucket:")
    for bucket, items in buckets.items():
        if isinstance(items, list):
            print(f"  {bucket}: {len(items)} samples")
        else:
            for subbucket, subitems in items.items():
                print(f"  {subbucket}: {len(subitems)} samples")

    # Output if specified
    if args.output:
        with open(args.output, "w") as f:
            for item in processed:
                f.write(json.dumps(item) + "\n")
        print(f"\n✓ Processed {len(processed)} samples written to {args.output}")
