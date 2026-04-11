"""
Benchmark dataset loader with support for configurable excerpt sizes.
"""

import json
import os
from typing import List, Tuple, Optional
from pathlib import Path


def load_dataset(
    dataset_name: str, max_messages: int = None
) -> Tuple[List[dict], Optional[List[str]]]:
    """Load a dataset by name with optional message limit.

    Args:
        dataset_name: Name of the dataset (e.g., 'chatbot_conversations', 'test')
        max_messages: Maximum number of messages to load (None = all)

    Returns:
        Tuple of (user_messages, assistant_references)
    """
    # Check for test dataset first
    test_path = Path("data/test_dataset.jsonl")
    if dataset_name == "chatbot_conversations" and test_path.exists():
        return load_test_dataset(str(test_path), max_messages=max_messages)

    # Check for conversation dataset
    conv_path = Path("data/conversations.jsonl")
    if dataset_name == "conversations" and conv_path.exists():
        return load_conversation_dataset(str(conv_path), max_messages=max_messages)

    # Try loading from data directory
    data_path = Path("data") / f"{dataset_name}.jsonl"
    if data_path.exists():
        return load_jsonl_dataset(str(data_path), max_messages=max_messages)
    
    # Try with _unified suffix
    unified_path = Path("data") / f"{dataset_name}_unified.jsonl"
    if unified_path.exists():
        return load_jsonl_dataset(str(unified_path), max_messages=max_messages)
    
    # Try in datasets/ subdirectory
    datasets_path = Path("data") / "datasets" / f"{dataset_name}.jsonl"
    if datasets_path.exists():
        return load_jsonl_dataset(str(datasets_path), max_messages=max_messages)
    
    datasets_unified = Path("data") / "datasets" / f"{dataset_name}_unified.jsonl"
    if datasets_unified.exists():
        return load_jsonl_dataset(str(datasets_unified), max_messages=max_messages)

    # Try loading from benchmark_results directory
    result_path = Path("benchmark_results") / f"{dataset_name}.jsonl"
    if result_path.exists():
        return load_jsonl_dataset(str(result_path), max_messages=max_messages)

    raise FileNotFoundError(
        f"Dataset '{dataset_name}' not found. "
        f"Looked in data/, data/datasets/, benchmark_results/, and data/test_dataset.jsonl"
    )


def load_test_dataset(
    path: str, max_messages: int = None
) -> Tuple[List[dict], Optional[List[str]]]:
    """Load test dataset with optional message limit."""
    messages = []
    references = []

    with open(path, "r") as f:
        for line in f:
            msg = json.loads(line.strip())
            messages.append(msg)

            # Collect assistant messages as references
            if msg.get("role") == "assistant":
                references.append(msg.get("content", ""))

    # Apply limit if specified
    if max_messages:
        messages = messages[:max_messages]

    # Separate user messages for processing
    user_messages = [m for m in messages if m.get("role") == "user"]

    return user_messages, references if references else None


def load_conversation_dataset(
    path: str, max_messages: int = None
) -> Tuple[List[dict], Optional[List[str]]]:
    """Load conversation dataset with optional message limit."""
    messages = []
    references = []

    with open(path, "r") as f:
        for line in f:
            msg = json.loads(line.strip())
            messages.append(msg)

            if msg.get("role") == "assistant":
                references.append(msg.get("content", ""))

    if max_messages:
        messages = messages[:max_messages]

    user_messages = [m for m in messages if m.get("role") == "user"]

    return user_messages, references if references else None


def load_jsonl_dataset(
    path: str, max_messages: int = None
) -> Tuple[List[dict], Optional[List[str]]]:
    """Load generic JSONL dataset."""
    messages = []
    references = []

    with open(path, "r") as f:
        for line in f:
            msg = json.loads(line.strip())
            messages.append(msg)

            if msg.get("role") == "assistant":
                references.append(msg.get("content", ""))

    if max_messages:
        messages = messages[:max_messages]

    user_messages = [m for m in messages if m.get("role") == "user"]

    return user_messages, references if references else None


def load_from_directory(
    directory: str, max_messages: int = None
) -> Tuple[List[dict], Optional[List[str]]]:
    """Load all .jsonl files from a directory."""
    all_messages = []
    all_references = []

    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory '{directory}' not found")

    for jsonl_file in dir_path.glob("*.jsonl"):
        messages, references = load_jsonl_dataset(str(jsonl_file), max_messages=None)
        all_messages.extend(messages)
        all_references.extend(references)

    if max_messages:
        all_messages = all_messages[:max_messages]

    user_messages = [m for m in all_messages if m.get("role") == "user"]

    return user_messages, all_references if all_references else None


def get_available_datasets() -> List[str]:
    """Get list of available dataset names."""
    datasets = []

    # Check standard locations
    if Path("data/test_dataset.jsonl").exists():
        datasets.append("chatbot_conversations")
    if Path("data/conversations.jsonl").exists():
        datasets.append("conversations")

    # Check data directory
    data_dir = Path("data")
    if data_dir.exists():
        for file in data_dir.glob("*.jsonl"):
            datasets.append(file.stem)

    # Check benchmark_results directory
    result_dir = Path("benchmark_results")
    if result_dir.exists():
        for file in result_dir.glob("*.jsonl"):
            datasets.append(file.stem)

    return list(set(datasets))  # Remove duplicates
