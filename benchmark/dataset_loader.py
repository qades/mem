"""Dataset loader for benchmark harness."""

from typing import List, Dict, Any, Optional
import json
from pathlib import Path


def load_dataset(dataset_name: str, data_dir: str = "data/datasets") -> tuple:
    """
    Load a dataset for benchmarking.

    Args:
        dataset_name: Name of the dataset to load
        data_dir: Directory containing dataset files

    Returns:
        Tuple of (messages, references) where:
            - messages: List of message dicts with 'role' and 'content' keys
            - references: List of reference answers (optional)
    """
    # Try to load from unified dataset files
    data_path = Path(data_dir) / f"{dataset_name}_unified.jsonl"

    if data_path.exists():
        return _load_unified_dataset(data_path)

    # Try direct HuggingFace dataset
    try:
        from datasets import load_dataset as hf_load_dataset
        from datasets.exceptions import DatasetNotFoundError

        dataset = hf_load_dataset(dataset_name)

        if isinstance(dataset, dict):
            # DatasetDict - combine all splits
            all_data = []
            for split in dataset.values():
                all_data.extend(list(split))
        else:
            all_data = list(dataset)

        return _convert_hf_dataset_to_messages(all_data)
    except (ImportError, Exception) as e:
        # Fallback: use mock data for testing
        if dataset_name == "chatbot_conversations" or dataset_name == "test_dataset":
            return _get_mock_dataset()
        raise ValueError(
            f"Could not load dataset '{dataset_name}': {e}. "
            f"Ensure the dataset exists on HuggingFace Hub or is downloaded locally."
        )


def _load_unified_dataset(path: Path) -> tuple:
    """Load from unified JSONL format."""
    messages = []
    references = []

    with open(path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            turns = item.get("turns", [])

            # Extract messages and references from turns
            for turn in turns:
                if turn.get("role") == "user":
                    messages.append(turn)
                elif turn.get("role") == "assistant":
                    references.append(turn.get("content", ""))

    # If no references found, return empty list
    if not references:
        references = [None] * len(messages)

    return messages, references


def _convert_hf_dataset_to_messages(data: List[Dict[str, Any]]) -> tuple:
    """Convert HuggingFace dataset to messages format."""
    messages = []
    references = []

    for item in data:
        # Try to extract turns
        if "turns" in item:
            for turn in item["turns"]:
                if turn.get("role") == "user":
                    messages.append(turn)
                elif turn.get("role") == "assistant":
                    references.append(turn.get("content", ""))
        # Try to extract dialogue
        elif "dialogue" in item:
            for turn in item["dialogue"]:
                role = turn.get("speaker", "user").lower()
                content = turn.get("utterance", "")
                if role == "user":
                    messages.append({"role": "user", "content": content})
                else:
                    references.append(content)
        # Try to extract conversation
        elif "conversation" in item:
            conv = item["conversation"]
            for turn in conv:
                if turn.get("human"):
                    messages.append({"role": "user", "content": turn["human"]})
                if turn.get("answer"):
                    references.append(turn["answer"])
        # Generic fallback - look for common keys
        else:
            # Try to find any text content
            for key in ["text", "content", "input", "query"]:
                if key in item:
                    messages.append({"role": "user", "content": str(item[key])})
                    break

    # If no references found, return empty list
    if not references and messages:
        references = [None] * len(messages)

    return messages, references


def _get_mock_dataset() -> tuple:
    """Return mock dataset for testing when no real dataset is available."""
    messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {
            "role": "assistant",
            "content": "I'm doing well, thank you for asking! How can I help you today?",
        },
        {
            "role": "user",
            "content": "Can you explain how memory management works in Python?",
        },
        {
            "role": "assistant",
            "content": "Python memory management involves several key components...",
        },
        {"role": "user", "content": "What about garbage collection?"},
        {
            "role": "assistant",
            "content": "Python uses reference counting and cyclic garbage collection...",
        },
        {"role": "user", "content": "Can you tell me about the GIL?"},
        {
            "role": "assistant",
            "content": "The Global Interpreter Lock (GIL) is a mutex that protects access to Python objects...",
        },
        {"role": "user", "content": "Thanks for the explanation!"},
        {
            "role": "assistant",
            "content": "You're welcome! Feel free to ask if you have more questions.",
        },
    ]
    references = [msg["content"] for msg in messages if msg["role"] == "assistant"]
    return messages, references


def list_available_datasets(data_dir: str = "data/datasets") -> List[str]:
    """List available datasets in the data directory."""
    datasets = []
    data_path = Path(data_dir)

    if data_path.exists():
        for file in data_path.glob("*_unified.jsonl"):
            datasets.append(file.stem.replace("_unified", ""))

    return datasets
