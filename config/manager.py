"""
Configuration management for benchmark harness.
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from enum import Enum


class ContextManagerType(Enum):
    BASELINE = "baseline"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    VECTOR_DB = "vector_db"
    MUNINNDB = "muninndb"
    TRUSTGRAPH = "trustgraph"
    OPENAI_PARSER = "openai_parser"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    context_manager_type: ContextManagerType
    dataset_name: str
    max_messages: Optional[int] = None
    use_embeddings: bool = True
    k_retrieval: int = 5
    enable_metrics: bool = True
    output_dir: str = "benchmark_results"
    params: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_manager_type": self.context_manager_type.value,
            "dataset_name": self.dataset_name,
            "max_messages": self.max_messages,
            "use_embeddings": self.use_embeddings,
            "k_retrieval": self.k_retrieval,
            "enable_metrics": self.enable_metrics,
            "output_dir": self.output_dir,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class ModelConfig:
    """Configuration for the LLM models used."""

    provider: str = "openai"
    model_name: str = "gpt-3.5-turbo"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.7
    max_tokens: int = 1000

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConfigManager:
    """Manage benchmark configurations."""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)

        self.default_config = BenchmarkConfig(
            context_manager_type=ContextManagerType.BASELINE,
            dataset_name="chatbot_conversations",
        )

        self.model_config = ModelConfig()

    def load_config(self, config_path: str) -> BenchmarkConfig:
        """Load configuration from file."""
        if not os.path.exists(config_path):
            return self.default_config

        with open(config_path, "r") as f:
            data = json.load(f)

        context_manager_type_str = data.get("context_manager_type", "baseline")
        context_manager_type = ContextManagerType(context_manager_type_str)

        return BenchmarkConfig(
            context_manager_type=context_manager_type,
            dataset_name=data.get("dataset_name", "chatbot_conversations"),
            max_messages=data.get("max_messages"),
            use_embeddings=data.get("use_embeddings", True),
            k_retrieval=data.get("k_retrieval", 5),
            enable_metrics=data.get("enable_metrics", True),
            output_dir=data.get("output_dir", "benchmark_results"),
            params=data.get("params", {}),
        )

    def save_config(self, config: BenchmarkConfig, config_path: str) -> None:
        """Save configuration to file."""
        with open(config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2)

    def load_model_config(self, config_path: str = None) -> ModelConfig:
        """Load model configuration."""
        if config_path is None:
            config_path = os.path.join(self.config_dir, "model.json")

        if not os.path.exists(config_path):
            return self.model_config

        with open(config_path, "r") as f:
            data = json.load(f)

        return ModelConfig(**data)

    def save_model_config(self, config: ModelConfig, config_path: str = None) -> None:
        """Save model configuration."""
        if config_path is None:
            config_path = os.path.join(self.config_dir, "model.json")

        with open(config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2)


def get_config() -> BenchmarkConfig:
    """Get default benchmark configuration."""
    return BenchmarkConfig(
        context_manager_type=ContextManagerType.BASELINE,
        dataset_name="chatbot_conversations",
    )
