"""
Configuration management for benchmark harness.
"""

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List
from enum import Enum


class ContextManagerType(Enum):
    BASELINE = "baseline"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    VECTOR_DB = "vector_db"
    MUNINNDB = "muninndb"
    TRUSTGRAPH = "trustgraph"
    OPENAI_PARSER = "openai_parser"


class VectorStoreType(Enum):
    CHROMADB = "chromadb"
    FAISS = "faiss"
    IN_MEMORY = "in_memory"


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
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_manager_type": self.context_manager_type.value,
            "dataset_name": self.dataset_name,
            "max_messages": self.max_messages,
            "use_embeddings": self.use_embeddings,
            "k_retrieval": self.k_retrieval,
            "enable_metrics": self.enable_metrics,
            "output_dir": self.output_dir,
            "params": self.params,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class ModelConfig:
    """Configuration for the LLM models used."""

    provider: str = "openai"
    chat_model: str = "Qwen3-Coder-Next-Q4_K_M"
    parser_model: str = "LFM2.5-1.2B-Instruct-Q8_0"
    embedding_model: str = "LFM2.5-1.2B-Instruct-Q8_0"
    api_url: str = "http://localhost:58080/v1"
    api_key: str = None
    temperature: float = 0.7
    max_tokens: int = 1000

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VectorStoreConfig:
    """Configuration for vector stores."""

    store_type: VectorStoreType = VectorStoreType.IN_MEMORY
    collection_name: str = "memory"
    dimension: int = 768
    metric: str = "cosine"

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
        self.vector_store_config = VectorStoreConfig()

    def load_config(self, config_path: str) -> BenchmarkConfig:
        """Load configuration from file."""
        if not os.path.exists(config_path):
            return self.default_config

        with open(config_path, "r") as f:
            data = json.load(f)

        context_manager_type_str = data.get("context_manager_type", "baseline")
        context_manager_type = ContextManagerType(context_manager_type_str)

        # Load params with defaults
        params = data.get("params", {})
        if "api_url" not in params:
            params["api_url"] = self.model_config.api_url
        if "chat_model" not in params:
            params["chat_model"] = self.model_config.chat_model
        if "parser_model" not in params:
            params["parser_model"] = self.model_config.parser_model

        return BenchmarkConfig(
            context_manager_type=context_manager_type,
            dataset_name=data.get("dataset_name", "chatbot_conversations"),
            max_messages=data.get("max_messages"),
            use_embeddings=data.get("use_embeddings", True),
            k_retrieval=data.get("k_retrieval", 5),
            enable_metrics=data.get("enable_metrics", True),
            output_dir=data.get("output_dir", "benchmark_results"),
            params=params,
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

    def load_vector_store_config(self, config_path: str = None) -> VectorStoreConfig:
        """Load vector store configuration."""
        if config_path is None:
            config_path = os.path.join(self.config_dir, "vector_store.json")

        if not os.path.exists(config_path):
            return self.vector_store_config

        with open(config_path, "r") as f:
            data = json.load(f)

        store_type_str = data.get("store_type", "in_memory")
        store_type = VectorStoreType(store_type_str)

        return VectorStoreConfig(
            store_type=store_type,
            collection_name=data.get("collection_name", "memory"),
            dimension=data.get("dimension", 768),
            metric=data.get("metric", "cosine"),
        )

    def save_vector_store_config(
        self, config: VectorStoreConfig, config_path: str = None
    ) -> None:
        """Save vector store configuration."""
        if config_path is None:
            config_path = os.path.join(self.config_dir, "vector_store.json")

        with open(config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2)


def get_config() -> BenchmarkConfig:
    """Get default benchmark configuration."""
    return BenchmarkConfig(
        context_manager_type=ContextManagerType.BASELINE,
        dataset_name="chatbot_conversations",
    )
