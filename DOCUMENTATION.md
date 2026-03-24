# Documentation

This document provides comprehensive documentation for the memory/context management system.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Benchmark Harness                         │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Context Manager Strategy                              │  │
│  │  - Baseline (full history)                            │  │
│  │  - OpenAI Parser (LLM extraction)                     │  │
│  │  - Vector DB (similarity)                             │  │
│  │  - Knowledge Graph (relations)                        │  │
│  │  - MuninnDB (API)                                     │  │
│  │  - TrustGraph (API + benchmarking)                    │  │
│  └────────────────────────────────────────────────────────┘  │
│                         │                                     │
│  ┌──────────────────────┴──────────────────────────────────┐ │
│  │                    Memory Store                         │ │
│  │  - ChromaDB / FAISS / in-memory (vector)               │ │
│  │  - Knowledge graph (triple store)                      │ │
│  │  - API (MuninnDB, TrustGraph)                          │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

### Model Configuration

**File: `config/model.json`**

```json
{
  "provider": "openai",
  "chat_model": "Qwen3-Coder-Next-Q4_K_M",
  "parser_model": "LFM2.5-1.2B-Instruct-Q8_0",
  "embedding_model": "LFM2.5-1.2B-Instruct-Q8_0",
  "api_url": "http://localhost:58080/v1",
  "api_key": null,
  "temperature": 0.7,
  "max_tokens": 1000
}
```

### Vector Store Configuration

**File: `config/vector_store.json`**

```json
{
  "store_type": "in_memory",
  "collection_name": "memory",
  "dimension": 768,
  "metric": "cosine"
}
```

Options:
- `in_memory`: Fast, no dependencies
- `chromadb`: Persistent vector storage
- `faiss`: Fast similarity search

### Benchmark Configuration

**File: `config/*.json`**

```json
{
  "context_manager_type": "openai_parser",
  "dataset_name": "chatbot_conversations",
  "max_messages": 20,
  "use_embeddings": true,
  "k_retrieval": 5,
  "enable_metrics": true,
  "output_dir": "benchmark_results",
  "params": {
    "api_url": "http://localhost:58080/v1",
    "parser_model": "LFM2.5-1.2B-Instruct-Q8_0",
    "enable_benchmarking": true
  }
}
```

## Usage

### Python API

#### Load Configuration

```python
from config.manager import ConfigManager, ContextManagerType, VectorStoreType

config_mgr = ConfigManager()

# Load model config
model_config = config_mgr.load_model_config()
print(model_config.chat_model)  # Qwen3-Coder-Next-Q4_K_M

# Load vector store config
vector_config = config_mgr.load_vector_store_config()
print(vector_config.store_type)  # VectorStoreType.IN_MEMORY

# Load benchmark config
benchmark_config = config_mgr.load_config("config/openai_parser.json")
print(benchmark_config.context_manager_type)  # ContextManagerType.OPENAI_PARSER
```

#### Run Benchmark

```python
from benchmark.harness import BenchmarkHarness, BenchmarkConfig
from config.manager import ContextManagerType

# Configure
config = BenchmarkConfig(
    context_manager_type=ContextManagerType.OPENAI_PARSER,
    dataset_name="chatbot_conversations",
    max_messages=20,
    params={
        "api_url": "http://localhost:58080/v1",
        "parser_model": "LFM2.5-1.2B-Instruct-Q8_0",
    }
)

# Run
harness = BenchmarkHarness(config)
result = harness.run_benchmark(messages)

print(f"Context size: {result.context_size} tokens")
print(f"Response time: {result.response_time_ms:.2f}ms")
```

#### Use Context Manager Directly

```python
from context_managers.openai_parser import OpenAICompatibleContextManager

# Create parser (NOT a memory store)
parser = OpenAICompatibleContextManager(
    api_url="http://localhost:58080/v1",
    parser_model="LFM2.5-1.2B-Instruct-Q8_0",
    k_retrieval=5
)

# Process messages
parser.process_message({"role": "user", "content": "Hello, I love Python"})
parser.process_message({"role": "assistant", "content": "Hi! What do you want to build?"})

# Get context
context = parser.get_context({"role": "user", "content": "What should I use?"})
print(context)

# Get benchmark summary
summary = parser.get_benchmark_summary()
print(summary)  # {'avg_time_ms': 150.5, ...}

# Reset
parser.reset()
```

#### Use Vector DB Store

```python
from memory_stores.vector_db import VectorDBStore

# In-memory (default)
store = VectorDBStore(store_type="in_memory", dimension=768)

# Or ChromaDB
store = VectorDBStore(
    store_type="chromadb",
    collection_name="my_collection",
    api_url="http://localhost:8000"
)

# Or FAISS
store = VectorDBStore(store_type="faiss", dimension=768)

# Insert
vec_id = store.insert("user", "likes", "python", {"timestamp": "2024-01-01"})

# Retrieve
results = store.retrieve("python programming", k=5)
for r in results:
    print(f"{r['value']} (score: {r['score']})")

# Get stats
print(store.get_stats())
```

## Dataset Format

### JSONL Format

```json
{"role": "user", "content": "Hello"}
{"role": "assistant", "content": "Hi there"}
{"role": "user", "content": "What can you do?"}
```

### Conversation ID Format

```json
{"role": "user", "content": "Hello", "conversation_id": 0}
{"role": "assistant", "content": "Hi", "conversation_id": 0}
{"role": "user", "content": "Hello", "conversation_id": 1}
```

## Strategies

### Baseline

Full conversation history. No parsing or extraction.

```python
config = BenchmarkConfig(
    context_manager_type=ContextManagerType.BASELINE,
    dataset_name="chatbot_conversations"
)
```

### OpenAI Parser

Extracts structured information using LLM API.

```python
config = BenchmarkConfig(
    context_manager_type=ContextManagerType.OPENAI_PARSER,
    dataset_name="chatbot_conversations",
    params={
        "api_url": "http://localhost:58080/v1",
        "parser_model": "LFM2.5-1.2B-Instruct-Q8_0",
    }
)
```

### Vector DB

Vector similarity-based retrieval.

```python
config = BenchmarkConfig(
    context_manager_type=ContextManagerType.VECTOR_DB,
    dataset_name="chatbot_conversations",
    use_embeddings=True,
    params={
        "store_type": "in_memory",  # or chromadb, faiss
        "dimension": 768,
        "k_retrieval": 5,
    }
)
```

### Knowledge Graph

Relationship-based retrieval.

```python
config = BenchmarkConfig(
    context_manager_type=ContextManagerType.KNOWLEDGE_GRAPH,
    dataset_name="chatbot_conversations",
    use_embeddings=False,
)
```

### MuninnDB

Memory store with OpenAI-compatible API.

```python
config = BenchmarkConfig(
    context_manager_type=ContextManagerType.MUNINNDB,
    dataset_name="chatbot_conversations",
    params={
        "api_url": "http://localhost:8000",
        "vault": "default",
    }
)
```

### TrustGraph

Benchmarkable data stores with OpenAI-compatible API.

```python
config = BenchmarkConfig(
    context_manager_type=ContextManagerType.TRUSTGRAPH,
    dataset_name="chatbot_conversations",
    params={
        "api_url": "http://localhost:3000",
        "vault": "default",
        "enable_benchmarking": True,
    }
)
```

## Testing

### Run All Tests

```bash
python test_new_system.py
```

### Quick Start

```bash
./quickstart.sh
```

### Run Benchmark

```bash
# Default 20 messages
./run_benchmark.sh

# Custom size
./run_benchmark.sh 50

# With dataset name
./run_benchmark.sh 20 conversation_dataset
```

## Output

Results saved to `benchmark_results/`:

```json
{
  "context_manager_type": "openai_parser",
  "dataset_name": "chatbot_conversations",
  "total_messages": 20,
  "context_size": 1000,
  "response_time_ms": 150.5,
  "memory_usage_mb": 10.2,
  "metrics": {},
  "metadata": {
    "k_retrieval": 5,
    "use_embeddings": false,
    "parser_summary": {
      "total_messages": 20,
      "avg_time_ms": 150.5,
      "min_time_ms": 120.0,
      "max_time_ms": 200.0
    }
  }
}
```

## Performance Considerations

### Caching

- Parser results are cached per session
- Vector DB can use ChromaDB/FAISS for persistent storage
- Consider implementing request caching for production

### Async I/O

- Can add async support with `aiohttp` for parallel API calls
- Measure performance impact before implementing

### Memory Usage

- Baseline: O(n) where n = messages
- Vector DB: O(n * d) where d = embedding dimension
- Parser: O(n) for parsed data

## Troubleshooting

### API Connection Failed

- Check API URL: `http://localhost:58080/v1`
- Verify server is running
- Check firewall settings

### Missing Dependencies

```bash
# ChromaDB
pip install chromadb

# FAISS
pip install faiss
```

### Dataset Not Found

- Check `data/` directory
- Verify JSONL format
- Check file permissions

## API Reference

### ConfigManager

- `load_config(path)`: Load benchmark config
- `save_config(config, path)`: Save benchmark config
- `load_model_config(path)`: Load model config
- `save_model_config(config, path)`: Save model config
- `load_vector_store_config(path)`: Load vector store config
- `save_vector_store_config(config, path)`: Save vector store config

### BenchmarkHarness

- `run_benchmark(messages, references)`: Run benchmark
- `run_comparison(datasets)`: Run across datasets
- `save_results(results, path)`: Save to file

### OpenAICompatibleContextManager

- `process_message(msg)`: Parse and store
- `get_context(msg)`: Get relevant context
- `get_context_size()`: Get token count
- `reset()`: Clear temporary memory
- `get_benchmark_summary()`: Performance metrics
- `get_parsed_history()`: Get all parsed data
- `get_entity_index()`: Entity → message index

### VectorDBStore

- `insert(entity, relation, value, metadata)`: Insert vector
- `retrieve(query, k, use_embedding)`: Search
- `update(id, entity, relation, value)`: Update
- `delete(id)`: Remove vector
- `clear()`: Clear all
- `get_stats()`: Statistics
