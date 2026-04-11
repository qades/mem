# Documentation

This document provides comprehensive documentation for the memory/context management system.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Benchmark Harness                                │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                    Context Manager Strategies                       │  │
│  │  - Baseline (full history)                                         │  │
│  │  - OpenAI Parser (LLM extraction)                                  │  │
│  │  - Memory-based (various stores)                                   │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                    │                                     │
│  ┌─────────────────────────────────┴──────────────────────────────────┐ │
│  │                        Memory Stores                                │ │
│  │  ┌────────────────────────────────────────────────────────────────┐ │ │
│  │  │  Reference Libraries (pip installable)                        │ │ │
│  │  │  • Mem0 - Hybrid (vector + graph + KV)                        │ │ │
│  │  │  • Zep - Temporal knowledge graph                             │ │ │
│  │  │  • Graphiti - Episode-based temporal KG                       │ │ │
│  │  │  • Letta (MemGPT) - OS-style hierarchical                     │ │ │
│  │  │  • MemPalace - Spatial (method of loci)                       │ │ │
│  │  └────────────────────────────────────────────────────────────────┘ │ │
│  │  ┌────────────────────────────────────────────────────────────────┐ │ │
│  │  │  Custom Implementations                                       │ │ │
│  │  │  • MuninnDB - Graph with OpenAI API                           │ │ │
│  │  │  • TrustGraph - API + benchmarking                            │ │ │
│  │  │  • VectorDB - ChromaDB/FAISS/in-memory                        │ │ │
│  │  │  • KnowledgeGraph - Local triple store                        │ │ │
│  │  └────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
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

## Memory Store Reference Implementations

This benchmark suite includes wrappers for the latest memory libraries. Each follows the `BaseMemoryStore` interface for fair comparison.

### Mem0 (mem0ai)

**Paper**: Chhikara et al. (2025) - Mem0: Universal memory layer for AI Agents
**Library**: `pip install mem0ai`

Hybrid datastore architecture combining:
- **Vector store**: Semantic similarity search
- **Graph store**: Entity-relationship tracking
- **Key-value store**: Fast fact lookups

Features automatic fact extraction with ADD/UPDATE/DELETE/NOOP operations.

```python
from memory_stores.mem0_store import Mem0Store

store = Mem0Store(
    api_key="your-openai-key",
    user_id="user_123",
    vector_store_provider="qdrant",
    graph_store_provider="neo4j",
)

# Insert - Mem0 auto-extracts facts
store.insert("user", "prefers", "python")

# Retrieve - Hybrid search across all stores
results = store.retrieve("programming language", k=5)
```

### Zep (zep-python)

**Paper**: Rasmussen et al. (2025) - Zep: Temporal knowledge graph for agents
**Library**: `pip install zep-python neo4j`

Temporal knowledge graph with bi-temporal modeling:
- **Event time**: When the fact was true in the real world
- **Transaction time**: When the fact was recorded

Enables time-travel queries: "What was Project X's status last week?"

```python
from memory_stores.zep_store import ZepStore

store = ZepStore(
    api_key="your-zep-key",
    base_url="http://localhost:8000",
    user_id="user_123",
    session_id="session_456",
)

# Insert - Creates temporal fact
store.insert("project", "status", "in_progress")

# Retrieve at specific time
from datetime import datetime
results = store.retrieve_at_time(
    "project status", 
    timestamp=datetime(2025, 1, 1),
    k=5
)
```

### Graphiti (graphiti-core)

**Library**: `pip install graphiti-core neo4j`

Temporal knowledge graph engine from Zep AI:
- Episode-based memory organization
- Dynamic graph construction from conversations
- Multi-hop graph traversal

```python
from memory_stores.graphiti_store import GraphitiStore

store = GraphitiStore(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password",
)

# Add episode - Graphiti extracts entities/edges
store.store_message({
    "role": "user",
    "content": "My name is Alice and I work on Project X"
})

# Search graph
results = store.retrieve("Alice's project", k=5)
```

### Letta (formerly MemGPT)

**Paper**: Packer et al. (2023) - MemGPT: LLMs as Operating Systems
**Library**: `pip install letta`

OS-style hierarchical memory:
- **Core memory**: Working context (RAM)
- **Archival memory**: Long-term storage (disk)
- **Agent-controlled**: Self-managing via function calls

```python
from memory_stores.letta_store import LettaStore

store = LettaStore(
    agent_name="benchmark_agent",
    user_id="user_123",
    context_window_limit=8000,
)

# Agent manages its own memory paging
store.store_message({
    "role": "user", 
    "content": "I need help with Python"
})

# Get core memory (working context)
core = store.get_core_memory()
```

### MemPalace

**Library**: `pip install mempalace chromadb`

Spatial memory organization using method of loci:
- **Wings**: People/projects
- **Rooms**: Topics within wings
- **Halls**: Connections between rooms
- **Closets**: Summaries
- **Drawers**: Verbatim storage

~170 token wake-up cost with 4-layer memory stack.

```python
from memory_stores.mempalace_store import MemPalaceStore

store = MemPalaceStore(
    palace_path="~/.mempalace",
    wing="project_alpha",
    room="backend",
    use_aaak=True,  # Enable compression
)

# Store in spatial structure
store.insert("auth", "uses", "JWT tokens", metadata={"hall": "decisions"})

# Get L0+L1 wake-up context (~170 tokens)
context = store.get_wake_up_context()

# Search within room
results = store.retrieve_in_room("authentication", room="backend")
```

### A-MEM (optional)

**Paper**: Xu et al. (2025) - A-MEM: Agentic Memory for LLM Agents
**Library**: `pip install a-mem litellm`

Zettelkasten-style memory:
- Atomic notes with metadata
- Agentic linking between notes
- Self-organizing knowledge graph

Note: Requires additional dependencies (litellm, sentence-transformers).

## Comparison Matrix

| Store | Architecture | Key Feature | Dependencies |
|-------|-------------|-------------|--------------|
| **Mem0** | Hybrid (vector+graph+KV) | Auto fact extraction | mem0ai, qdrant-client |
| **Zep** | Temporal KG | Bi-temporal queries | zep-python, neo4j |
| **Graphiti** | Temporal KG | Episode-based | graphiti-core, neo4j |
| **Letta** | Hierarchical | Self-managing agent | letta |
| **MemPalace** | Spatial | Method of loci | mempalace, chromadb |
| **MuninnDB** | Graph (API) | OpenAI-compatible API | requests |
| **VectorDB** | Vector | Multiple backends | chromadb/faiss |
| **Knowledge Graph** | Triple store | Local graph | (none) |

## Quick Start

```python
from memory_stores import create_store, get_available_stores

# See what's installed
print(get_available_stores())
# {'mem0': Mem0Store, 'zep': ZepStore, ...}

# Create any store by name
store = create_store("mem0", api_key="...", user_id="user_1")

# All stores follow BaseMemoryStore interface
store.insert("user", "likes", "python")
results = store.retrieve("programming", k=5)
```

## Troubleshooting

### Mem0: "No module named 'mem0'"
```bash
pip install mem0ai
# Import is: from mem0 import Memory
```

### Zep: Connection refused
- Ensure Zep server is running: `docker run -p 8000:8000 zepai/zep`
- Check `base_url` matches server address

### Graphiti: Neo4j errors
```bash
# Start Neo4j
docker run -p 7474:7474 -p 7687:7687 neo4j:latest
```

### MemPalace: ChromaDB version conflict
```bash
# Use compatible version
pip install "chromadb>=0.5.0,<0.7"
```

### Letta: Database errors
- Letta requires PostgreSQL for persistence
- Or use in-memory mode: `create_store("letta")` without config
