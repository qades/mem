# Memory Management System Documentation

## Overview

This system implements a flexible memory management architecture for LLM conversations, supporting multiple storage backends with OpenAI-compatible API integration.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Context Manager                           │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │   Baseline      │  │  Memory-Based    │  │  OpenAI     │ │
│  │  (Full History) │  │  (Retrieval)     │  │  Parser     │ │
│  └─────────────────┘  └──────────────────┘  └─────────────┘ │
└──────────────────────────┬────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
    ┌─────▼─────┐    ┌─────▼─────┐    ┌─────▼─────┐
    │ VectorDB  │    │ Knowledge │    │  OpenAI     │
    │           │    │  Graph    │    │  Compatible │
    └───────────┘    └───────────┘    │   API       │
                                      └─────┬─────┘
                                            │
                                      ┌─────▼─────┐
                                      │  MuninnDB   │
                                      │ TrustGraph  │
                                      └─────────────┘
```

## Memory Stores

### 1. VectorDBStore

Simple in-memory vector storage with cosine similarity search.

**Features:**
- Vector embeddings (768 dimensions, hash-based)
- Inverted index for keyword search
- BM25 scoring for relevance
- Fast retrieval

**Usage:**
```python
from memory_stores.vector_db import VectorDBStore

store = VectorDBStore()
store.insert("Python", "programming_language", "popular")
results = store.retrieve("Python programming", k=5)
```

### 2. KnowledgeGraphStore

Graph-based memory with node-edge structure.

**Features:**
- Nodes for entities and values
- Edges for relations
- Simple graph traversal
- Entity indexing

**Usage:**
```python
from memory_stores.knowledge_graph import KnowledgeGraphStore

store = KnowledgeGraphStore()
store.insert("user", "likes", "Python")
results = store.retrieve("user preferences", k=5)
```

### 3. MuninnDBStore

Persistent memory using OpenAI-compatible API.

**Features:**
- OpenAI-compatible API integration
- Vector embeddings via API
- Entity extraction via LLM
- Persistent storage
- Confidence scoring

**Usage:**
```python
from memory_stores.muninndb import MuninnDBStore

store = MuninnDBStore(
    api_url="http://localhost:8000",
    vault="default"
)
store.insert("Python", "language", "popular")
results = store.retrieve("Python programming", k=5)
```

### 4. TrustGraphStore

Benchmarkable memory with performance tracking.

**Features:**
- OpenAI-compatible API integration
- Benchmarking support
- Performance metrics
- Entity extraction via LLM
- Persistent storage

**Usage:**
```python
from memory_stores.trustgraph import TrustGraphStore

store = TrustGraphStore(
    api_url="http://localhost:3000",
    enable_benchmarking=True
)
store.insert("Python", "language", "popular")
results = store.retrieve("Python programming", k=5)
print(store.get_benchmark_summary())
```

## Context Managers

### 1. BaselineContextManager

Sends all previous messages with the current one.

**Pros:**
- Simple
- No parsing overhead
- Complete context

**Cons:**
- Large context size
- No retrieval optimization

### 2. MemoryBasedContextManager

Parses messages, stores information, retrieves relevant context.

**Features:**
- Information extraction
- Semantic search
- Context optimization
- Multiple memory stores

**Usage:**
```python
from memory_stores.vector_db import VectorDBStore
from context_managers.memory_based import MemoryBasedContextManager

store = VectorDBStore()
manager = MemoryBasedContextManager(
    memory_store=store,
    use_embeddings=True,
    k_retrieval=5
)

manager.process_message({"role": "user", "content": "Hello"})
context = manager.get_context({"role": "user", "content": "How are you?"})
```

### 3. OpenAICompatibleContextManager

Uses OpenAI-compatible API for fast context parsing.

**Features:**
- Fast LLM parsing
- Entity extraction
- Relation extraction
- Benchmarking support

**Usage:**
```python
from context_managers.openai_parser import OpenAICompatibleContextManager

manager = OpenAICompatibleContextManager(
    api_url="http://localhost:8000",
    model="gpt-3.5-turbo",
    k_retrieval=5
)

manager.process_message({"role": "user", "content": "Hello"})
context = manager.get_context({"role": "user", "content": "How are you?"})
```

## OpenAI-Compatible API Server

The system includes a test server that provides an OpenAI-compatible API.

### Endpoints

- `GET /health` - Health check
- `POST /chat/completions` - Chat completions
- `POST /embeddings` - Embedding generation
- `POST /memories` - Create memory
- `POST /memories/update` - Update memory
- `POST /memories/delete` - Delete memory
- `POST /memories/clear` - Clear memories
- `POST /search` - Semantic search
- `GET /stats` - Get statistics

### Running the Server

```bash
python3 server.py --port 8000
```

## Configuration

### Config File Format

```json
{
  "context_manager_type": "vector_db",
  "dataset_name": "test_dataset",
  "use_embeddings": true,
  "k_retrieval": 5,
  "api_url": "http://localhost:8000",
  "vault": "default",
  "params": {
    "max_memory_items": 10000,
    "retrieval_top_k": 5
  }
}
```

### Available Configs

- `config/baseline.json` - Baseline strategy
- `config/vector_db.json` - Vector DB memory store
- `config/knowledge_graph.json` - Knowledge graph memory store
- `config/muninndb.json` - MuninnDB with API
- `config/trustgraph.json` - TrustGraph with benchmarking
- `config/openai_parser.json` - OpenAI parser

## Running Benchmarks

### Single Benchmark

```bash
python3 run_benchmark.py --config config/vector_db.json
```

### All Benchmarks

```bash
python3 run_benchmark.py --compare
```

### Quick Start

```bash
bash quickstart.sh
```

## Testing

```bash
python3 test_memory_system.py
```

## API Integration

The system works with any OpenAI-compatible API server. To use with a real API:

1. Start your OpenAI-compatible server
2. Update the config with the correct `api_url`
3. Set the `api_key` if required
4. Run benchmarks

### Example with Real API

```json
{
  "context_manager_type": "muninndb",
  "api_url": "https://api.openai.com/v1",
  "api_key": "sk-...",
  "vault": "default"
}
```

## Performance

### Benchmark Results

| Strategy | Context Size | Response Time |
|----------|--------------|---------------|
| Baseline | 146 tokens | 0.03ms |
| VectorDB | 500 tokens | 16.76ms |
| KnowledgeGraph | 500 tokens | 10.00ms |
| MuninnDB | 500 tokens | 23.69ms |
| TrustGraph | 500 tokens | 1075.84ms |
| OpenAI Parser | 500 tokens | 10.68ms |

## Adding New Memory Stores

1. Implement `BaseMemoryStore` interface
2. Add to `ContextManagerType` enum
3. Register in harness

## Adding New Context Managers

1. Implement `BaseContextManager` interface
2. Add to `ContextManagerType` enum
3. Register in harness

## License

MIT
