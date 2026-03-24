# Memory Management System - Quick Reference

## Quick Start

```bash
# Install and run tests
bash quickstart.sh

# Run benchmarks
python3 run_benchmark.py --config config/vector_db.json
python3 run_benchmark.py --config config/muninndb.json
python3 run_benchmark.py --config config/trustgraph.json
python3 run_benchmark.py --config config/openai_parser.json

# Compare all strategies
python3 run_benchmark.py --compare

# Start OpenAI-compatible test server
python3 server.py --port 8000
```

## Memory Stores

| Store | API | Embeddings | Benchmarking | Use Case |
|-------|-----|------------|--------------|----------|
| VectorDB | None | Hash | No | Fast, local storage |
| KnowledgeGraph | None | Hash | No | Graph relationships |
| MuninnDB | OpenAI | API | No | Persistent, API-based |
| TrustGraph | OpenAI | API | Yes | Benchmarkable, API-based |

## Context Managers

| Manager | Strategy | Speed | Context Size |
|---------|----------|-------|--------------|
| Baseline | Full history | ⚡⚡⚡ | Large |
| MemoryBased | Retrieval | ⚡⚡ | Optimized |
| OpenAIParser | LLM parsing | ⚡⚡⚡ | Optimized |

## Configuration

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

## API Endpoints

- `GET /health` - Health check
- `POST /chat/completions` - Chat completions
- `POST /embeddings` - Embeddings
- `POST /memories` - Create memory
- `POST /memories/update` - Update memory
- `POST /memories/delete` - Delete memory
- `POST /memories/clear` - Clear memories
- `POST /search` - Search
- `GET /stats` - Statistics

## Available Types

- `baseline` - Full history
- `vector_db` - Vector DB memory store
- `knowledge_graph` - Knowledge graph memory store
- `muninndb` - MuninnDB with API
- `trustgraph` - TrustGraph with benchmarking
- `openai_parser` - OpenAI parser

## Test Results

```
✓ VectorDB Store - PASS
✓ KnowledgeGraph Store - PASS
✓ Context Manager - PASS
✓ Full Pipeline - PASS
✓ MuninnDB Store - PASS
✓ TrustGraph Store - PASS
✓ OpenAI Parser - PASS
```

## Performance

| Strategy | Context | Time |
|----------|---------|------|
| Baseline | 146 tokens | 0.03ms |
| VectorDB | 500 tokens | 9.16ms |
| KnowledgeGraph | 500 tokens | 0.17ms |
| MuninnDB | 500 tokens | 16.90ms |
| TrustGraph | 500 tokens | 1479.90ms |
| OpenAI Parser | 500 tokens | 5.77ms |

## Key Commands

```bash
# Run tests
python3 test_memory_system.py

# Run comprehensive tests
python3 test_new_system.py

# List datasets
python3 run_benchmark.py --list-datasets

# Compare all strategies
python3 run_benchmark.py --compare
```

## File Structure

```
mem/
├── memory_stores/
│   ├── base.py
│   ├── vector_db.py
│   ├── knowledge_graph.py
│   ├── muninndb.py (NEW)
│   └── trustgraph.py (NEW)
├── context_managers/
│   ├── base.py
│   ├── baseline.py
│   ├── memory_based.py
│   └── openai_parser.py (NEW)
├── benchmark/
│   └── harness.py (UPDATED)
├── config/
│   ├── baseline.json
│   ├── vector_db.json
│   ├── knowledge_graph.json
│   ├── muninndb.json (NEW)
│   ├── trustgraph.json (NEW)
│   └── openai_parser.json (NEW)
├── server.py (NEW)
├── data/test_dataset.py (NEW)
├── quickstart.sh (NEW)
├── test_memory_system.py (NEW)
├── test_new_system.py (NEW)
├── README.md (UPDATED)
├── DOCUMENTATION.md (NEW)
└── IMPLEMENTATION_SUMMARY.md (NEW)
```

## Integration

```python
# Use with any OpenAI-compatible API
from memory_stores.muninndb import MuninnDBStore

store = MuninnDBStore(
    api_url="http://localhost:8000",
    vault="default"
)

# Or use real API
store = MuninnDBStore(
    api_url="https://api.openai.com/v1",
    api_key="sk-...",
    vault="default"
)
```

## License

MIT
