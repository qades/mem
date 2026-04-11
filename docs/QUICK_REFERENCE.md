# Quick Reference Card

## Configuration

```python
from config.manager import ConfigManager, ContextManagerType, VectorStoreType

config_mgr = ConfigManager()

# Model config
config_mgr.load_model_config()  # api_url: localhost:58080/v1
config_mgr.save_model_config(model_config)

# Vector store config
config_mgr.load_vector_store_config()  # store_type: in_memory
config_mgr.save_vector_store_config(vector_config)

# Benchmark config
config_mgr.load_config("config/openai_parser.json")
```

## Run Benchmark

```python
from benchmark.harness import BenchmarkHarness, BenchmarkConfig

config = BenchmarkConfig(
    context_manager_type=ContextManagerType.OPENAI_PARSER,
    dataset_name="chatbot_conversations",
    max_messages=20,
    params={
        "api_url": "http://localhost:58080/v1",
        "parser_model": "LFM2.5-1.2B-Instruct-Q8_0",
    }
)

harness = BenchmarkHarness(config)
result = harness.run_benchmark(messages)
```

## OpenAI Parser

```python
from context_managers.openai_parser import OpenAICompatibleContextManager

parser = OpenAICompatibleContextManager(
    api_url="http://localhost:58080/v1",
    parser_model="LFM2.5-1.2B-Instruct-Q8_0",
    k_retrieval=5
)

parser.process_message({"role": "user", "content": "Hello"})
context = parser.get_context({"role": "user", "content": "What?"})
parser.reset()
```

## Vector DB

```python
from memory_stores.vector_db import VectorDBStore

# In-memory (default)
store = VectorDBStore()

# ChromaDB
store = VectorDBStore(store_type="chromadb")

# FAISS
store = VectorDBStore(store_type="faiss")

store.insert("user", "likes", "python")
results = store.retrieve("python", k=5)
```

## Scripts

```bash
# Quick setup
./quickstart.sh

# Run benchmark (20 messages default)
./run_benchmark.sh

# Custom excerpt size
./run_benchmark.sh 50

# Run tests
python test_new_system.py
```

## File Locations

- Config: `config/*.json`
- Data: `data/*.jsonl`
- Results: `benchmark_results/*.json`
- Tests: `test_new_system.py`

## Common Tasks

| Task | Command/Code |
|------|-------------|
| Load config | `config_mgr.load_config("config/openai_parser.json")` |
| Create parser | `OpenAICompatibleContextManager(api_url=...)` |
| Insert vector | `store.insert("e", "r", "v")` |
| Search | `store.retrieve("query", k=5)` |
| Run benchmark | `harness.run_benchmark(messages)` |
| Save results | `harness.save_results(results, path)` |

## API Endpoints

- Chat: `http://localhost:58080/v1/chat/completions`
- Embeddings: `http://localhost:58080/v1/embeddings`
- Models: `http://localhost:58080/v1/models`

## Strategy Types

- `ContextManagerType.BASELINE`
- `ContextManagerType.OPENAI_PARSER`
- `ContextManagerType.VECTOR_DB`
- `ContextManagerType.KNOWLEDGE_GRAPH`
- `ContextManagerType.MUNINNDB`
- `ContextManagerType.TRUSTGRAPH`

## Vector Store Types

- `VectorStoreType.IN_MEMORY`
- `VectorStoreType.CHROMADB`
- `VectorStoreType.FAISS`
