# Memory/Context Management System

This system provides multiple context management strategies for LLM conversations, with support for:
- **Baseline**: Full conversation history
- **OpenAI-compatible Parser**: Fast context dissection using LLM API
- **Vector DB**: Vector similarity-based retrieval (ChromaDB, FAISS, in-memory)
- **Knowledge Graph**: Relationship-based retrieval
- **MuninnDB**: Memory store with OpenAI-compatible API
- **TrustGraph**: Benchmarkable data stores

## Configuration

### API Endpoints
- Default: `http://localhost:58080/v1`
- Chat model: `Qwen3-Coder-Next-Q4_K_M`
- Parser model: `LFM2.5-1.2B-Instruct-Q8_0`

### Configuration Files
- `config/model.json`: Model settings
- `config/vector_store.json`: Vector store backend (ChromaDB/FAISS/in-memory)
- `config/baseline.json`: Baseline strategy
- `config/openai_parser.json`: OpenAI parser settings
- `config/vector_db.json`: Vector DB settings
- `config/knowledge_graph.json`: Knowledge graph settings

## Quick Start

```bash
# Run quick setup
./quickstart.sh

# Run benchmark with 20 messages
./run_benchmark.sh 20

# Run benchmark with custom excerpt size
./run_benchmark.sh 50 conversation_dataset
```

## Python API

### Configuration
```python
from config.manager import ConfigManager, ContextManagerType

config_mgr = ConfigManager()
config = config_mgr.load_config("config/openai_parser.json")
```

### Context Manager
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

## Vector Store Backends

- **in_memory**: Default, fast, no dependencies
- **ChromaDB**: Install `pip install chromadb`
- **FAISS**: Install `pip install faiss`

## Dataset Format

JSONL format with role/content:
```json
{"role": "user", "content": "Hello"}
{"role": "assistant", "content": "Hi there"}
```

## Output

Results saved to `benchmark_results/`:
- Performance metrics
- Context sizes
- Response times
- Memory usage
