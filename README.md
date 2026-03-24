# Context Management Benchmark with MuninnDB, TrustGraph-AI, and OpenAI-Compatible API

A benchmark harness to compare different context management strategies for LLM conversations, featuring:
- **MuninnDB**: Persistent memory with OpenAI-compatible API
- **TrustGraph-AI**: Benchmarkable data structures with performance tracking
- **OpenAI-Compatible API**: Fast utility LLM for context parsing

## Overview

This project implements and benchmarks multiple approaches to context management:

1. **Baseline**: Sends all previous conversation messages with the current one (full history)
2. **Memory-based**: Parses messages, extracts information, stores in a memory system, and retrieves relevant context
3. **MuninnDB**: Persistent memory using OpenAI-compatible API for context parsing
4. **TrustGraph-AI**: Memory with benchmarkable data structures and performance tracking
5. **OpenAI Parser**: Fast context parsing using OpenAI-compatible LLM API

## Project Structure

```
mem/
├── memory_stores/      # Memory storage implementations
│   ├── base.py         # Base interface
│   ├── vector_db.py    # Vector DB memory store
│   ├── knowledge_graph.py  # Knowledge graph memory store
│   ├── muninndb.py     # MuninnDB memory store
│   └── trustgraph.py   # TrustGraph-AI memory store
├── context_managers/   # Context manager strategies
│   ├── base.py         # Base interface
│   ├── baseline.py     # Baseline strategy
│   ├── memory_based.py # Memory-based strategy
│   └── openai_parser.py # OpenAI-compatible parser
├── benchmark/          # Benchmark harness implementation
├── config/             # Configuration files
├── data/               # Test datasets
├── server.py           # OpenAI-compatible test server
├── requirements.txt    # Dependencies
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

### Running the OpenAI-Compatible Test Server

For testing with real API calls, start the OpenAI-compatible test server:

```bash
python3 server.py --port 8000
```

This server provides:
- `/health` - Health check
- `/chat/completions` - Chat completions with context parsing
- `/embeddings` - Embedding generation
- `/memories` - Memory CRUD operations
- `/search` - Semantic search
- `/stats` - Statistics

## Usage

### Run a Single Benchmark

```bash
python3 run_benchmark.py --config config/baseline.json
python3 run_benchmark.py --config config/vector_db.json
python3 run_benchmark.py --config config/knowledge_graph.json
python3 run_benchmark.py --config config/muninndb.json
python3 run_benchmark.py --config config/trustgraph.json
python3 run_benchmark.py --config config/openai_parser.json
```

### Compare All Strategies

```bash
python3 run_benchmark.py --compare
```

### List Available Datasets

```bash
python3 run_benchmark.py --list-datasets
```

### Run Tests

```bash
python3 test_memory_system.py
```

## Configuration

Configuration files are in the `config/` directory:

- `baseline.json` - Baseline strategy (full history)
- `vector_db.json` - Vector DB memory store
- `knowledge_graph.json` - Knowledge graph memory store
- `muninndb.json` - MuninnDB with OpenAI-compatible API
- `trustgraph.json` - TrustGraph-AI with benchmarking
- `openai_parser.json` - OpenAI-compatible context parser

### Configuration Format

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

## Benchmark Results

The benchmark measures:
- **Context size**: Number of tokens in the final context
- **Response time**: Time to process messages and retrieve context
- **Memory usage**: Memory footprint during execution
- **Metrics**: Evaluation metrics (BLEU, ROUGE-L, exact match, semantic similarity)

### Example Results

```
vector_db:
  Context: 500 tokens
  Time: 16.76ms

trustgraph:
  Context: 500 tokens
  Time: 1075.84ms (includes benchmarking overhead)

muninndb:
  Context: 500 tokens
  Time: 23.69ms

openai_parser:
  Context: 500 tokens
  Time: 10.68ms
```

## Adding New Strategies

1. Implement `BaseContextManager` in `context_managers/`
2. Implement `BaseMemoryStore` in `memory_stores/` (if needed)
3. Register the new type in `ContextManagerType` enum
4. Update the harness to use the new implementation

## Memory Store Interfaces

### BaseMemoryStore

```python
class BaseMemoryStore(ABC):
    @abstractmethod
    def insert(entity, relation, value, metadata=None) -> str:
        """Insert a piece of information into memory."""
    
    @abstractmethod
    def retrieve(query, k=5, use_embedding=True) -> List[Dict[str, Any]]:
        """Retrieve relevant information from memory."""
    
    @abstractmethod
    def update(id, entity=None, relation=None, value=None) -> bool:
        """Update existing information."""
    
    @abstractmethod
    def delete(id) -> bool:
        """Delete information from memory."""
    
    @abstractmethod
    def clear() -> None:
        """Clear all memory."""
    
    @abstractmethod
    def get_stats() -> Dict[str, Any]:
        """Get memory store statistics."""
```

## OpenAI-Compatible API

The system works with any OpenAI-compatible API server. The test server (`server.py`) provides a minimal implementation that mimics OpenAI's API for development/testing.

### Supported Endpoints

- `GET /health` - Health check
- `POST /chat/completions` - Chat completions with context parsing
- `POST /embeddings` - Embedding generation
- `POST /memories` - Create memory
- `POST /memories/update` - Update memory
- `POST /memories/delete` - Delete memory
- `POST /memories/clear` - Clear memories
- `POST /search` - Semantic search
- `GET /stats` - Get statistics

## License

MIT
