# MemBench: Memory Benchmark Suite

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive benchmarking suite for evaluating long-term memory approaches in LLM-based agents.

## Features

- **9+ Memory Stores**: Mem0, Zep, Graphiti, Letta (MemGPT), MemPalace, MuninnDB, VectorDB, Knowledge Graph, TrustGraph
- **Multiple Datasets**: LoCoMo, BABILong, MuTual, AgentBench, ProLong
- **Docker Support**: Complete infrastructure with Neo4j, Qdrant, ChromaDB
- **Fair Comparison**: All stores implement the same `BaseMemoryStore` interface
- **Extensible**: Easy to add new memory stores and datasets

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/membench.git
cd membench

# Install with all memory store dependencies
pip install -e ".[all]"

# Or install selectively
pip install -e ".[mem0,zep,letta]"

# Initialize environment (suppresses Pydantic warnings from external libs)
source init.sh
```

### Using Docker (Recommended)

```bash
# Start infrastructure services
docker-compose -f docker/docker-compose.yml up -d neo4j qdrant chromadb

# Run quick smoke test
docker-compose -f docker/docker-compose.yml --profile benchmark run benchmark

# Or run with custom config
docker-compose -f docker/docker-compose.yml --profile benchmark run -e BENCHMARK_CONFIG=full benchmark
```

### CLI Usage

```bash
# Initialize environment first (suppresses warnings)
source init.sh

# List available memory stores
membench list-stores

# List available datasets
membench list-datasets

# Run quick benchmark
membench run --config quick

# Run specific stores on specific datasets
membench run --stores mem0 zep --datasets locomo babilong
```

**Note:** Some memory libraries (graphiti, letta) emit Pydantic V2 deprecation warnings at import time that cannot be suppressed via `PYTHONWARNINGS`. Use `membench-quiet` (after sourcing `init.sh`) or append `2>/dev/null` to hide them.

## Project Structure

```
membench/
├── src/membench/              # Main package
│   ├── memory_stores/         # Memory store implementations
│   │   ├── mem0_store.py      # Mem0 (hybrid: vector+graph+KV)
│   │   ├── zep_store.py       # Zep (temporal knowledge graph)
│   │   ├── graphiti_store.py  # Graphiti (temporal KG)
│   │   ├── letta_store.py     # Letta/OS-style hierarchical
│   │   ├── mempalace_store.py # MemPalace (spatial/method of loci)
│   │   ├── muninndb.py        # MuninnDB
│   │   ├── vector_db.py       # Vector DB (Chroma/FAISS)
│   │   ├── knowledge_graph.py # Knowledge Graph
│   │   └── trustgraph.py      # TrustGraph
│   ├── context_managers/      # Context management strategies
│   ├── benchmark/             # Benchmark harness
│   ├── config/                # Configuration management
│   └── eval/                  # Evaluation metrics
├── scripts/                   # Utility scripts
│   ├── coordinator.py         # Benchmark orchestration
│   ├── download_datasets.py   # Dataset downloader
│   └── compress_prolong.py    # Data compression
├── docker/                    # Docker infrastructure
│   ├── docker-compose.yml     # Infrastructure services
│   ├── Dockerfile             # Benchmark runner image
│   └── data/                  # Persistent data volumes
├── configs/                   # Benchmark configurations
├── tests/                     # Test suite
├── data/                      # Datasets
└── docs/                      # Documentation
```

## Memory Stores Comparison

| Store | Architecture | Key Feature | Install |
|-------|-------------|-------------|---------|
| **Mem0** | Hybrid (vector+graph+KV) | Auto fact extraction | `pip install mem0ai` |
| **Zep** | Temporal KG | Bi-temporal queries | `pip install zep-python` |
| **Graphiti** | Temporal KG | Episode-based | `pip install graphiti-core` |
| **Letta** | Hierarchical | Self-managing agent | `pip install letta` |
| **MemPalace** | Spatial | Method of loci | `pip install mempalace` |

## Configuration

Benchmark configurations are JSON files in `configs/`:

```json
{
  "name": "My Benchmark",
  "stores": ["mem0", "zep", "letta"],
  "datasets": ["locomo", "babilong"],
  "max_messages": 50,
  "k_retrieval": 5
}
```

Run with: `membench run --config configs/my_benchmark.json`

## Development

```bash
# Install in development mode
pip install -e ".[all,dev]"

# Run tests
pytest tests/

# Format code
black src/ tests/ scripts/

# Build Docker image
docker build -f docker/Dockerfile -t membench:latest .
```

## Documentation

- [Full Documentation](docs/DOCUMENTATION.md)
- [Implementation Summary](docs/IMPLEMENTATION_SUMMARY.md)
- [Next Steps](docs/NEXT_STEPS.md)

## License

MIT License - see [LICENSE](LICENSE) file.

## Citation

```bibtex
@software{membench2024,
  title={MemBench: Memory Benchmark Suite for LLM Agents},
  author={Memory Benchmark Team},
  year={2024},
  url={https://github.com/yourusername/membench}
}
```
