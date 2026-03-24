# Context Management Benchmark

A benchmark harness to compare different context management strategies for LLM conversations.

## Overview

This project implements and benchmarks two approaches to context management:

1. **Baseline**: Sends all previous conversation messages with the current one (full history)
2. **Memory-based**: Parses messages, extracts information, stores in a memory system, and retrieves relevant context

## Project Structure

```
mem/
├── benchmark/          # Benchmark harness implementation
├── context_managers/   # Context manager strategies
├── memory_stores/      # Memory storage implementations
├── datasets/           # Test datasets
├── eval/              # Evaluation metrics
├── config/            # Configuration files
├── run_benchmark.py   # Entry point
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run a single benchmark

```bash
python run_benchmark.py --config config/baseline.json
python run_benchmark.py --config config/knowledge_graph.json
python run_benchmark.py --config config/vector_db.json
```

### Compare all strategies

```bash
python run_benchmark.py --compare
```

### List available datasets

```bash
python run_benchmark.py --list-datasets
```

## Benchmark Results

The benchmark measures:
- **Context size**: Number of tokens in the final context
- **Response time**: Time to process messages and retrieve context
- **Memory usage**: Memory footprint during execution
- **Metrics**: Evaluation metrics (BLEU, ROUGE-L, exact match, semantic similarity)

## Adding New Strategies

1. Implement `BaseContextManager` in `context_managers/`
2. Implement `BaseMemoryStore` in `memory_stores/` (if needed)
3. Register the new type in `ContextManagerType` enum
4. Update the harness to use the new implementation

## License

MIT
