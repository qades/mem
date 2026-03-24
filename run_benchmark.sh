#!/bin/bash

# Run benchmark with configurable excerpt size
# Usage: ./run_benchmark.sh [num_messages] [dataset_name]

NUM_MESSAGES=${1:-20}
DATASET_NAME=${2:-chatbot_conversations}

echo "=========================================="
echo "Running Benchmark"
echo "=========================================="
echo "Messages: $NUM_MESSAGES"
echo "Dataset: $DATASET_NAME"
echo ""

# Create test dataset if needed
if [ ! -f "data/test_dataset.jsonl" ]; then
    echo "Creating test dataset with $NUM_MESSAGES messages..."
    python3 -c "
import sys
sys.path.insert(0, '.')
from data.test_dataset import create_test_dataset
create_test_dataset('data/test_dataset.jsonl', num_messages=$NUM_MESSAGES)
"
fi

# Run benchmark for each strategy
echo ""
echo "Running benchmarks..."
echo ""

python3 -c "
import sys
import json
sys.path.insert(0, '.')
from config.manager import ConfigManager, ContextManagerType
from benchmark.harness import BenchmarkHarness, BenchmarkConfig
from data.test_dataset import load_test_dataset

# Load test dataset
messages, references = load_test_dataset('data/test_dataset.jsonl', max_messages=$NUM_MESSAGES)
print(f'Loaded {len(messages)} messages')

# Config manager
config_mgr = ConfigManager()

# Run each strategy
strategies = [
    (ContextManagerType.BASELINE, {}),
    (ContextManagerType.OPENAI_PARSER, {
        'api_url': 'http://localhost:58080/v1',
        'model': 'LFM2.5-1.2B-Instruct-Q8_0',
        'enable_benchmarking': True,
    }),
    (ContextManagerType.VECTOR_DB, {
        'store_type': 'in_memory',
        'k_retrieval': 5,
    }),
]

results = []
for strategy, params in strategies:
    print(f'\n--- Testing {strategy.value} ---')
    
    config = BenchmarkConfig(
        context_manager_type=strategy,
        dataset_name='test_dataset',
        max_messages=$NUM_MESSAGES,
        enable_metrics=False,
        params=params,
    )
    
    harness = BenchmarkHarness(config)
    result = harness.run_benchmark(messages)
    
    print(f'Context size: {result.context_size} tokens')
    print(f'Response time: {result.response_time_ms:.2f}ms')
    print(f'Memory usage: {result.memory_usage_mb:.2f}MB')
    
    if hasattr(result, 'metadata') and result.metadata:
        print(f'Metadata: {json.dumps(result.metadata, indent=2)}')
    
    results.append(result)

# Save results
output_path = f'benchmark_results/benchmark_{len(messages)}_messages.json'
with open(output_path, 'w') as f:
    json.dump([r.__dict__ for r in results], f, indent=2)

print(f'\nResults saved to {output_path}')
"

echo ""
echo "=========================================="
echo "Benchmark complete!"
echo "=========================================="
