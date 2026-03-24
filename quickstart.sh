#!/bin/bash

# Quickstart script for the memory/context management system
# This script sets up and runs a quick benchmark test

set -e

echo "=========================================="
echo "Memory/Context Management System Setup"
echo "=========================================="

# Create directories
mkdir -p data benchmark_results

# Create test dataset (20 messages)
echo "Creating test dataset with 20 messages..."
python -c "
import sys
sys.path.insert(0, '.')
from data.test_dataset import create_test_dataset
create_test_dataset('data/test_dataset.jsonl', num_messages=20)
"

# Run configuration tests
echo ""
echo "Testing configuration..."
python -c "
import sys
sys.path.insert(0, '.')
from config.manager import ConfigManager, ContextManagerType, VectorStoreType

config_mgr = ConfigManager()

# Check model config
model_config = config_mgr.load_model_config()
print(f'✓ Model config loaded: {model_config.chat_model}')
assert model_config.api_url == 'http://localhost:58080/v1'

# Check vector store config
vector_config = config_mgr.load_vector_store_config()
print(f'✓ Vector store config: {vector_config.store_type.value}')

# Check benchmark configs
for config_name in ['baseline', 'openai_parser', 'vector_db', 'knowledge_graph', 'muninndb', 'trustgraph']:
    config = config_mgr.load_config(f'config/{config_name}.json')
    print(f'✓ {config_name} config loaded')
"

# Run quick test
echo ""
echo "Running quick system test..."
python test_new_system.py

echo ""
echo "=========================================="
echo "Setup complete! Ready for benchmark runs."
echo "=========================================="
echo ""
echo "To run benchmarks:"
echo "  python benchmark/harness.py"
echo ""
echo "Or with a specific config:"
echo "  python -c \"from benchmark.harness import BenchmarkHarness, BenchmarkConfig; from config.manager import ContextManagerType; h = BenchmarkHarness(BenchmarkConfig(ContextManagerType.OPENAI_PARSER, 'chatbot_conversations', max_messages=20, params={'api_url': 'http://localhost:58080/v1'})); print(h.run_benchmark([{'role': 'user', 'content': 'test'}]))\""
