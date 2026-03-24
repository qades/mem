#!/bin/bash
# Quick-start script for the memory management system

set -e

echo "============================================================"
echo "Memory Management System - Quick Start"
echo "============================================================"
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Install dependencies
echo ""
echo "Installing dependencies..."
pip3 install -r requirements.txt --quiet --break-system-packages

# Create test dataset
echo ""
echo "Creating test dataset..."
python3 -c "from data.test_dataset import create_test_dataset; create_test_dataset()"

# Run tests
echo ""
echo "Running tests..."
python3 test_memory_system.py

echo ""
echo "============================================================"
echo "To run benchmarks:"
echo "  python3 run_benchmark.py --config config/vector_db.json"
echo "  python3 run_benchmark.py --config config/muninndb.json"
echo "  python3 run_benchmark.py --config config/trustgraph.json"
echo ""
echo "To start OpenAI-compatible test server:"
echo "  python3 server.py --port 8000"
echo "============================================================"
