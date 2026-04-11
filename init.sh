#!/bin/bash
#
# MemBench Environment Initialization Script
#
# Source this script before using membench to set up the environment.
#
# Usage:
#   source init.sh
#   membench list-stores              # Shows warnings
#   membench-quiet list-stores        # Hides warnings (stderr filtered)
#
# Or make it permanent by adding to your ~/.bashrc or ~/.zshrc:
#   source /path/to/membench/init.sh
#

# Optional: Set default API keys (uncomment and fill in)
# export OPENAI_API_KEY="your-key-here"
# export ANTHROPIC_API_KEY="your-key-here"

# Optional: Set default infrastructure URLs for Docker services
# export NEO4J_URI="bolt://localhost:7687"
# export NEO4J_USER="neo4j"
# export NEO4J_PASSWORD="password"
# export QDRANT_URL="http://localhost:6333"
# export CHROMADB_URL="http://localhost:8000"

# Wrapper function to run membench without Pydantic warnings
# Note: Some external libraries (graphiti, letta) emit Pydantic V2 
# deprecation warnings at import time that cannot be suppressed via 
# PYTHONWARNINGS. This wrapper filters stderr to hide them.
membench-quiet() {
    membench "$@" 2>/dev/null
}

# Alternative: Only filter lines containing warning patterns (filters both stdout and stderr)
membench-clean() {
    membench "$@" 2>&1 | grep -v -E "PydanticDeprecated|deprecat|warning:|class SearchInterface|class ToolCallMessage|warnings\.warn|type: ResponseFormatType"
}

echo "MemBench environment initialized."
echo ""
echo "Available commands:"
echo "  membench list-stores              - List memory stores (shows warnings)"
echo "  membench list-datasets            - List datasets (shows warnings)"
echo "  membench run --config quick       - Run quick benchmark (shows warnings)"
echo ""
echo "Warning-free alternatives:"
echo "  membench-quiet list-stores        - Same but hides all stderr"
echo "  membench-clean list-stores        - Same but filters only warnings"
echo ""
echo "Or use: membench list-stores 2>/dev/null"
echo ""
echo "Note: Pydantic V2 warnings come from external libraries (graphiti, letta)"
echo "      at import time and cannot be suppressed via PYTHONWARNINGS."
