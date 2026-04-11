#!/bin/bash
#
# MemBench Environment Initialization Script
#
# Source this script before using membench to suppress Pydantic deprecation
# warnings from external libraries (graphiti, letta).
#
# Usage:
#   source init.sh
#   membench list-stores
#
# Or make it permanent by adding to your ~/.bashrc or ~/.zshrc:
#   export PYTHONWARNINGS="ignore::pydantic.warnings.PydanticDeprecatedSince20"
#

# Suppress Pydantic V2 deprecation warnings
export PYTHONWARNINGS="ignore::pydantic.warnings.PydanticDeprecatedSince20,ignore::DeprecationWarning,ignore::UserWarning"

# Optional: Set default API keys (uncomment and fill in)
# export OPENAI_API_KEY="your-key-here"
# export ANTHROPIC_API_KEY="your-key-here"

# Optional: Set default infrastructure URLs for Docker services
# export NEO4J_URI="bolt://localhost:7687"
# export NEO4J_USER="neo4j"
# export NEO4J_PASSWORD="password"
# export QDRANT_URL="http://localhost:6333"
# export CHROMADB_URL="http://localhost:8000"

echo "MemBench environment initialized."
echo "Pydantic warnings suppressed."
echo ""
echo "Available commands:"
echo "  membench list-stores      - List available memory stores"
echo "  membench list-datasets    - List available datasets"
echo "  membench run --config quick  - Run quick benchmark"
