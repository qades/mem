#!/bin/bash
#
# MemBench CLI wrapper script
#
# This script suppresses Pydantic deprecation warnings from external libraries
# (graphiti, letta) and then calls the Python CLI.
#
# Usage:
#   membench list-stores
#   membench list-datasets
#   membench run --config quick
#

# Suppress Pydantic deprecation warnings
export PYTHONWARNINGS="ignore::DeprecationWarning,ignore::UserWarning,ignore::pydantic.warnings.PydanticDeprecatedSince20"

# Run the Python CLI
exec python -m membench.cli "$@"
