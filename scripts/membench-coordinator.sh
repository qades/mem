#!/bin/bash
#
# MemBench Coordinator Wrapper
#
# Runs the coordinator script with proper Python path setup.
#

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Run coordinator
PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH" \
    python3 "$PROJECT_ROOT/scripts/coordinator.py" "$@"
