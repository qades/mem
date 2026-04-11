#!/usr/bin/env python3
"""
Entry point for membench CLI.

This module handles warning suppression before importing the main CLI.
"""

import os
import sys

# Suppress Pydantic deprecation warnings BEFORE any imports
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning,ignore::UserWarning"

# Now import and run the CLI
from membench.cli import main

if __name__ == "__main__":
    sys.exit(main())
