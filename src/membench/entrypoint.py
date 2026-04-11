#!/usr/bin/env python3
"""
Entry point wrapper that suppresses warnings before importing membench.

This ensures Pydantic deprecation warnings from external libraries
(graphiti, letta) are suppressed before they can be triggered.
"""

import os
import sys
import warnings


def main():
    """Main entry point with warning suppression."""
    # Suppress Pydantic deprecation warnings BEFORE any membench imports
    # Use environment variable + warnings module for maximum coverage
    os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning,ignore::UserWarning"
    
    # Force ignore all deprecation and user warnings
    warnings.resetwarnings()
    warnings.simplefilter("ignore", DeprecationWarning)
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FutureWarning)
    
    # Also filter by message content
    warnings.filterwarnings("ignore", message=".*Pydantic.*")
    warnings.filterwarnings("ignore", message=".*deprecated.*", category=DeprecationWarning)
    
    # Now safe to import membench
    from membench.cli import main as cli_main
    return cli_main()


if __name__ == "__main__":
    sys.exit(main())
