#!/usr/bin/env python
"""
Convenience script to run the test suite.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py -k scores    # Run only score tests
    python run_tests.py -v           # Verbose output
    python run_tests.py --cov        # With coverage (requires pytest-cov)
"""

import sys
import pytest

if __name__ == "__main__":
    # Default args if none provided
    args = sys.argv[1:] if len(sys.argv) > 1 else ["-v"]
    
    # Run pytest
    exit_code = pytest.main(["tests/"] + args)
    sys.exit(exit_code)
