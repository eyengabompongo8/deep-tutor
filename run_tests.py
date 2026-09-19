#!/usr/bin/env python3
"""Convenience test runner wrapper around pytest for FrenchTutor.

Usage:
    python run_tests.py          # Run fast unit & API tests (excludes ML models)
    python run_tests.py --all    # Run all tests including ML models
    python run_tests.py --ml     # Run ML model tests only
    python run_tests.py <args>   # Pass through any pytest arguments
"""

import sys
import pytest


def main():
    args = sys.argv[1:]
    pytest_args = []
    
    if "--all" in args:
        args.remove("--all")
        pytest_args.extend(["-m", ""])
    elif "--ml" in args:
        args.remove("--ml")
        pytest_args.extend(["-m", "ml"])
        
    pytest_args.extend(args)
    sys.exit(pytest.main(pytest_args))


if __name__ == "__main__":
    main()
