#!/usr/bin/env python3
"""Dedicated test runner script for FrenchTutor.

Usage:
    python run_tests.py          # Run fast unit and API tests (default, excludes slow ML models)
    python run_tests.py --all    # Run all tests including heavy ML models
    python run_tests.py --ml     # Run only ML model tests
    python run_tests.py <path>   # Run specific test file or directory
"""

import os
import sys
import argparse
import pytest

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


def main():
    parser = argparse.ArgumentParser(
        description="Run FrenchTutor test suite with automatic categorization."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests including heavy ML model tests",
    )
    parser.add_argument(
        "--ml",
        action="store_true",
        help="Run only heavy ML model tests",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Run pytest in verbose mode",
    )
    parser.add_argument(
        "-k",
        type=str,
        default=None,
        help="Run tests matching given keyword expression",
    )
    parser.add_argument(
        "targets",
        nargs="*",
        default=[],
        help="Specific test file(s) or pattern(s) to run",
    )

    args, extra_args = parser.parse_known_args()

    pytest_args = []

    if args.verbose:
        pytest_args.append("-v")

    if args.k:
        pytest_args.extend(["-k", args.k])

    if args.all:
        pytest_args.extend(["-m", ""])
    elif args.ml:
        pytest_args.extend(["-m", "ml"])
    # Default is handled by pyproject.toml addopts (-m 'not ml')

    if args.targets:
        pytest_args.extend(args.targets)

    pytest_args.extend(extra_args)

    print(f"🚀 Running FrenchTutor test suite with: pytest {' '.join(pytest_args)}")
    exit_code = pytest.main(pytest_args)
    import os
    os._exit(int(exit_code))


if __name__ == "__main__":
    main()
