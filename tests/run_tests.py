#!/usr/bin/env python3
"""
Test runner for agent-forge tests.

Usage:
    python tests/run_tests.py          # Run all tests
    python tests/run_tests.py unit      # Run only unit tests
    python tests/run_tests.py integration  # Run only integration tests
"""

import subprocess
import sys
from pathlib import Path


def run_tests(test_type="all"):
    """Run tests of the specified type."""
    project_root = Path(__file__).resolve().parent.parent
    project_root = str(project_root)

    base = [sys.executable, "-m", "pytest", "-v", "--tb=short"]
    if test_type == "all":
        print("Running all tests...")  # noqa: T201
        result = subprocess.run([*base, "tests/"], check=False, cwd=project_root)
    elif test_type == "unit":
        print("Running unit tests...")  # noqa: T201
        result = subprocess.run([*base, "tests/unit/"], check=False, cwd=project_root)
    elif test_type == "integration":
        print("Running integration tests...")  # noqa: T201
        result = subprocess.run([*base, "tests/integration/"], check=False, cwd=project_root)
    else:
        print(f"Unknown test type: {test_type}")  # noqa: T201
        print("Usage: python tests/run_tests.py [all|unit|integration]")  # noqa: T201
        sys.exit(1)

    return result.returncode


if __name__ == "__main__":
    test_type = sys.argv[1] if len(sys.argv) > 1 else "all"
    exit_code = run_tests(test_type)
    sys.exit(exit_code)
