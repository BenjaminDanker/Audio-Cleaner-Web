#!/usr/bin/env python3
"""
Test runner for streaming service tests.

This script runs all tests without requiring external dependencies like
Azure SDK, Cosmos DB, or environment variables.
"""
import sys
import os
import subprocess
from pathlib import Path

# Add the current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    """Run all tests with proper reporting."""
    print("🧪 Running Audio Cleaner Streaming Service Tests")
    print("=" * 60)
    
    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        print("❌ pytest not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytest"])
        import pytest
    
    # Test files to run
    test_files = [
        "test_auth.py",
        "test_security.py", 
        "test_session.py",
        "test_billing.py"
    ]
    
    # Run tests with verbose output
    args = [
        "-v",           # Verbose output
        "--tb=short",   # Short traceback format
        "--color=yes",  # Colored output
        "-x",           # Stop on first failure (optional)
    ] + test_files
    
    print(f"Running tests: {', '.join(test_files)}")
    print("-" * 60)
    
    # Run pytest
    exit_code = pytest.main(args)
    
    print("-" * 60)
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ Tests failed with exit code {exit_code}")
    
    return exit_code


def run_single_test(test_name: str):
    """Run a single test file."""
    try:
        import pytest
    except ImportError:
        print("❌ pytest not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytest"])
        import pytest
    
    print(f"🧪 Running {test_name}")
    print("=" * 40)
    
    exit_code = pytest.main(["-v", "--tb=short", test_name])
    
    if exit_code == 0:
        print(f"✅ {test_name} passed!")
    else:
        print(f"❌ {test_name} failed!")
    
    return exit_code


def list_tests():
    """List all available tests."""
    test_files = [f for f in os.listdir(".") if f.startswith("test_") and f.endswith(".py")]
    
    print("📋 Available tests:")
    for test_file in sorted(test_files):
        print(f"  - {test_file}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "list":
            list_tests()
        elif command.startswith("test_"):
            # Run specific test
            exit_code = run_single_test(command)
            sys.exit(exit_code)
        else:
            print(f"Unknown command: {command}")
            print("Usage: python run_tests.py [list|test_<name>.py]")
            sys.exit(1)
    else:
        # Run all tests
        exit_code = main()
        sys.exit(exit_code)
