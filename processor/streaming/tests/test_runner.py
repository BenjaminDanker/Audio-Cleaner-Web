#!/usr/bin/env python3
"""
Standalone test runner for Audio Cleaner Streaming Service.

This script tests the core functionality without requiring:
- Docker containers
- Azure SDK connections
- Environment variables (uses safe defaults)
- External services (Cosmos DB, Azure OpenAI, etc.)

Perfect for development and CI/CD pipelines.
"""
import sys
import os
import subprocess
from pathlib import Path
import tempfile
import json

def ensure_dependencies():
    """Ensure test dependencies are installed."""
    print("📦 Checking test dependencies...")
    
    required_packages = [
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0", 
        "numpy>=1.20.0"
    ]
    
    for package in required_packages:
        try:
            import importlib
            pkg_name = package.split(">=")[0].split("==")[0]
            importlib.import_module(pkg_name.replace("-", "_"))
            print(f"  ✅ {pkg_name}")
        except ImportError:
            print(f"  📥 Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"  ✅ {pkg_name} installed")


def run_unit_tests():
    """Run unit tests for core modules."""
    print("\n🧪 Running Unit Tests")
    print("=" * 50)
    
    test_dir = Path(__file__).parent
    
    # Import pytest after ensuring it's installed
    import pytest
    
    # Test files that don't require external dependencies
    test_files = [
        test_dir / "test_auth.py",
        test_dir / "test_security.py",
        test_dir / "test_session.py", 
        test_dir / "test_billing.py"
    ]
    
    # Filter to only existing files
    existing_tests = [f for f in test_files if f.exists()]
    
    if not existing_tests:
        print("❌ No test files found!")
        return False
    
    print(f"Running {len(existing_tests)} test files...")
    
    # Run tests with detailed output
    args = [
        "-v",                    # Verbose
        "--tb=short",           # Short traceback
        "--disable-warnings",   # Hide pytest warnings  
        "-q",                   # Quiet mode
        *[str(f) for f in existing_tests]
    ]
    
    exit_code = pytest.main(args)
    
    if exit_code == 0:
        print("✅ All unit tests passed!")
        return True
    else:
        print(f"❌ Unit tests failed (exit code: {exit_code})")
        return False


def run_integration_tests():
    """Run integration tests for API endpoints."""
    print("\n🌐 Running Integration Tests")
    print("=" * 50)
    
    test_dir = Path(__file__).parent
    api_test_file = test_dir / "test_api.py"
    
    if not api_test_file.exists():
        print("⚠️  No API test file found, skipping...")
        return True
    
    import pytest
    
    # Need additional dependencies for API tests
    try:
        import fastapi
        import httpx
    except ImportError:
        print("📥 Installing API test dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "fastapi>=0.100.0", "httpx>=0.24.0"
        ], stdout=subprocess.DEVNULL)
    
    print("Testing FastAPI endpoints...")
    
    args = [
        "-v",
        "--tb=short", 
        "--disable-warnings",
        str(api_test_file)
    ]
    
    exit_code = pytest.main(args)
    
    if exit_code == 0:
        print("✅ All integration tests passed!")
        return True
    else:
        print(f"❌ Integration tests failed (exit code: {exit_code})")
        return False


def run_security_tests():
    """Run security-focused tests."""
    print("\n🔒 Running Security Tests")
    print("=" * 50)
    
    # Test authentication
    print("🔐 Testing authentication...")
    try:
        # Import and test auth module directly
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from auth import verify_session_token, _b64url_no_pad, _b64url_decode
        
        # Quick smoke test
        test_data = b"test"
        encoded = _b64url_no_pad(test_data)
        decoded = _b64url_decode(encoded)
        assert decoded == test_data
        print("  ✅ Base64url utilities working")
        
        # Test invalid token rejection (no key configured)
        result = verify_session_token("invalid.token", "session")
        assert result is False
        print("  ✅ Invalid tokens properly rejected")
        
    except Exception as e:
        print(f"  ❌ Auth test failed: {e}")
        return False
    
    # Test session validation
    print("📝 Testing input validation...")
    try:
        from security import validate_session_id, SESSION_ID_PATTERN
        
        # Valid session IDs
        valid_ids = ["test123", "session-456", "user_session"]
        for sid in valid_ids:
            assert SESSION_ID_PATTERN.match(sid)
        print("  ✅ Valid session IDs accepted")
        
        # Invalid session IDs
        invalid_ids = ["", "session with spaces", "session@invalid"]
        for sid in invalid_ids:
            assert not SESSION_ID_PATTERN.match(sid)
        print("  ✅ Invalid session IDs rejected")
        
    except Exception as e:
        print(f"  ❌ Validation test failed: {e}")
        return False
    
    print("✅ Security tests passed!")
    return True


def generate_test_report():
    """Generate a test report summary."""
    print("\n📊 Test Report Summary")
    print("=" * 50)
    
    report_data = {
        "timestamp": "2025-08-17T12:00:00Z",
        "test_suite": "Audio Cleaner Streaming Service",
        "coverage_areas": [
            "Authentication & Token Verification",
            "Session State Management", 
            "Security Validation & Rate Limiting",
            "Billing Logic (without external calls)",
            "API Endpoint Structure",
            "Input Validation & Sanitization"
        ],
        "external_dependencies_mocked": [
            "Azure Cosmos DB",
            "Azure OpenAI",
            "Azure Service Bus",
            "Environment Variables"
        ],
        "security_features_tested": [
            "HMAC token verification",
            "Session ID validation",
            "Rate limiting configuration",
            "CORS setup",
            "Input sanitization",
            "Connection limiting",
            "Origin validation"
        ]
    }
    
    print(f"🎯 Coverage Areas ({len(report_data['coverage_areas'])}):")
    for area in report_data['coverage_areas']:
        print(f"  ✅ {area}")
    
    print(f"\n🔒 Security Features Tested ({len(report_data['security_features_tested'])}):")
    for feature in report_data['security_features_tested']:
        print(f"  🛡️  {feature}")
    
    print(f"\n🏗️  Mocked Dependencies ({len(report_data['external_dependencies_mocked'])}):")
    for dep in report_data['external_dependencies_mocked']:
        print(f"  🎭 {dep}")
    
    print("\n💡 What These Tests DON'T Cover:")
    print("  ⚠️  Actual Azure SDK calls")
    print("  ⚠️  Real WebSocket connections")
    print("  ⚠️  Audio processing pipeline")
    print("  ⚠️  Network-level security")
    print("  ⚠️  Container security")
    
    return report_data


def main():
    """Run the complete test suite."""
    print("🚀 Audio Cleaner Streaming Service Test Suite")
    print("=" * 60)
    print("Testing core functionality without external dependencies")
    print("-" * 60)
    
    # Ensure we can run tests
    try:
        ensure_dependencies()
    except Exception as e:
        print(f"❌ Failed to install dependencies: {e}")
        return 1
    
    # Run test phases
    results = []
    
    try:
        results.append(("Security Tests", run_security_tests()))
        results.append(("Unit Tests", run_unit_tests())) 
        results.append(("Integration Tests", run_integration_tests()))
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return 1
    
    # Generate report
    try:
        generate_test_report()
    except Exception as e:
        print(f"⚠️  Report generation failed: {e}")
    
    # Summary
    print(f"\n🏁 Test Results Summary")
    print("=" * 30)
    
    passed = sum(1 for name, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")
    
    print(f"\nOverall: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 All tests passed! The streaming service core is working correctly.")
        return 0
    else:
        print("💔 Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
