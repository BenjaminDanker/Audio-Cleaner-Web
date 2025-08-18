# Audio Cleaner Streaming Service Test Suite

## Overview
This test suite validates the core functionality of the refactored streaming service without requiring external dependencies like Azure SDKs, Docker containers, or live services.

## What Gets Tested

### 🔐 Authentication & Security
- HMAC token verification with timing-attack resistance
- Base64url encoding/decoding utilities
- Session ID validation patterns
- Token extraction from headers
- Client IP detection and origin validation
- Connection limiting and rate limiting setup

### 📊 Session Management  
- SessionState initialization and configuration
- Audio buffer management and language tracking
- Billing state tracking and client pause controls
- SessionManager operations and lifecycle management

### 💰 Billing Logic
- Credit rate calculations and deduction timing
- Incremental billing with proper timing
- Low credits detection and signaling
- Cosmos DB failure handling (mocked)

### 🌐 API Endpoints
- FastAPI application structure and middleware
- Health endpoint with rate limiting
- Stop endpoint with authentication and validation
- WebSocket routing setup
- CORS configuration and input validation

## What's NOT Tested
- Real Azure SDK calls (Cosmos DB, OpenAI, Service Bus)
- Actual WebSocket connections with audio data
- Audio processing pipeline (DFNet, ASR, translation)
- Docker container security and networking
- Production environment configurations

## Running Tests

### Quick Test (Standalone)
```bash
cd processor/streaming/tests
python test_runner.py
```

### Individual Test Modules
```bash
# Authentication tests
python -m pytest test_auth.py -v

# Security utilities tests  
python -m pytest test_security.py -v

# Session management tests
python -m pytest test_session.py -v

# Billing logic tests
python -m pytest test_billing.py -v

# API integration tests
python -m pytest test_api.py -v
```

### All Tests with Coverage
```bash
python -m pytest tests/ -v --cov=../ --cov-report=term-missing
```

## Test Dependencies
The test suite automatically installs required dependencies:
- `pytest>=7.0.0` - Test framework
- `pytest-asyncio>=0.21.0` - Async test support
- `numpy>=1.20.0` - For mocking audio data
- `fastapi>=0.100.0` - For API tests
- `httpx>=0.24.0` - For HTTP client testing

## Mock Strategy
All external dependencies are mocked:
- **Environment variables**: Safe defaults provided
- **Azure Cosmos DB**: In-memory dictionaries
- **Azure OpenAI**: Simulated responses
- **Azure Service Bus**: Mock message sending
- **Audio processing**: Numpy arrays with fake data

## Security Validation
The tests specifically validate security measures:
- ✅ HMAC signatures properly verified
- ✅ Invalid tokens rejected
- ✅ Session IDs validated against patterns  
- ✅ Rate limiting configuration correct
- ✅ Input sanitization working
- ✅ Connection limits enforced
- ✅ CORS headers properly set

## Continuous Integration
This test suite is designed for CI/CD pipelines:
- No external dependencies or secrets required
- Fast execution (< 30 seconds)
- Clear pass/fail indicators
- Detailed error reporting
- Safe to run in isolated environments

## Architecture Validation
These tests confirm the refactoring achieved its goals:
- ✅ Modular design with single responsibility
- ✅ Clean separation of concerns
- ✅ Proper security middleware integration
- ✅ Authentication working across all endpoints
- ✅ Session management isolated and testable
- ✅ Billing logic independent of external calls
