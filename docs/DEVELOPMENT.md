<!-- markdownlint-disable MD031 MD032 MD040 MD022 MD036 MD058 MD026 MD009 MD024-->
# Developer Guide

## Overview

This guide covers development practices, code organization, and contribution guidelines for Audio Cleaner Pro.

## Getting Started

### Prerequisites

**Required Tools:**
- [Node.js](https://nodejs.org/) v18 LTS
- [Python](https://python.org/) 3.11+
- [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)
- [Azure Developer CLI](https://learn.microsoft.com/en-us/azure/developer/azure-developer-cli/install-azd)
- [Git](https://git-scm.com/)

**Optional Tools:**
- [Visual Studio Code](https://code.visualstudio.com/) with recommended extensions
- [Docker Desktop](https://www.docker.com/products/docker-desktop) for container testing

### Project Setup

```bash
# Clone repository
git clone <your-repo-url>
cd Audio-Cleaner-Web

# Install frontend dependencies
cd frontend
npm install

# Install API dependencies
cd ../api
npm install

# Set up Python environment
cd ../processor
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Project Structure

```
Audio-Cleaner-Web/
├── frontend/                 # React frontend application
│   ├── src/
│   │   ├── components/      # Reusable UI components
│   │   ├── pages/          # Route-based page components
│   │   ├── hooks/          # Custom React hooks
│   │   ├── services/       # API client services
│   │   └── utils/          # Utility functions
│   ├── public/             # Static assets
│   └── package.json
├── api/                     # Azure Functions API
│   ├── {function-name}/    # Individual function directories
│   │   ├── function.json   # Function configuration
│   │   └── index.js        # Function implementation
│   ├── shared/             # Shared utilities
│   └── package.json
├── processor/              # Python AI processing service
│   ├── src/                # Source code
│   │   ├── processor_app.py    # Flask web app
│   │   ├── processor_main.py   # Service Bus consumer
│   │   └── video_handler.py    # Video processing logic
│   ├── models/             # AI model files
│   └── requirements.txt
├── infra/                  # Infrastructure as Code
│   ├── main.bicep          # Main Bicep template
│   ├── main.parameters.json # Configuration parameters
│   └── app/                # Application-specific resources
└── docs/                   # Documentation
```

## Development Workflow

### Feature Development

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Develop and Test**
   - Make your changes
   - Test locally (see Testing section)
   - Commit regularly with clear messages

3. **Submit Pull Request**
   ```bash
   git push origin feature/your-feature-name
   # Create PR on GitHub
   ```

4. **Code Review Process**
   - Automated checks must pass
   - Peer review required
   - Address feedback
   - Merge after approval

### Coding Standards

#### JavaScript/TypeScript (Frontend & API)
- **Style Guide**: Prettier + ESLint configuration
- **Naming**: camelCase for variables/functions, PascalCase for components
- **Structure**: Barrel exports for clean imports
- **Comments**: JSDoc for public APIs

Example:
```javascript
/**
 * Uploads a file to Azure Blob Storage
 * @param {File} file - The file to upload
 * @param {string} containerName - Target container
 * @returns {Promise<string>} The uploaded file URL
 */
async function uploadFile(file, containerName) {
  // Implementation
}
```

#### Python (Processor)
- **Style Guide**: PEP 8 with Black formatter
- **Type Hints**: Required for all function signatures
- **Docstrings**: Google style docstrings
- **Imports**: Absolute imports preferred

Example:
```python
def process_audio(input_path: str, output_path: str) -> bool:
    """
    Process audio file to remove background noise.
    
    Args:
        input_path: Path to input audio file
        output_path: Path for processed output file
        
    Returns:
        True if processing successful, False otherwise
    """
    # Implementation
```

#### Infrastructure (Bicep)
- **Naming**: Kebab-case for resources
- **Parameters**: Use parameters for configurable values
- **Comments**: Explain complex resource configurations
- **Modularity**: Split into logical modules

### Testing Strategy

#### Frontend Testing
```bash
cd frontend

# Unit tests (Jest + React Testing Library)
npm test

# Component tests
npm run test:components

# E2E tests (Playwright)
npm run test:e2e
```

#### API Testing
```bash
cd api

# Unit tests
npm test

# Integration tests
npm run test:integration

# Load tests
npm run test:load
```

#### Processor Testing
```bash
cd processor

# Unit tests
python -m pytest tests/

# Integration tests
python -m pytest tests/integration/

# Model validation
python -m pytest tests/model/
```

### Local Development

#### Cloud Development (Recommended)

Deploy to Azure for development:
```bash
# Create development environment
azd env new dev

# Deploy to Azure
azd up
```

Benefits:
- Full Azure service integration
- Realistic performance testing
- No local service dependencies
- Easy sharing with team

#### API Development

For API-only development:
```bash
cd api

# Install Azure Functions Core Tools
npm install -g azure-functions-core-tools@4

# Start local development server
func start
```

Environment configuration in `local.settings.json`:
```json
{
  "IsEncrypted": false,
  "Values": {
    "FUNCTIONS_WORKER_RUNTIME": "node",
    "AZURE_STORAGE_CONNECTION_STRING": "your-dev-connection-string",
    "COSMOS_CONNECTION_STRING": "your-dev-cosmos-connection"
  }
}
```

#### Frontend Development

```bash
cd frontend

# Start development server
npm run dev

# Access at http://localhost:5173
```

Environment configuration in `.env.local`:
```bash
VITE_API_BASE_URL=https://your-dev-api.azurewebsites.net
VITE_AZURE_AD_CLIENT_ID=your-client-id
VITE_AZURE_AD_TENANT_ID=your-tenant-id
```

### Debugging

#### VS Code Configuration

Recommended VS Code extensions:
- Azure Functions
- Azure Account
- Azure Resources
- Azure Storage
- REST Client
- Thunder Client

Launch configuration (`.vscode/launch.json`):
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug Frontend",
      "type": "node",
      "request": "launch",
      "program": "${workspaceFolder}/frontend/node_modules/.bin/vite",
      "args": ["dev"],
      "cwd": "${workspaceFolder}/frontend"
    },
    {
      "name": "Debug API Function",
      "type": "node",
      "request": "attach",
      "port": 9229,
      "restart": true,
      "preLaunchTask": "func-start"
    }
  ]
}
```

#### Debugging Azure Functions

```bash
# Start with debugging enabled
func start --javascript-debug 9229

# In VS Code, attach debugger to port 9229
```

#### Debugging Python Processor

```bash
# Run with debugger
python -m debugpy --listen 5678 --wait-for-client src/processor_main.py

# In VS Code, connect to remote debugger
```

### Performance Optimization

#### Frontend Optimization
- **Code Splitting**: Route-based and component-based splitting
- **Lazy Loading**: Defer non-critical resources
- **Image Optimization**: WebP format with fallbacks
- **Bundle Analysis**: `npm run analyze` to check bundle size

#### API Optimization
- **Connection Pooling**: Reuse database connections
- **Caching**: Redis for frequently accessed data
- **Compression**: gzip compression for responses
- **Monitoring**: Application Insights for performance tracking

#### Processor Optimization
- **Async Processing**: Non-blocking I/O operations
- **Resource Management**: Proper cleanup of temporary files
- **Model Caching**: Cache loaded models between requests
- **Batch Processing**: Process multiple files efficiently

### Contributing Guidelines

#### Pull Request Process

1. **Branch Naming**
   - `feature/description` - New features
   - `bugfix/description` - Bug fixes
   - `hotfix/description` - Critical production fixes
   - `docs/description` - Documentation updates

2. **Commit Messages**
   ```
   type(scope): description
   
   feat(api): add file upload progress tracking
   fix(frontend): resolve upload button disabled state
   docs(readme): update deployment instructions
   ```

3. **PR Requirements**
   - [ ] All tests passing
   - [ ] Code coverage maintained
   - [ ] Documentation updated
   - [ ] No security vulnerabilities
   - [ ] Performance impact assessed

#### Code Review Checklist

**Functionality**
- [ ] Code works as intended
- [ ] Edge cases handled
- [ ] Error handling implemented
- [ ] Input validation present

**Quality**
- [ ] Code follows style guidelines
- [ ] No code duplication
- [ ] Functions are focused and testable
- [ ] Comments explain complex logic

**Security**
- [ ] No hardcoded secrets
- [ ] Input sanitization implemented
- [ ] Authentication/authorization correct
- [ ] SQL injection prevention

**Performance**
- [ ] No obvious performance issues
- [ ] Database queries optimized
- [ ] Appropriate caching used
- [ ] Resource cleanup implemented

### Environment Management

#### Development Environment
```bash
# Create isolated development environment
azd env new dev-yourname

# Deploy with minimal resources
azd env set CONTAINER_APP_MIN_REPLICAS 0
azd env set FUNCTION_APP_PLAN_SKU Y1  # Consumption plan
azd up
```

#### Feature Testing Environment
```bash
# Create temporary environment for feature testing
azd env new feature-123

# Deploy and test
azd up

# Clean up when done
azd down --force --purge
```

### Documentation Standards

#### Code Documentation
- **README files**: Each major component should have a README
- **API Documentation**: OpenAPI/Swagger specs for APIs
- **Inline Comments**: Explain business logic and complex algorithms
- **Architecture Decisions**: Document significant technical decisions

#### Documentation Updates
- Update relevant docs with code changes
- Include screenshots for UI changes
- Test code examples before committing
- Keep documentation current with implementation

For deployment and infrastructure details, see [Deployment Guide](DEPLOYMENT.md).
For security considerations, see [Security Guide](SECURITY.md).
