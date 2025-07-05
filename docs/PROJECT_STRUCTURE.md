# Audio Cleaner Pro - Project Structure

This document describes the structure of the Audio Cleaner Pro project after restoration.

## Project Overview
Audio Cleaner Pro is a web application that uses AI to clean audio tracks from video files. It consists of a React frontend, Azure Functions API backend, and Azure infrastructure.

## Directory Structure

```
d:\Coding\Python\Audio-Cleaner-Web\
├── api/                           # Azure Functions API
│   ├── auth/                      # Authentication function
│   ├── get-subscription/          # Subscription management function
│   ├── enqueue-job/              # Job queue function
│   ├── job-status/               # Job status function
│   ├── package.json              # API dependencies
│   ├── host.json                 # Azure Functions configuration
│   └── local.settings.json       # Local development settings
│
├── frontend/                      # React frontend application
│   ├── src/                      # Source code
│   │   ├── components/           # React components
│   │   │   ├── AuthContext.jsx   # Authentication context
│   │   │   ├── Login.jsx         # Login component
│   │   │   ├── Dashboard.jsx     # Main dashboard
│   │   │   ├── Navigation.jsx    # Navigation component
│   │   │   ├── VideoUpload.jsx   # Video upload component
│   │   │   ├── JobStatus.jsx     # Job status component
│   │   │   └── SubscriptionInfo.jsx # Subscription info
│   │   ├── App.jsx               # Main App component
│   │   ├── main.jsx              # Application entry point
│   │   ├── index.css             # Global styles
│   │   └── App.css               # App-specific styles
│   ├── index.html                # HTML template
│   ├── vite.config.js            # Vite configuration
│   └── package.json              # Frontend dependencies
│
├── infra/                         # Infrastructure as Code (Bicep)
│   ├── core/                     # Core infrastructure modules
│   │   ├── monitor/              # Monitoring resources
│   │   ├── storage/              # Storage resources
│   │   ├── security/             # Security resources
│   │   └── host/                 # Hosting resources
│   ├── app/                      # Application-specific resources
│   │   ├── api.bicep             # API container app
│   │   └── web.bicep             # Web container app
│   ├── main.bicep                # Main infrastructure template
│   ├── main.parameters.json      # Deployment parameters
│   └── abbreviations.json        # Azure resource abbreviations
│
├── scripts/                       # Development scripts
│   ├── start-dev.ps1             # Start development environment
│   ├── stop-dev.ps1              # Stop development environment
│   └── health-check.ps1          # Health check script
│
├── .github/                       # GitHub Actions workflows
│   └── workflows/
│       ├── ci.yml                # Continuous Integration
│       └── deploy.yml            # Deployment workflow
│
├── azure.yaml                     # Azure Developer CLI configuration
├── .env.local                     # Local environment variables
├── .env.local.example            # Environment variables template
├── .gitignore                    # Git ignore rules
└── README.md                     # Project documentation
```

## Technology Stack

### Frontend
- **React 18** - UI framework
- **Vite** - Build tool and dev server
- **React Router** - Client-side routing
- **Axios** - HTTP client
- **Lucide React** - Icon library

### Backend
- **Azure Functions** - Serverless API
- **Node.js 20** - Runtime environment
- **JavaScript/CommonJS** - Programming language

### Infrastructure
- **Azure Container Apps** - Application hosting
- **Azure Container Registry** - Container image storage
- **Azure Storage Account** - File storage
- **Azure Key Vault** - Secrets management
- **Azure Monitor** - Logging and monitoring
- **Bicep** - Infrastructure as Code

### Development Tools
- **PowerShell** - Development scripts
- **GitHub Actions** - CI/CD pipelines
- **Azure Developer CLI (azd)** - Deployment tool
- **Docker** - Containerization (optional for local dev)

## Key Features

### Implemented
- ✅ User authentication system
- ✅ Video file upload interface
- ✅ Job queue management
- ✅ Subscription management
- ✅ Responsive web design
- ✅ Azure Functions API endpoints
- ✅ Infrastructure as Code with Bicep
- ✅ CI/CD pipelines with GitHub Actions
- ✅ Local development environment

### Development Environment

#### Prerequisites
- Node.js 20.x LTS
- npm
- Azure Functions Core Tools v4
- PowerShell (for scripts)
- Docker (optional)

#### Quick Start
1. Install dependencies:
   ```powershell
   cd api && npm install
   cd ../frontend && npm install
   ```

2. Start development environment:
   ```powershell
   scripts\start-dev.ps1
   ```

3. Access the application:
   - Frontend: http://localhost:5173
   - API: http://localhost:7071

4. Stop development environment:
   ```powershell
   scripts\stop-dev.ps1
   ```

#### Health Check
Run the health check script to verify everything is working:
```powershell
scripts\health-check.ps1
```

## API Endpoints

- `GET/POST /api/auth` - Authentication
- `GET /api/get-subscription` - Get user subscription details
- `POST /api/enqueue-job` - Queue a new audio cleaning job
- `GET /api/job-status` - Get job status by ID

## Deployment

The project uses Azure Developer CLI (azd) for deployment:

```bash
# Login to Azure
azd auth login

# Provision infrastructure and deploy
azd up
```

Alternatively, use GitHub Actions for automated deployment when pushing to the main branch.

## Environment Variables

See `.env.local.example` for required environment variables. Copy to `.env.local` and configure for local development.

## Troubleshooting

1. **API not starting**: Check Node.js version (should be 20.x)
2. **Frontend build errors**: Ensure all dependencies are installed
3. **Port conflicts**: Stop other services using ports 7071 or 5173
4. **Azure Functions errors**: Check local.settings.json configuration

For more detailed troubleshooting, run `scripts\health-check.ps1`.

## Next Steps

1. Implement actual audio processing logic
2. Add Azure Storage integration for file uploads
3. Configure authentication providers
4. Add monitoring and alerting
5. Implement automated testing
6. Add error handling and validation
7. Configure production secrets management
