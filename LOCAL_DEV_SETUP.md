# Local Development Setup

This guide helps you set up the Audio Cleaner Pro project for local development.

## Prerequisites

1. **Node.js** (v18 or later)
2. **Azure Functions Core Tools** v4
3. **Python** 3.9+ (for the AI processor)
4. **Azure CLI** (optional, for cloud deployment)

## Quick Start

### 1. Install Dependencies

```powershell
# Frontend dependencies
cd frontend
npm install
cd ..

# API dependencies  
cd api
npm install
cd ..

# Python dependencies
pip install -r requirements.txt
```

### 2. Start Development Servers

Use the provided PowerShell script:

```powershell
.\scripts\start-dev.ps1
```

Or start services manually:

```powershell
# Terminal 1: Start Frontend (Vite)
cd frontend
npm run dev

# Terminal 2: Start Azure Functions
cd api
func start --port 7071

# Terminal 3: Start Python Processor (optional for full testing)
python processor_main.py
```

## Local Development Features

### API Functions
- **Local Development Mode**: All API functions detect when running locally and return mock data
- **No Azure Services Required**: Functions work without real Cosmos DB, Service Bus, or Stripe connections
- **Mock Authentication**: Simulates Azure Static Web Apps authentication

### Available Endpoints
- `http://localhost:7071/api/get-subscription` - Returns mock subscription data
- `http://localhost:7071/api/enqueue-job` - Simulates job queueing
- `http://localhost:7071/api/job-status` - Returns random job status
- `http://localhost:7071/api/create-checkout-session` - Returns mock Stripe session

### Frontend
- **Development Auth**: Automatically simulates logged-in user
- **API Integration**: Connects to local Azure Functions
- **Hot Reload**: Vite provides instant updates

## Configuration

### Frontend (.env.local)
The frontend uses environment variables for API endpoints:

```bash
VITE_API_BASE_URL=http://localhost:7071
```

### API (api/local.settings.json)
The API functions are pre-configured for local development with mock connection strings.

## Testing the Full Flow

1. **Start all services** using `.\scripts\start-dev.ps1`
2. **Open browser** to `http://localhost:5173`
3. **Upload a video** (any small video file)
4. **Check job status** in the dashboard
5. **View subscription** information

## Troubleshooting

### Common Issues

**"Failed to load resource: 500 Internal Server Error"**
- Ensure Azure Functions Core Tools is installed
- Check that `func start` is running in the `api` directory
- Verify `api/local.settings.json` exists

**"Auth check failed"**
- This is normal in local development
- The app automatically switches to development mode

**Frontend not connecting to API**
- Check that Functions are running on port 7071
- Verify CORS is enabled in `api/local.settings.json`

## Production Deployment

When ready to deploy:

1. Set up Azure resources: `azd up`
2. Configure GitHub secrets with your Azure resource names
3. Push to main branch to trigger deployment

## Environment Variables

For production deployment, ensure these secrets are set in GitHub:

- `AZURE_CLIENT_ID`
- `AZURE_ENV_NAME`
- `AZURE_LOCATION`
- `AZURE_PROCESSOR_APP_NAME`
- `AZURE_RESOURCE_GROUP`
- `AZURE_STATIC_WEB_APPS_API_TOKEN`
- `AZURE_SUBSCRIPTION_ID`
- `AZURE_TENANT_ID`
