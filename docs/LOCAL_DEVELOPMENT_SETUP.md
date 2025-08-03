# Local Development Setup Guide

This guide explains how to set up local development for the Audio Cleaner Web application while using cloud Azure services.

## Overview

Your application has two development modes:

1. **Local Mode**: Uses Azurite (local storage emulator) and file-based simulation for all Azure services
2. **Cloud Mode**: Uses actual Azure services (Storage, Cosmos DB, Service Bus) while running locally

## Quick Start

### 1. Get Cloud Configuration

First, retrieve the connection strings from your deployed Azure infrastructure:

```powershell
# Run from the root directory
.\scripts\get-cloud-config.ps1
```

This script will:
- Extract connection strings from your Terraform state or Azure resources
- Create `api/local.settings.cloud.json` with cloud service connections
- Create `processor/.env.cloud` for Docker processor

### 2. Switch Development Mode

Use the switcher script to toggle between local and cloud development:

```powershell
# Switch to cloud development (uses Azure services)
.\scripts\switch-dev-mode.ps1 -Mode cloud

# Switch to local development (uses local emulators)
.\scripts\switch-dev-mode.ps1 -Mode local
```

### 3. Start Development Services

#### Cloud Development Mode
```powershell
# Start the processor with cloud connections
docker-compose -f docker-compose.dev.yml up processor

# In another terminal, start the frontend and API
swa start
```

#### Local Development Mode
```powershell
# Start local storage emulator (if not using Azurite)
# Then start the application
swa start
```

## File Structure

```
/
├── scripts/
│   ├── get-cloud-config.ps1     # Retrieves cloud connection strings
│   └── switch-dev-mode.ps1      # Switches between local/cloud modes
├── api/
│   ├── local.settings.json      # Active configuration (changes based on mode)
│   ├── local.settings.local.json # Local development configuration
│   └── local.settings.cloud.json # Cloud development configuration
├── processor/
│   └── .env.cloud               # Cloud environment variables for Docker
├── docker-compose.dev.yml       # Development Docker setup
└── ...
```

## Configuration Details

### Cloud Mode Configuration

When in cloud mode, your local development will:
- ✅ Use Azure Storage Blob for file operations
- ✅ Use Azure Cosmos DB for database operations
- ✅ Use Azure Service Bus for queue operations
- ✅ Run frontend and API locally with SWA CLI
- ✅ Run processor in Docker with cloud connections

### Local Mode Configuration

When in local mode, your local development will:
- 🏠 Use Azurite for storage emulation
- 🏠 Use file-based simulation for Cosmos DB
- 🏠 Use file-based simulation for Service Bus
- 🏠 Run everything locally

## Environment Variables

### Key Variables Used

- `AZURE_STORAGE_CONNECTION_STRING`: Storage account connection
- `COSMOS_CONNECTION_STRING`: Cosmos DB connection
- `AZURE_SERVICE_BUS_CONNECTION_STRING`: Service Bus connection


## Benefits

✅ **Minimal Code Changes**: No production code modifications required
✅ **Environment Isolation**: Clear separation between local and cloud development
✅ **Easy Switching**: One command to switch between modes
✅ **Real Cloud Testing**: Test against actual Azure services during development
✅ **Cost Effective**: Only pay for actual usage during development

## Troubleshooting

### Connection Issues
1. Ensure you're logged into Azure CLI: `az login`
2. Verify you have access to the resource group
3. Check that Terraform state is available

### Docker Issues
1. Ensure Docker is running
2. Check that the `.env.cloud` file was created
3. Verify processor Dockerfile builds successfully

### SWA CLI Issues
1. Ensure Node.js dependencies are installed: `npm install` in both `frontend/` and `api/`
2. Check that the correct `local.settings.json` is active
3. Verify CORS settings if needed

## Commands Reference

```powershell
# Setup cloud configuration
.\scripts\get-cloud-config.ps1

# Switch to cloud development
.\scripts\switch-dev-mode.ps1 -Mode cloud

# Switch to local development  
.\scripts\switch-dev-mode.ps1 -Mode local

# Start cloud development
docker-compose -f docker-compose.dev.yml up processor
swa start

# Start local development
swa start

# View current configuration
Get-Content api\local.settings.json | ConvertFrom-Json | Select-Object -ExpandProperty Values
```

## Security Notes

- Connection strings are marked as sensitive in scripts
- Cloud configuration files are excluded from git (add to .gitignore)
- Use environment-specific credentials appropriately
- Local development should use test/development Azure resources when possible
