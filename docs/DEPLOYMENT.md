<!-- markdownlint-disable MD031 MD032 MD040 MD022 MD036 MD058 MD026 MD009 MD024-->
# Deployment Guide

## Overview

Audio Cleaner Pro uses Azure Developer CLI (azd) for streamlined deployment with Infrastructure as Code (Bicep templates) and automated CI/CD pipelines.

## Prerequisites

### Required Tools
- [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli) (v2.50+)
- [Azure Developer CLI](https://learn.microsoft.com/en-us/azure/developer/azure-developer-cli/install-azd) (v1.5+)
- [Git](https://git-scm.com/downloads)
- [Node.js](https://nodejs.org/) (v18 LTS)

### Azure Requirements
- Azure subscription with **Contributor** or **Owner** access
- Subscription must support:
  - Azure Static Web Apps
  - Azure Functions Premium
  - Azure Container Apps
  - Azure Cosmos DB
  - Azure Storage (General Purpose v2)

### Optional Tools
- [Visual Studio Code](https://code.visualstudio.com/) with Azure extensions
- [Docker](https://docker.com/) for local container testing

## Quick Deployment

### 1. Initial Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd Audio-Cleaner-Web

# Login to Azure
azd auth login

# Initialize the environment
azd init
```

When prompted:
- **Environment name**: Choose a unique name (e.g., `audio-cleaner-prod`)
- **Azure location**: Select your preferred region (e.g., `eastus`)
- **Subscription**: Select your target subscription

### 2. Deploy Everything

```bash
# Deploy infrastructure and applications
azd up
```

This single command will:
- ✅ Provision all Azure resources using Bicep templates
- ✅ Build and deploy the React frontend
- ✅ Deploy Azure Functions API
- ✅ Build and deploy the Python AI processor container
- ✅ Configure networking, security, and monitoring

### 3. Post-Deployment Verification

```bash
# Check deployment status
azd show

# View application logs
azd logs

# Get environment details
azd env get-values
```

## Environment Configuration

### Development Environment

```bash
# Create development environment
azd env new dev

# Set development-specific settings
azd env set AZURE_LOCATION "eastus"
azd env set AZURE_RESOURCE_GROUP_NAME "rg-audio-cleaner-dev"

# Deploy development environment
azd up
```

### Production Environment

```bash
# Create production environment
azd env new prod

# Set production-specific settings
azd env set AZURE_LOCATION "eastus"
azd env set AZURE_RESOURCE_GROUP_NAME "rg-audio-cleaner-prod"

# Deploy production environment
azd up
```

## CI/CD Setup

### GitHub Actions Integration

#### 1. Prepare GitHub Repository

```bash
# Push your code to GitHub
git remote add origin https://github.com/your-username/Audio-Cleaner-Web
git push -u origin main
```

#### 2. Configure Azure Service Principal

```bash
# Create service principal for GitHub Actions
azd pipeline config
```

This will:
- Create an Azure Service Principal
- Configure required permissions
- Add secrets to your GitHub repository

#### 3. Required GitHub Secrets

The following secrets are automatically configured by `azd pipeline config`:

| Secret Name | Description |
|-------------|-------------|
| `AZURE_CLIENT_ID` | Service Principal Client ID |
| `AZURE_TENANT_ID` | Azure AD Tenant ID |
| `AZURE_SUBSCRIPTION_ID` | Target Azure Subscription |
| `AZURE_CREDENTIALS` | Service Principal credentials |

#### 4. Environment-Specific Secrets

For multiple environments, add these to your GitHub repository secrets:

**Development Environment:**
```
AZURE_ENV_NAME_DEV=audio-cleaner-dev
AZURE_LOCATION_DEV=eastus
```

**Production Environment:**
```
AZURE_ENV_NAME_PROD=audio-cleaner-prod
AZURE_LOCATION_PROD=eastus
```

### Workflow Configuration

The repository includes pre-configured GitHub Actions workflows:

#### `.github/workflows/azure-dev.yml`
- **Trigger**: Push to `main` branch
- **Purpose**: Deploy to production environment
- **Steps**: Build, test, and deploy all components

#### `.github/workflows/azure-dev-pr.yml`
- **Trigger**: Pull requests to `main` branch
- **Purpose**: Validate changes in development environment
- **Steps**: Build, test, and temporary deployment for testing

## Manual Deployment Steps

### Infrastructure Only

```bash
# Deploy only infrastructure resources
azd provision
```

### Application Code Only

```bash
# Deploy only application code (after infrastructure exists)
azd deploy
```

### Individual Services

```bash
# Deploy only the frontend
azd deploy frontend

# Deploy only the API
azd deploy api

# Deploy only the processor
azd deploy processor
```

## Configuration Management

### Environment Variables

#### Automatic Configuration
Most environment variables are automatically set during deployment:

- `AZURE_STORAGE_CONNECTION_STRING`
- `AZURE_COSMOS_CONNECTION_STRING`
- `AZURE_SERVICE_BUS_CONNECTION_STRING`
- `AZURE_KEY_VAULT_NAME`

#### Manual Configuration
Some variables require manual setup:

```bash
# Stripe configuration (after creating Stripe account)
azd env set STRIPE_SECRET_KEY "sk_live_..."
azd env set STRIPE_WEBHOOK_SECRET "whsec_..."

# Optional: Custom domain
azd env set CUSTOM_DOMAIN "app.yourdomain.com"

# Redeploy to apply changes
azd up
```

#### Azure Key Vault Integration

Sensitive configuration is automatically stored in Azure Key Vault:

```bash
# View secrets in Key Vault
az keyvault secret list --vault-name $(azd env get-value AZURE_KEY_VAULT_NAME)

# Add custom secret
az keyvault secret set \
  --vault-name $(azd env get-value AZURE_KEY_VAULT_NAME) \
  --name "CustomSecret" \
  --value "your-secret-value"
```

### Resource Configuration

#### Scaling Configuration

Edit `infra/main.parameters.json` to adjust scaling:

```json
{
  "parameters": {
    "containerAppMinReplicas": {
      "value": 0
    },
    "containerAppMaxReplicas": {
      "value": 10
    },
    "functionAppPlanSku": {
      "value": "EP1"
    }
  }
}
```

#### Regional Deployment

```bash
# Deploy to multiple regions
azd env set AZURE_LOCATION "westus2"
azd up --environment prod-west

azd env set AZURE_LOCATION "eastus"
azd up --environment prod-east
```

## Monitoring and Maintenance

### Health Checks

```bash
# Check application health
curl https://$(azd env get-value AZURE_FRONTEND_URL)/api/health

# View metrics
az monitor metrics list \
  --resource $(azd env get-value AZURE_CONTAINER_APP_ID) \
  --metric "Requests"
```

### Log Management

```bash
# Stream live logs
azd logs --follow

# Query specific logs
az monitor log-analytics query \
  --workspace $(azd env get-value AZURE_LOG_ANALYTICS_WORKSPACE_ID) \
  --analytics-query "ContainerAppConsoleLogs_CL | limit 100"
```

### Backup and Recovery

```bash
# Backup current environment configuration
azd env get-values > backup-$(date +%Y%m%d).env

# Export resource configuration
az group export \
  --resource-group $(azd env get-value AZURE_RESOURCE_GROUP) \
  --output json > backup-resources-$(date +%Y%m%d).json
```

## Troubleshooting

### Common Issues

#### Deployment Fails - Insufficient Permissions
```bash
# Check current permissions
az role assignment list --assignee $(az account show --query user.name -o tsv)

# Required roles: Contributor or Owner on subscription/resource group
```

#### Container App Build Fails
```bash
# Check container logs
az containerapp logs show \
  --name $(azd env get-value AZURE_PROCESSOR_APP_NAME) \
  --resource-group $(azd env get-value AZURE_RESOURCE_GROUP)
```

#### Function App Cold Start Issues
```bash
# Check if Premium plan is enabled
az functionapp show \
  --name $(azd env get-value AZURE_API_APP_NAME) \
  --resource-group $(azd env get-value AZURE_RESOURCE_GROUP) \
  --query "sku"
```

### Debugging Commands

```bash
# View all environment variables
azd env get-values

# Check resource deployment status
az deployment group list \
  --resource-group $(azd env get-value AZURE_RESOURCE_GROUP) \
  --query "[0].properties.provisioningState"

# Validate Bicep templates
az deployment group validate \
  --resource-group $(azd env get-value AZURE_RESOURCE_GROUP) \
  --template-file infra/main.bicep \
  --parameters @infra/main.parameters.json
```

### Performance Optimization

#### Cost Optimization
```bash
# Review resource costs
az consumption usage list \
  --start-date $(date -d "30 days ago" +%Y-%m-%d) \
  --end-date $(date +%Y-%m-%d)

# Scale down development environments
azd env select dev
azd env set CONTAINER_APP_MIN_REPLICAS 0
azd env set FUNCTION_APP_PLAN_SKU "Y1"  # Consumption plan
azd up
```

#### Performance Tuning
```bash
# Enable Application Insights profiling
az monitor app-insights component update \
  --app $(azd env get-value AZURE_APPLICATION_INSIGHTS_NAME) \
  --resource-group $(azd env get-value AZURE_RESOURCE_GROUP) \
  --sampling-percentage 10
```

## Security Considerations

### Network Security
- All resources deployed with private endpoints where possible
- Network Security Groups configured for minimal required access
- Azure Front Door provides DDoS protection

### Access Control
- Managed Identity used for service-to-service authentication
- RBAC configured with principle of least privilege
- Azure Key Vault integration for secrets management

### Compliance
- All data encrypted at rest and in transit
- Audit logging enabled for all resources
- GDPR compliance features available

For more security details, see [Security Guide](SECURITY.md).
