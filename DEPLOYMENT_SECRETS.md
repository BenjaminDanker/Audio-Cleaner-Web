# GitHub Actions Deployment Configuration

This document describes how the GitHub Actions deployment workflow is configured to use your production environment secrets.

## GitHub Environment
- **Environment Name**: `production`
- **Required for**: Deployment to production Azure resources

## Required Secrets

The following secrets must be configured in your GitHub repository's `production` environment:

### Azure Authentication
| Secret Name | Description | Usage |
|-------------|-------------|-------|
| `AZURE_CLIENT_ID` | Azure Service Principal Client ID | Authentication for Azure CLI and azd |
| `AZURE_TENANT_ID` | Azure Active Directory Tenant ID | Authentication scope |
| `AZURE_SUBSCRIPTION_ID` | Target Azure Subscription ID | Deployment target |

### Azure Resources
| Secret Name | Description | Usage |
|-------------|-------------|-------|
| `AZURE_ENV_NAME` | Environment name for AZD | Resource naming and identification |
| `AZURE_LOCATION` | Azure region (e.g., eastus, westus2) | Deployment region |
| `AZURE_RESOURCE_GROUP` | Target resource group name | Resource organization |
| `AZURE_PROCESSOR_APP_NAME` | Processor application name | Audio processing service configuration |

### Additional Services
| Secret Name | Description | Usage |
|-------------|-------------|-------|
| `AZURE_STATIC_WEB_APPS_API_TOKEN` | Static Web Apps deployment token | Frontend deployment (if using SWA) |

## Workflow Triggers

### Automatic Deployment (Production)
- **Trigger**: Push to `main` branch
- **Environment**: `production`
- **Actions**: 
  1. Lint and test code
  2. Build applications
  3. Deploy to Azure using `azd`

### Pull Request Validation
- **Trigger**: Pull request to `main` branch
- **Environment**: `production` (read-only for validation)
- **Actions**: 
  1. Lint and test code
  2. Validate deployment configuration

### Manual Deployment
- **Trigger**: `workflow_dispatch` (manual trigger from GitHub UI)
- **Environment**: `production`
- **Actions**: Same as automatic deployment

## Deployment Process

1. **Authentication**
   ```yaml
   - uses: azure/login@v1
     with:
       client-id: ${{ secrets.AZURE_CLIENT_ID }}
       tenant-id: ${{ secrets.AZURE_TENANT_ID }}
       subscription-id: ${{ secrets.AZURE_SUBSCRIPTION_ID }}
   ```

2. **Environment Setup**
   ```bash
   azd env set AZURE_ENV_NAME "${{ secrets.AZURE_ENV_NAME }}"
   azd env set AZURE_LOCATION "${{ secrets.AZURE_LOCATION }}"
   azd env set AZURE_SUBSCRIPTION_ID "${{ secrets.AZURE_SUBSCRIPTION_ID }}"
   azd env set AZURE_RESOURCE_GROUP "${{ secrets.AZURE_RESOURCE_GROUP }}"
   azd env set AZURE_PROCESSOR_APP_NAME "${{ secrets.AZURE_PROCESSOR_APP_NAME }}"
   ```

3. **Deployment**
   ```bash
   azd provision --no-prompt
   azd deploy --no-prompt
   ```

## Service Principal Requirements

Your Azure Service Principal (identified by `AZURE_CLIENT_ID`) needs the following permissions:

### Required Azure RBAC Roles
- **Contributor** - On the target subscription or resource group
- **User Access Administrator** - For managing role assignments (if needed)

### Required Permissions
- Create and manage resource groups
- Deploy ARM/Bicep templates
- Manage Container Apps and Container Registry
- Manage Storage Accounts
- Manage Key Vault
- Manage Application Insights and Log Analytics

## Security Best Practices

### Secret Management
- ✅ All sensitive values stored as GitHub secrets
- ✅ Secrets scoped to `production` environment
- ✅ No secrets in code or configuration files
- ✅ Service Principal authentication (no passwords)

### Access Control
- ✅ Environment protection rules can be enabled
- ✅ Required reviewers can be configured
- ✅ Deployment branches restricted to `main`

## Troubleshooting

### Common Issues

1. **Authentication Failures**
   - Verify `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, and `AZURE_SUBSCRIPTION_ID` are correct
   - Ensure Service Principal has required permissions
   - Check if Service Principal is enabled and not expired

2. **Resource Group Issues**
   - Verify `AZURE_RESOURCE_GROUP` exists or can be created
   - Check Service Principal has Contributor access to the resource group
   - Ensure resource group is in the specified `AZURE_LOCATION`

3. **Deployment Failures**
   - Check Azure resource quotas in the target region
   - Verify `AZURE_ENV_NAME` follows naming conventions
   - Review Azure activity logs for detailed error messages

### Debugging Steps

1. **Check Secret Configuration**
   ```bash
   # In GitHub Actions, secrets are masked but you can verify they're set:
   echo "Secrets configured: $(if [ -n "${{ secrets.AZURE_CLIENT_ID }}" ]; then echo "✓"; else echo "✗"; fi)"
   ```

2. **Verify Azure Access**
   ```bash
   az account show
   az group list --query "[?name=='${{ secrets.AZURE_RESOURCE_GROUP }}']"
   ```

3. **Test AZD Configuration**
   ```bash
   azd env list
   azd env get-values
   ```

## Manual Setup Commands

If you need to set up the Service Principal manually:

```bash
# Create Service Principal
az ad sp create-for-rbac --name "audio-cleaner-github-actions" \
  --role Contributor \
  --scopes /subscriptions/{subscription-id} \
  --sdk-auth

# Get required values for secrets
az account show --query '{subscriptionId:id, tenantId:tenantId}'
```

## Static Web Apps Integration

If using Azure Static Web Apps (indicated by the `AZURE_STATIC_WEB_APPS_API_TOKEN` secret):

1. The token is automatically configured in the deployment process
2. Frontend builds are deployed directly to Static Web Apps
3. API integration is handled through the static web app configuration

This setup provides a secure, automated deployment pipeline that follows Azure and GitHub best practices.
