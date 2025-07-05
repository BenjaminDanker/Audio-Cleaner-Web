# Azure Static Web Apps Secret Troubleshooting

## Understanding the "Context access might be invalid" Warning

The warning `Context access might be invalid: AZURE_STATIC_WEB_APPS_API_TOKEN` that you're seeing in VS Code is **NOT an error** - it's just a validation warning because VS Code cannot verify if the secret exists in your GitHub repository.

## How to Verify Your Secret is Correct

### 1. Check Secret in GitHub Repository
1. Go to your GitHub repository
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Look for `AZURE_STATIC_WEB_APPS_API_TOKEN` in the Repository secrets section
4. Verify it exists and was created recently

### 2. Where to Get the Correct Token

The `AZURE_STATIC_WEB_APPS_API_TOKEN` should come from:

1. **Azure Portal** → **Static Web Apps** → Your app → **Manage deployment token**
2. Or when you first create the Static Web App, Azure automatically adds it to GitHub

### 3. Common Issues and Solutions

#### Issue: Secret doesn't exist
**Solution:** Get the deployment token from Azure:
```bash
# Using Azure CLI
az staticwebapp secrets list --name <your-app-name> --resource-group <your-rg>
```

#### Issue: Wrong secret name
**Solution:** The exact name should be `AZURE_STATIC_WEB_APPS_API_TOKEN` (case-sensitive)

#### Issue: Token expired or invalid
**Solution:** Regenerate the token in Azure Portal:
1. Azure Portal → Static Web Apps → Your app
2. Click "Manage deployment token"
3. Copy the new token
4. Update the GitHub secret

### 4. Test the Secret

Create a simple test workflow to verify the secret:

```yaml
name: Test Secret
on:
  workflow_dispatch:
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Test Secret
        run: |
          if [ -n "${{ secrets.AZURE_STATIC_WEB_APPS_API_TOKEN }}" ]; then
            echo "✅ Secret is available"
            echo "Token length: ${#GITHUB_TOKEN}"
          else
            echo "❌ Secret is not available"
          fi
        env:
          GITHUB_TOKEN: ${{ secrets.AZURE_STATIC_WEB_APPS_API_TOKEN }}
```

### 5. Alternative: Use Azure Service Principal

If the deployment token continues to cause issues, you can use a Service Principal instead:

```yaml
- name: Build And Deploy
  uses: Azure/static-web-apps-deploy@v1
  with:
    azure_static_web_apps_api_token: ${{ secrets.AZURE_STATIC_WEB_APPS_API_TOKEN }}
    # Alternative authentication:
    # subscription-id: ${{ secrets.AZURE_SUBSCRIPTION_ID }}
    # resource-group: ${{ secrets.AZURE_RESOURCE_GROUP }}
    # client-id: ${{ secrets.AZURE_CLIENT_ID }}
    # tenant-id: ${{ secrets.AZURE_TENANT_ID }}
    # client-secret: ${{ secrets.AZURE_CLIENT_SECRET }}
```

## Ignore the VS Code Warning

The warning in VS Code is cosmetic. If your GitHub Actions workflow runs successfully, the secret is working correctly. You can safely ignore this warning.

## Quick Verification Steps

1. ✅ Secret exists in GitHub repo settings
2. ✅ Secret name is exactly `AZURE_STATIC_WEB_APPS_API_TOKEN`
3. ✅ Token is valid (not expired)
4. ✅ Workflow has run successfully at least once
5. ✅ You can ignore VS Code warnings about secret validation

If all steps pass, your setup is correct and the warning can be ignored.
