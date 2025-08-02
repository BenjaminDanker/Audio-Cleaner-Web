#!/usr/bin/env pwsh

# Frontend Environment Setup Script
# This script fetches your Azure IDs dynamically and sets up the frontend .env file

param(
    [switch]$Help
)

if ($Help) {
    Write-Host @"
Frontend Environment Setup Script

This script automatically configures your frontend .env file with your Azure credentials.

Usage:
  .\setup-frontend-env.ps1

Prerequisites:
  - Azure CLI must be installed and you must be logged in (az login)
  - You must have access to your Azure subscription

The script will:
  1. Fetch your Azure tenant ID and user info from Azure CLI
  2. Update the frontend/.env file with the correct values
  3. Keep VITE_API_BASE_URL and other settings unchanged

"@
    exit 0
}

$ErrorActionPreference = "Stop"

# Check if Azure CLI is available
try {
    $null = az account show 2>$null
} catch {
    Write-Error "Azure CLI is not available or you're not logged in. Please run 'az login' first."
    exit 1
}

# Get the script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$frontendDir = Join-Path $scriptDir ".." "frontend"
$envFile = Join-Path $frontendDir ".env"

Write-Host "Setting up frontend environment..." -ForegroundColor Green

# Fetch Azure account information
Write-Host "Fetching Azure account information..." -ForegroundColor Yellow
try {
    $account = az account show --output json | ConvertFrom-Json
    $tenantId = $account.tenantId
    $clientId = $account.user.name  # This might be the user principal name or client ID
    
    Write-Host "  Tenant ID: $tenantId" -ForegroundColor Cyan
    Write-Host "  Client ID: $clientId" -ForegroundColor Cyan
} catch {
    Write-Error "Failed to get Azure account information. Please ensure you're logged in with 'az login'."
    exit 1
}

# Read existing .env file to preserve other settings
if (Test-Path $envFile) {
    $envContent = Get-Content $envFile
} else {
    Write-Error "Frontend .env file not found at: $envFile"
    exit 1
}

# Update the .env file
$newContent = @()
foreach ($line in $envContent) {
    if ($line -match "^VITE_AZURE_CLIENT_ID=") {
        $newContent += "VITE_AZURE_CLIENT_ID=$clientId"
    } elseif ($line -match "^VITE_AZURE_TENANT_ID=") {
        $newContent += "VITE_AZURE_TENANT_ID=$tenantId"
    } else {
        $newContent += $line
    }
}

# Write the updated content back to the file
$newContent | Out-File -FilePath $envFile -Encoding UTF8

Write-Host "Frontend environment configured successfully!" -ForegroundColor Green
Write-Host "Updated file: $envFile" -ForegroundColor Cyan
Write-Host ""
Write-Host "Note: If you're using a service principal for production, you may need to manually update the VITE_AZURE_CLIENT_ID" -ForegroundColor Yellow
