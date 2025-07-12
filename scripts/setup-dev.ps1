# Audio Cleaner Pro - Local Development Setup
param(
    [switch]$CloudBackend = $false,
    [switch]$LocalAPI = $false,
    [switch]$LocalProcessor = $false,
    [string]$EnvName = "dev-$env:USERNAME"
)

Write-Host "🛠️ Audio Cleaner Pro - Local Development Setup" -ForegroundColor Green

# Show development options
if (-not $CloudBackend -and -not $LocalAPI -and -not $LocalProcessor) {
    Write-Host "`n📋 Development Options:" -ForegroundColor Blue
    Write-Host "1. Frontend + Cloud Backend (Recommended)" -ForegroundColor Cyan
    Write-Host "   ./setup-dev.ps1 -CloudBackend" -ForegroundColor White
    Write-Host "   ✅ Fast iteration, real Azure services" -ForegroundColor Green
    Write-Host ""
    Write-Host "2. Frontend + Local API + Cloud Services" -ForegroundColor Cyan
    Write-Host "   ./setup-dev.ps1 -LocalAPI" -ForegroundColor White
    Write-Host "   ✅ API debugging, still uses real storage/database" -ForegroundColor Green
    Write-Host ""
    Write-Host "3. Fully Local (Frontend + API + Mock Processor)" -ForegroundColor Cyan
    Write-Host "   ./setup-dev.ps1 -LocalProcessor" -ForegroundColor White
    Write-Host "   ✅ Complete offline development" -ForegroundColor Green
    Write-Host ""
    Write-Host "Choose an option above, or press Ctrl+C to exit" -ForegroundColor Yellow
    exit 0
}

# Check prerequisites
Write-Host "`n📋 Checking prerequisites..." -ForegroundColor Blue
$requiredTools = @("node", "npm")

if ($CloudBackend -or $LocalAPI) {
    $requiredTools += @("az", "azd")
}

if ($LocalAPI) {
    $requiredTools += @("func")
}

if ($LocalProcessor) {
    $requiredTools += @("python")
}

foreach ($tool in $requiredTools) {
    if (!(Get-Command $tool -ErrorAction SilentlyContinue)) {
        Write-Error "❌ $tool is not installed or not in PATH"
        if ($tool -eq "func") {
            Write-Host "Install with: npm install -g azure-functions-core-tools@4" -ForegroundColor Yellow
        }
        exit 1
    }
    Write-Host "✅ $tool found" -ForegroundColor Green
}

# Setup Frontend (always local)
Write-Host "`n🎨 Setting up Frontend..." -ForegroundColor Blue
Push-Location frontend

if (!(Test-Path "node_modules")) {
    Write-Host "Installing frontend dependencies..." -ForegroundColor Yellow
    npm install
}

if ($CloudBackend) {
    Write-Host "Configuring frontend for cloud backend..." -ForegroundColor Yellow
    
    # Check if Azure environment exists
    try {
        azd env select $EnvName 2>$null
        $apiUrl = azd env get-value AZURE_API_URL
        if (-not $apiUrl) {
            Write-Host "⚠️ Environment exists but not deployed. Deploying..." -ForegroundColor Yellow
            azd up
            $apiUrl = azd env get-value AZURE_API_URL
        } else {
            Write-Host "✅ Using existing Azure environment: $EnvName" -ForegroundColor Green
        }
    } catch {
        Write-Host "⚠️ Creating new Azure environment..." -ForegroundColor Yellow
        azd env new $EnvName
        azd env select $EnvName
        Write-Host "Deploying backend to Azure (this takes 5-10 minutes)..." -ForegroundColor Yellow
        azd up
        $apiUrl = azd env get-value AZURE_API_URL
    }
    
    @"
# Cloud Backend Configuration
VITE_API_BASE_URL=$apiUrl
VITE_AZURE_AD_CLIENT_ID=$(azd env get-value AZURE_AD_CLIENT_ID)
VITE_AZURE_AD_TENANT_ID=$(azd env get-value AZURE_AD_TENANT_ID)
"@ | Out-File -FilePath ".env.local" -Encoding UTF8

    Write-Host "✅ Frontend configured for cloud backend" -ForegroundColor Green
} else {
    # Local backend configuration
    @"
# Local Backend Configuration  
VITE_API_BASE_URL=http://localhost:7071
VITE_AZURE_AD_CLIENT_ID=mock-client-id
VITE_AZURE_AD_TENANT_ID=mock-tenant-id
"@ | Out-File -FilePath ".env.local" -Encoding UTF8

    Write-Host "✅ Frontend configured for local backend" -ForegroundColor Green
}

Pop-Location

# Setup Local API if requested
if ($LocalAPI -or $LocalProcessor) {
    Write-Host "`n🔧 Setting up Local API..." -ForegroundColor Blue
    Push-Location api
    
    if (!(Test-Path "node_modules")) {
        Write-Host "Installing API dependencies..." -ForegroundColor Yellow
        npm install
    }
    
    if ($LocalAPI -and -not $LocalProcessor) {
        # Local API with cloud services
        Write-Host "Configuring API for cloud services..." -ForegroundColor Yellow
        
        $localSettings = @{
            IsEncrypted = $false
            Values = @{
                AzureWebJobsStorage = "UseDevelopmentStorage=true"
                FUNCTIONS_WORKER_RUNTIME = "node"
                AZURE_STORAGE_CONNECTION_STRING = (azd env get-value AZURE_STORAGE_CONNECTION_STRING -Environment $EnvName)
                COSMOS_CONNECTION_STRING = (azd env get-value COSMOS_CONNECTION_STRING -Environment $EnvName)
                AZURE_SERVICE_BUS_CONNECTION_STRING = (azd env get-value AZURE_SERVICE_BUS_CONNECTION_STRING -Environment $EnvName)
            }
        }
    } else {
        # Fully local development
        Write-Host "Configuring API for local development..." -ForegroundColor Yellow
        
        $localSettings = @{
            IsEncrypted = $false
            Values = @{
                AzureWebJobsStorage = "UseDevelopmentStorage=true"
                FUNCTIONS_WORKER_RUNTIME = "node"
                AZURE_STORAGE_CONNECTION_STRING = "UseDevelopmentStorage=true"
                COSMOS_CONNECTION_STRING = "AccountEndpoint=https://localhost:8081/;AccountKey=C2y6yDjf5/R+ob0N8A7Cgv30VRDJIWEHLM+4QDU5DE2nQ9nDuVTqobD4b8mGGyPMbIZnqyMsEcaGQy67XIw/Jw=="
                AZURE_SERVICE_BUS_CONNECTION_STRING = "mock://localhost"
                LOCAL_DEVELOPMENT = "true"
            }
        }
    }
    
    $localSettings | ConvertTo-Json -Depth 3 | Out-File -FilePath "local.settings.json" -Encoding UTF8
    Write-Host "✅ API configuration complete" -ForegroundColor Green
    Pop-Location
}

# Setup Local Processor if requested
if ($LocalProcessor) {
    Write-Host "`n🤖 Setting up Local Processor..." -ForegroundColor Blue
    
    # Create simple local processor script
    $localProcessorScript = @"
#!/usr/bin/env python3
"""
Local Development Processor
Simulates the AI processing for local development
"""

import time
import json
import os
import shutil
from pathlib import Path

def simulate_processing(input_file, output_file):
    """Simulate audio processing by copying file with delay"""
    print(f"🎵 Processing {input_file}...")
    
    # Simulate processing time
    time.sleep(3)
    
    # Copy input to output (in real app, this would be AI processing)
    shutil.copy2(input_file, output_file)
    
    print(f"✅ Processing complete: {output_file}")
    return True

def watch_for_jobs():
    """Watch for job files and process them"""
    jobs_dir = Path("api/temp/jobs")
    uploads_dir = Path("api/temp/uploads") 
    processed_dir = Path("api/temp/processed")
    
    # Create directories if they don't exist
    jobs_dir.mkdir(parents=True, exist_ok=True)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print("👀 Watching for jobs in api/temp/jobs/")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            # Look for job files
            for job_file in jobs_dir.glob("*.json"):
                try:
                    with open(job_file, 'r') as f:
                        job = json.load(f)
                    
                    input_path = uploads_dir / job['filename']
                    output_path = processed_dir / f"processed_{job['filename']}"
                    
                    if input_path.exists():
                        # Process the file
                        simulate_processing(input_path, output_path)
                        
                        # Update job status
                        job['status'] = 'completed'
                        job['output_file'] = str(output_path)
                        
                        with open(job_file, 'w') as f:
                            json.dump(job, f, indent=2)
                        
                        print(f"📝 Updated job status: {job_file}")
                    
                except Exception as e:
                    print(f"❌ Error processing {job_file}: {e}")
            
            time.sleep(2)  # Check every 2 seconds
            
    except KeyboardInterrupt:
        print("\n👋 Local processor stopped")

if __name__ == "__main__":
    watch_for_jobs()
"@
    
    $localProcessorScript | Out-File -FilePath "local-processor.py" -Encoding UTF8
    Write-Host "✅ Local processor created" -ForegroundColor Green
}

# Create start scripts
Write-Host "`n📜 Creating start scripts..." -ForegroundColor Blue

if ($CloudBackend) {
    $startScript = @"
#!/usr/bin/env pwsh
Write-Host "🚀 Starting Audio Cleaner Pro (Frontend + Cloud Backend)" -ForegroundColor Green
Write-Host "Frontend: http://localhost:5173" -ForegroundColor Cyan
Write-Host "Backend: Azure Cloud" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

Push-Location frontend
npm run dev
"@
} elseif ($LocalAPI -and -not $LocalProcessor) {
    $startScript = @"
#!/usr/bin/env pwsh
Write-Host "🚀 Starting Audio Cleaner Pro (Local Frontend + API, Cloud Services)" -ForegroundColor Green
Write-Host "Frontend: http://localhost:5173" -ForegroundColor Cyan  
Write-Host "API: http://localhost:7071" -ForegroundColor Cyan
Write-Host "Services: Azure Cloud" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop all services" -ForegroundColor Yellow
Write-Host ""

# Start API in background
Start-Job -ScriptBlock {
    Set-Location api
    func start
} -Name "API"

# Start frontend (blocking)
Push-Location frontend
npm run dev

# Cleanup
Write-Host "Stopping background services..." -ForegroundColor Yellow
Get-Job -Name "API" | Stop-Job | Remove-Job
"@
} else {
    $startScript = @"
#!/usr/bin/env pwsh
Write-Host "🚀 Starting Audio Cleaner Pro (Fully Local)" -ForegroundColor Green
Write-Host "Frontend: http://localhost:5173" -ForegroundColor Cyan
Write-Host "API: http://localhost:7071" -ForegroundColor Cyan  
Write-Host "Processor: Local Python simulation" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop all services" -ForegroundColor Yellow
Write-Host ""

# Start API in background
Start-Job -ScriptBlock {
    Set-Location api
    func start
} -Name "API"

# Start processor in background  
Start-Job -ScriptBlock {
    python local-processor.py
} -Name "Processor"

# Start frontend (blocking)
Push-Location frontend
npm run dev

# Cleanup
Write-Host "Stopping background services..." -ForegroundColor Yellow
Get-Job -Name "API" | Stop-Job | Remove-Job
Get-Job -Name "Processor" | Stop-Job | Remove-Job  
"@
}

$startScript | Out-File -FilePath "start-dev.ps1" -Encoding UTF8

Write-Host "`n🎉 Development Environment Ready!" -ForegroundColor Green

if ($CloudBackend) {
    Write-Host "`n🌐 Cloud Backend Mode:" -ForegroundColor Blue
    Write-Host "✅ Frontend runs locally for fast iteration" -ForegroundColor Green
    Write-Host "✅ Backend uses real Azure services" -ForegroundColor Green  
    Write-Host "✅ No local service dependencies" -ForegroundColor Green
    Write-Host "`nTo start: ./start-dev.ps1" -ForegroundColor Cyan
} elseif ($LocalAPI) {
    Write-Host "`n🔧 Hybrid Mode:" -ForegroundColor Blue
    Write-Host "✅ Frontend and API run locally" -ForegroundColor Green
    Write-Host "✅ Storage/Database use Azure cloud" -ForegroundColor Green
    Write-Host "✅ Good for API debugging" -ForegroundColor Green
    Write-Host "`nTo start: ./start-dev.ps1" -ForegroundColor Cyan
} else {
    Write-Host "`n💻 Fully Local Mode:" -ForegroundColor Blue 
    Write-Host "✅ Everything runs locally" -ForegroundColor Green
    Write-Host "✅ Mock processor simulates AI" -ForegroundColor Green
    Write-Host "✅ Complete offline development" -ForegroundColor Green
    Write-Host "`nTo start: ./start-dev.ps1" -ForegroundColor Cyan
}

Write-Host "`n🛠️ Development Commands:" -ForegroundColor Blue
Write-Host "  ./start-dev.ps1           # Start development environment" -ForegroundColor White
Write-Host "  ./stop-dev.ps1            # Stop all services" -ForegroundColor White
if ($CloudBackend -or $LocalAPI) {
    Write-Host "  azd deploy frontend      # Deploy frontend changes to cloud" -ForegroundColor White
    Write-Host "  azd deploy api           # Deploy API changes to cloud" -ForegroundColor White
}

# Create stop script
$stopScript = @"
#!/usr/bin/env pwsh
Write-Host "🛑 Stopping Audio Cleaner Pro development services..." -ForegroundColor Yellow

# Stop background jobs
Get-Job | Stop-Job | Remove-Job

# Kill any running processes on development ports
try {
    Get-Process | Where-Object {`$_.ProcessName -like "*node*" -or `$_.ProcessName -like "*func*" -or `$_.ProcessName -like "*python*"} | Where-Object {`$_.MainWindowTitle -like "*5173*" -or `$_.MainWindowTitle -like "*7071*"} | Stop-Process -Force
} catch {
    # Ignore errors
}

Write-Host "✅ Development services stopped" -ForegroundColor Green
"@

$stopScript | Out-File -FilePath "stop-dev.ps1" -Encoding UTF8

Write-Host "`n💡 Next Steps:" -ForegroundColor Blue
Write-Host "1. Run: ./start-dev.ps1" -ForegroundColor White
Write-Host "2. Open: http://localhost:5173" -ForegroundColor White
Write-Host "3. Start coding! 🎨" -ForegroundColor White
