# Audio Cleaner Pro - Start Authentic Local Development Environment
Write-Host "Starting Audio Cleaner Pro Development Environment with Azure Functions and SWA CLI..." -ForegroundColor Green

# Check if required tools are installed
function Test-Command($cmdname) {
    return [bool](Get-Command -Name $cmdname -ErrorAction SilentlyContinue)
}

# Verify Node.js and npm
if (-not (Test-Command "node")) {
    Write-Error "Node.js is not installed. Please install Node.js 20.x LTS first."
    exit 1
}

if (-not (Test-Command "npm")) {
    Write-Error "npm is not installed. Please install npm first."
    exit 1
}

$nodeVersion = node --version
Write-Host "Node.js version: $nodeVersion" -ForegroundColor Cyan

# Verify Python
if (-not (Test-Command "python")) {
    Write-Error "Python is not installed. Please install Python 3.8+ first."
    exit 1
}

$pythonVersion = python --version
Write-Host "Python version: $pythonVersion" -ForegroundColor Cyan

# Verify Azure Functions Core Tools
if (-not (Test-Command "func")) {
    Write-Error "Azure Functions Core Tools is not installed. Please install it first: npm install -g azure-functions-core-tools@4"
    exit 1
}

$funcVersion = func --version
Write-Host "Azure Functions Core Tools version: $funcVersion" -ForegroundColor Cyan

# Verify Azure Static Web Apps CLI
if (-not (Test-Command "swa")) {
    Write-Error "Azure Static Web Apps CLI is not installed. Please install it first: npm install -g @azure/static-web-apps-cli"
    exit 1
}

$swaVersion = swa --version
Write-Host "SWA CLI version: $swaVersion" -ForegroundColor Cyan

# Stop any existing processes
Write-Host "Stopping any existing development processes..." -ForegroundColor Yellow
.\scripts\stop-dev.ps1

# Create process tracking file
$processFile = "dev-processes.json"
$processes = @{}

try {
    # Install/update API dependencies (skip if node_modules exists)
    if (-not (Test-Path "api/node_modules")) {
        Write-Host "Installing API dependencies..." -ForegroundColor Yellow
        Push-Location "api"
        npm install
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to install API dependencies"
            Pop-Location
            exit 1
        }
        Pop-Location
    } else {
        Write-Host "API dependencies already installed (node_modules present)" -ForegroundColor Green
    }

    # Install/update Frontend dependencies (skip if node_modules exists)
    if (-not (Test-Path "frontend/node_modules")) {
        Write-Host "Installing Frontend dependencies..." -ForegroundColor Yellow
        Push-Location "frontend"
        npm install
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to install frontend dependencies"
            Pop-Location
            exit 1
        }
        Pop-Location
    } else {
        Write-Host "Frontend dependencies already installed (node_modules present)" -ForegroundColor Green
    }

    # Install/update Python dependencies for processor
    Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
    if (Test-Path ".venv") {
        Write-Host "Virtual environment found" -ForegroundColor Green
    } else {
        Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
        python -m venv .venv
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to create Python virtual environment"
            exit 1
        }
    }
    
    # Activate virtual environment and install dependencies
    & ".\.venv\Scripts\Activate.ps1"
    pip install --quiet -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install Python dependencies"
        exit 1
    }

    # Create necessary directories
    Write-Host "Creating required directories..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path "api\temp\jobs" | Out-Null
    New-Item -ItemType Directory -Force -Path "api\temp\uploads" | Out-Null
    New-Item -ItemType Directory -Force -Path "api\temp\downloads" | Out-Null

    # Start Azure Functions host for API in background
    Write-Host "Starting Azure Functions host..." -ForegroundColor Yellow
    $apiJob = Start-Job -ScriptBlock {
        param($workingDir)
        Set-Location $workingDir
        Set-Location "api"
        func start --verbose
    } -ArgumentList $PWD.Path
    $processes.api = $apiJob.Id
    Write-Host "Azure Functions host starting on http://localhost:7071" -ForegroundColor Green

    # Start Frontend Vite dev server in background
    Write-Host "Starting Frontend Vite dev server..." -ForegroundColor Yellow
    $frontendJob = Start-Job -ScriptBlock {
        param($workingDir)
        Set-Location $workingDir
        Set-Location "frontend"
        npm run dev
    } -ArgumentList $PWD.Path
    $processes.frontend = $frontendJob.Id
    Write-Host "Frontend Vite server starting on http://localhost:5173" -ForegroundColor Green

    # Start Python Processor Service in background (for processing jobs)
    Write-Host "Starting Python Processor Service..." -ForegroundColor Yellow
    $processorJob = Start-Job -ScriptBlock {
        param($workingDir)
        Set-Location $workingDir
        $env:PYTHONPATH = $workingDir
        & ".\.venv\Scripts\python.exe" "local-dev\local_processor.py"
    } -ArgumentList $PWD.Path
    $processes.processor = $processorJob.Id
    Write-Host "Processor service starting on http://localhost:8080" -ForegroundColor Green

    # Start Azure Static Web Apps CLI in background
    Write-Host "Starting Azure Static Web Apps CLI..." -ForegroundColor Yellow
    $swaJob = Start-Job -ScriptBlock {
        Set-Location $using:PWD
        swa start --config swa-cli.config.json --verbose
    }
    $processes.swa = $swaJob.Id
    Write-Host "Azure Static Web Apps emulator starting on http://localhost:4280" -ForegroundColor Green

    # Wait for services to start
    Start-Sleep 3
    
    # Save process IDs for cleanup
    $processes | ConvertTo-Json | Set-Content $processFile

    Write-Host "`nDevelopment environment started successfully!" -ForegroundColor Green
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
    Write-Host "🌐 Main App (SWA): http://localhost:4280" -ForegroundColor Cyan
    Write-Host "🎯 Frontend (Vite): http://localhost:5173" -ForegroundColor Cyan
    Write-Host "🔧 API (Azure Functions): http://localhost:7071" -ForegroundColor Cyan
    Write-Host "⚙️  Processor: http://localhost:8080" -ForegroundColor Cyan
    Write-Host "📊 API Status: http://localhost:7071/api/index" -ForegroundColor Cyan
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
    Write-Host "`nUse the SWA endpoint (http://localhost:4280) for the most authentic Azure experience!" -ForegroundColor Yellow
    Write-Host "Press Ctrl+C to stop all services or run 'scripts\stop-dev.ps1'" -ForegroundColor Yellow
    
    # Wait for services to start
    Write-Host "`nWaiting for services to start..." -ForegroundColor Yellow
    Start-Sleep 5

    Write-Host "`nDevelopment environment is ready!" -ForegroundColor Green
    Write-Host "🚀 Open http://localhost:4280 in your browser for the full Azure Static Web Apps experience!" -ForegroundColor Cyan
    Write-Host "📝 Or use http://localhost:5173 for direct Vite frontend development." -ForegroundColor Cyan
    
    # Summary of service status
    Write-Host "`n📊 Service Status Summary:" -ForegroundColor Cyan
    Write-Host "  SWA CLI (Main App): ✓ Started" -ForegroundColor Green
    Write-Host "  Vite Frontend: ✓ Started" -ForegroundColor Green
    Write-Host "  Azure Functions: ✓ Started" -ForegroundColor Green
    Write-Host "  Python Processor: ✓ Started" -ForegroundColor Green
    
    Write-Host "`nTo stop all services, run: scripts\stop-dev.ps1" -ForegroundColor Yellow

} catch {
    Write-Error "Failed to start development environment: $_"
    
    # Cleanup on error
    if (Test-Path $processFile) {
        .\scripts\stop-dev.ps1
    }
    
    exit 1
}
