# Audio Cleaner Pro - Start Development Environment
Write-Host "Starting Audio Cleaner Pro Development Environment..." -ForegroundColor Green

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

# Stop any existing processes
Write-Host "Stopping any existing development processes..." -ForegroundColor Yellow
.\scripts\stop-dev.ps1

# Create process tracking file
$processFile = "dev-processes.json"
$processes = @{}

try {
    # Start Docker Desktop if not running (for future use with containers)
    Write-Host "Checking Docker Desktop..." -ForegroundColor Yellow
    if (Test-Command "docker") {
        $dockerStatus = docker info 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Docker is running" -ForegroundColor Green
        } else {
            Write-Host "Docker is not running. Some features may not work." -ForegroundColor Yellow
        }
    } else {
        Write-Host "Docker not found. Local development will work without containers." -ForegroundColor Yellow
    }

    # Install/update API dependencies
    Write-Host "Installing API dependencies..." -ForegroundColor Yellow
    Push-Location "api"
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install API dependencies"
        Pop-Location
        exit 1
    }
    Pop-Location

    # Install/update Frontend dependencies
    Write-Host "Installing Frontend dependencies..." -ForegroundColor Yellow
    Push-Location "frontend"
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install frontend dependencies"
        Pop-Location
        exit 1
    }
    Pop-Location

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
    pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install Python dependencies"
        exit 1
    }

    # Create necessary directories
    Write-Host "Creating required directories..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path "api\temp\jobs" | Out-Null
    New-Item -ItemType Directory -Force -Path "api\temp\uploads" | Out-Null
    New-Item -ItemType Directory -Force -Path "api\temp\downloads" | Out-Null

    # Start Express API server in background
    Write-Host "Starting Express API server..." -ForegroundColor Yellow
    Push-Location "api"
    $apiJob = Start-Job -ScriptBlock {
        Set-Location $using:PWD
        Set-Location "api"
        node local-server.js
    }
    Pop-Location
    $processes.api = $apiJob.Id
    Write-Host "API server starting on http://localhost:7071" -ForegroundColor Green

    # Wait a moment for API to start
    Start-Sleep 3

    # Start Python Processor Service in background
    Write-Host "Starting Python Processor Service..." -ForegroundColor Yellow
    $processorJob = Start-Job -ScriptBlock {
        param($workingDir)
        Set-Location $workingDir
        & ".\.venv\Scripts\python.exe" "local_processor.py"
    } -ArgumentList $PWD.Path
    
    $processes.processor = $processorJob.Id
    Write-Host "Processor service starting on http://localhost:8080" -ForegroundColor Green

    # Wait a moment for Processor to start (longer wait due to AI model loading)
    Start-Sleep 5
    
    # Check processor job status and provide better error handling
    $processorJobState = Get-Job -Id $processes.processor
    if ($processorJobState.State -eq "Failed") {
        Write-Host "Processor job failed. Checking error..." -ForegroundColor Red
        $processorError = Receive-Job -Id $processes.processor 2>&1 | Out-String
        if ($processorError -match "codec") {
            Write-Host "Unicode encoding issue detected. This is expected on some Windows systems." -ForegroundColor Yellow
            Write-Host "The processor should still work despite the warning." -ForegroundColor Yellow
        } else {
            Write-Host "Processor error: $processorError" -ForegroundColor Red
        }
        Write-Host "If processor doesn't respond, you may need to start manually: python local_processor.py" -ForegroundColor Yellow
    }

    # Start Frontend development server in background
    Write-Host "Starting Frontend development server..." -ForegroundColor Yellow
    Push-Location "frontend"
    $frontendJob = Start-Job -ScriptBlock {
        Set-Location $using:PWD
        Set-Location "frontend"
        npm run dev
    }
    Pop-Location
    $processes.frontend = $frontendJob.Id
    Write-Host "Frontend server starting on http://localhost:5173" -ForegroundColor Green

    # Save process IDs for cleanup
    $processes | ConvertTo-Json | Set-Content $processFile

    Write-Host "`nDevelopment environment started successfully!" -ForegroundColor Green
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
    Write-Host "🌐 Frontend: http://localhost:5173" -ForegroundColor Cyan
    Write-Host "🔧 API: http://localhost:7071" -ForegroundColor Cyan
    Write-Host "⚙️  Processor: http://localhost:8080" -ForegroundColor Cyan
    Write-Host "📊 API Status: http://localhost:7071/api/auth" -ForegroundColor Cyan
    Write-Host "🏥 Processor Health: http://localhost:8080/health" -ForegroundColor Cyan
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
    Write-Host "`nPress Ctrl+C to stop all services or run 'scripts\stop-dev.ps1'" -ForegroundColor Yellow
    
    # Wait for user input or keep running
    Write-Host "`nWaiting for services to start... (checking in 5 seconds)" -ForegroundColor Yellow
    Start-Sleep 5

    # Check if services are running
    Write-Host "`nChecking service status..." -ForegroundColor Yellow
    
    # Test API endpoint
    try {
        Invoke-WebRequest -Uri "http://localhost:7071/api/auth" -TimeoutSec 5 -UseBasicParsing | Out-Null
        Write-Host "✓ API is responding" -ForegroundColor Green
    } catch {
        Write-Host "⚠ API may still be starting up..." -ForegroundColor Yellow
    }

    # Test Processor (with retries due to AI model loading time)
    $processorHealthy = $false
    for ($i = 1; $i -le 3; $i++) {
        try {
            Invoke-WebRequest -Uri "http://localhost:8080/health" -TimeoutSec 10 -UseBasicParsing | Out-Null
            Write-Host "✓ Processor is responding" -ForegroundColor Green
            $processorHealthy = $true
            break
        } catch {
            if ($i -eq 3) {
                Write-Host "⚠ Processor not responding after 3 attempts" -ForegroundColor Yellow
                Write-Host "  Processor may need more time to load AI model..." -ForegroundColor Yellow
                Write-Host "  You can check status at: http://localhost:8080/health" -ForegroundColor Yellow
                
                # Check job status for more info
                $processorJobState = Get-Job -Id $processes.processor -ErrorAction SilentlyContinue
                if ($processorJobState -and $processorJobState.State -eq "Failed") {
                    Write-Host "  Processor job failed. You may need to start manually: python local_processor.py" -ForegroundColor Red
                }
            } else {
                Write-Host "⚠ Processor attempt $i/3 - waiting 5 more seconds..." -ForegroundColor Yellow
                Start-Sleep 5
            }
        }
    }

    # Test Frontend
    try {
        Invoke-WebRequest -Uri "http://localhost:5173" -TimeoutSec 5 -UseBasicParsing | Out-Null
        Write-Host "✓ Frontend is responding" -ForegroundColor Green
    } catch {
        Write-Host "⚠ Frontend may still be starting up..." -ForegroundColor Yellow
    }

    Write-Host "`nDevelopment environment is ready!" -ForegroundColor Green
    Write-Host "Open http://localhost:5173 in your browser to access the application." -ForegroundColor Cyan
    Write-Host "`nTo stop all services, run: scripts\stop-dev.ps1" -ForegroundColor Yellow

} catch {
    Write-Error "Failed to start development environment: $_"
    
    # Cleanup on error
    if (Test-Path $processFile) {
        .\scripts\stop-dev.ps1
    }
    
    exit 1
}
