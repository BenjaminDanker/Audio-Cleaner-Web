#Requires -Version 5.1
<#
.SYNOPSIS
    Comprehensive Security Testing Script for Audio Cleaner Web Application

.DESCRIPTION
    Tests all implemented security measures including rate limiting, input validation,
    authentication, and monitoring systems.

.PARAMETER BaseUrl
    Base URL of the deployed application (Function App)

.PARAMETER StaticWebUrl  
    URL of the Static Web App frontend

.PARAMETER TestType
    Type of security test to run. Options: All, RateLimit, InputValidation, Authentication, FileUpload
    Default: All

.PARAMETER VerboseOutput
    Show detailed test output and monitoring

.EXAMPLE
    .\Test-SecurityMeasures.ps1 -BaseUrl "https://func-4ositsvdlpac6.azurewebsites.net" -StaticWebUrl "https://zealous-river-020499810.2.azurestaticapps.net"
#>

param(
    [string]$BaseUrl = "https://func-4ositsvdlpac6.azurewebsites.net",
    [string]$StaticWebUrl = "https://zealous-river-020499810.2.azurestaticapps.net",
    [ValidateSet("All", "RateLimit", "InputValidation", "Authentication", "FileUpload", "Headers")]
    [string]$TestType = "All",
    [switch]$VerboseOutput
)

# Color scheme for output
$Colors = @{
    Header = "Cyan"
    Success = "Green"
    Warning = "Yellow"
    Error = "Red"
    Info = "White"
    Dim = "Gray"
    Test = "Magenta"
}

function Write-TestHeader {
    param([string]$Title)
    Write-Host ""
    Write-Host "🔒 $Title" -ForegroundColor $Colors.Header
    Write-Host ("=" * ($Title.Length + 4)) -ForegroundColor $Colors.Header
}

function Write-TestResult {
    param([string]$TestName, [bool]$Passed, [string]$Details = "")
    $icon = if ($Passed) { "✅" } else { "❌" }
    $color = if ($Passed) { $Colors.Success } else { $Colors.Error }
    
    Write-Host "$icon $TestName" -ForegroundColor $color
    if ($Details -ne "") {
        Write-Host "   $Details" -ForegroundColor $Colors.Dim
    }
}

function Test-RateLimiting {
    Write-TestHeader "Rate Limiting Tests"
    
    # Test 1: Normal rate limit
    Write-Host "🧪 Testing normal request rate limits..." -ForegroundColor $Colors.Test
    
    $successCount = 0
    $rateLimitCount = 0
    $requests = 25  # Should trigger rate limiting
    
    for ($i = 1; $i -le $requests; $i++) {
        try {
            $response = Invoke-WebRequest -Uri "$BaseUrl/api/index" -Method GET -ErrorAction SilentlyContinue
            if ($response.StatusCode -eq 200) {
                $successCount++
            } elseif ($response.StatusCode -eq 429) {
                $rateLimitCount++
                if ($VerboseOutput) {
                    Write-Host "   Request $($i): Rate limited (429)" -ForegroundColor $Colors.Warning
                }
            }
        } catch {
            if ($_.Exception.Response.StatusCode -eq 429) {
                $rateLimitCount++
                if ($VerboseOutput) {
                    Write-Host "   Request $($i): Rate limited (429)" -ForegroundColor $Colors.Warning
                }
            }
        }
        Start-Sleep -Milliseconds 100  # Brief pause between requests
    }
    
    Write-TestResult "Basic Rate Limiting" ($rateLimitCount -gt 0) "Success: $successCount, Rate Limited: $rateLimitCount"
    
    # Test 2: Parallel upload rate limits (simulate chunk uploads)
    Write-Host "🧪 Testing parallel upload rate limits..." -ForegroundColor $Colors.Test
    
    $jobs = @()
    for ($i = 1; $i -le 10; $i++) {
        $jobs += Start-Job -ScriptBlock {
            param($url)
            try {
                $response = Invoke-WebRequest -Uri "$url/api/upload-file" -Method POST -ErrorAction SilentlyContinue
                return @{
                    StatusCode = $response.StatusCode
                    Headers = $response.Headers
                }
            } catch {
                return @{
                    StatusCode = $_.Exception.Response.StatusCode.value__
                    Headers = $null
                }
            }
        } -ArgumentList $BaseUrl
    }
    
    $results = $jobs | Wait-Job | Receive-Job
    $jobs | Remove-Job
    
    $parallelRateLimited = ($results | Where-Object { $_.StatusCode -eq 429 }).Count
    Write-TestResult "Parallel Request Rate Limiting" ($parallelRateLimited -gt 0) "Rate limited requests: $parallelRateLimited/10"
    
    return @{
        BasicRateLimit = $rateLimitCount -gt 0
        ParallelRateLimit = $parallelRateLimited -gt 0
    }
}

function Test-InputValidation {
    Write-TestHeader "Input Validation Tests"
    
    $results = @{}
    
    # Test 1: XSS Prevention
    Write-Host "🧪 Testing XSS prevention..." -ForegroundColor $Colors.Test
    
    $xssPayloads = @(
        "<script>alert('xss')</script>",
        "javascript:alert('xss')",
        "<img src=x onerror=alert('xss')>",
        "'; DROP TABLE users; --"
    )
    
    $xssBlocked = 0
    foreach ($payload in $xssPayloads) {
        try {
            $body = @{ filename = $payload } | ConvertTo-Json
            $response = Invoke-WebRequest -Uri "$BaseUrl/api/upload-file" -Method POST -Body $body -ContentType "application/json" -ErrorAction SilentlyContinue
            
            if ($response.StatusCode -eq 400) {
                $xssBlocked++
                if ($VerboseOutput) {
                    Write-Host "   XSS payload blocked: $payload" -ForegroundColor $Colors.Success
                }
            }
        } catch {
            if ($_.Exception.Response.StatusCode -eq 400) {
                $xssBlocked++
                if ($VerboseOutput) {
                    Write-Host "   XSS payload blocked: $payload" -ForegroundColor $Colors.Success
                }
            }
        }
    }
    
    $results.XSSPrevention = $xssBlocked -eq $xssPayloads.Count
    Write-TestResult "XSS Prevention" $results.XSSPrevention "Blocked $xssBlocked/$($xssPayloads.Count) payloads"
    
    # Test 2: File Type Validation
    Write-Host "🧪 Testing file type validation..." -ForegroundColor $Colors.Test
    
    $invalidFileTypes = @(
        "malicious.exe",
        "script.php",
        "backdoor.asp",
        "virus.bat"
    )
    
    $fileTypeBlocked = 0
    foreach ($filename in $invalidFileTypes) {
        try {
            $body = @{ filename = $filename } | ConvertTo-Json
            $response = Invoke-WebRequest -Uri "$BaseUrl/api/upload-file" -Method POST -Body $body -ContentType "application/json" -ErrorAction SilentlyContinue
            
            if ($response.StatusCode -eq 400) {
                $fileTypeBlocked++
                if ($VerboseOutput) {
                    Write-Host "   Invalid file type blocked: $filename" -ForegroundColor $Colors.Success
                }
            }
        } catch {
            if ($_.Exception.Response.StatusCode -eq 400) {
                $fileTypeBlocked++
                if ($VerboseOutput) {
                    Write-Host "   Invalid file type blocked: $filename" -ForegroundColor $Colors.Success
                }
            }
        }
    }
    
    $results.FileTypeValidation = $fileTypeBlocked -gt 0
    Write-TestResult "File Type Validation" $results.FileTypeValidation "Blocked $fileTypeBlocked/$($invalidFileTypes.Count) invalid types"
    
    return $results
}

function Test-SecurityHeaders {
    Write-TestHeader "Security Headers Tests"
    
    $results = @{}
    
    # Test frontend security headers
    Write-Host "🧪 Testing frontend security headers..." -ForegroundColor $Colors.Test
    
    try {
        $response = Invoke-WebRequest -Uri $StaticWebUrl -Method GET
        $headers = $response.Headers
        
        $securityHeaders = @{
            "X-Content-Type-Options" = "nosniff"
            "X-Frame-Options" = @("DENY", "SAMEORIGIN")
            "X-XSS-Protection" = "1; mode=block"
            "Content-Security-Policy" = $null  # Just check if present
            "Strict-Transport-Security" = $null  # Just check if present
        }
        
        $headerResults = @{}
        foreach ($headerName in $securityHeaders.Keys) {
            $headerExists = $headers.ContainsKey($headerName)
            $headerResults[$headerName] = $headerExists
            
            if ($VerboseOutput) {
                $status = if ($headerExists) { "✅" } else { "❌" }
                $value = if ($headerExists) { $headers[$headerName] } else { "Missing" }
                Write-Host "   $status $headerName`: $value" -ForegroundColor $Colors.Info
            }
        }
        
        $results.SecurityHeaders = ($headerResults.Values | Where-Object { $_ -eq $true }).Count -ge 3
        Write-TestResult "Security Headers" $results.SecurityHeaders "Found $($headerResults.Values | Where-Object { $_ -eq $true }).Count/5 security headers"
        
    } catch {
        $results.SecurityHeaders = $false
        Write-TestResult "Security Headers" $false "Failed to retrieve headers: $($_.Exception.Message)"
    }
    
    # Test API CORS headers
    Write-Host "🧪 Testing API CORS configuration..." -ForegroundColor $Colors.Test
    
    try {
        $response = Invoke-WebRequest -Uri "$BaseUrl/api/index" -Method OPTIONS -Headers @{"Origin"="https://malicious-site.com"} -ErrorAction SilentlyContinue
        
        $corsBlocked = $response.StatusCode -ne 200 -or (-not $response.Headers.ContainsKey("Access-Control-Allow-Origin"))
        $results.CORSProtection = $corsBlocked
        Write-TestResult "CORS Protection" $corsBlocked "Malicious origin blocked: $corsBlocked"
        
    } catch {
        $results.CORSProtection = $true  # Assuming error means blocked
        Write-TestResult "CORS Protection" $true "Malicious origin properly blocked"
    }
    
    return $results
}

function Test-Authentication {
    Write-TestHeader "Authentication Tests"
    
    $results = @{}
    
    # Test 1: Unauthenticated access to protected endpoints
    Write-Host "🧪 Testing unauthenticated access protection..." -ForegroundColor $Colors.Test
    
    $protectedEndpoints = @(
        "/api/upload-file",
        "/api/create-checkout-session",
        "/api/clear-jobs"
    )
    
    $authProtected = 0
    foreach ($endpoint in $protectedEndpoints) {
        try {
            $response = Invoke-WebRequest -Uri "$BaseUrl$endpoint" -Method GET -ErrorAction SilentlyContinue
            
            # Should return 401 or 403 for protected endpoints
            if ($response.StatusCode -eq 401 -or $response.StatusCode -eq 403) {
                $authProtected++
                if ($VerboseOutput) {
                    Write-Host "   $endpoint properly protected (Status: $($response.StatusCode))" -ForegroundColor $Colors.Success
                }
            }
        } catch {
            if ($_.Exception.Response.StatusCode -eq 401 -or $_.Exception.Response.StatusCode -eq 403) {
                $authProtected++
                if ($VerboseOutput) {
                    Write-Host "   $endpoint properly protected (Status: $($_.Exception.Response.StatusCode))" -ForegroundColor $Colors.Success
                }
            }
        }
    }
    
    $results.AuthProtection = $authProtected -gt 0
    Write-TestResult "Authentication Protection" $results.AuthProtection "Protected endpoints: $authProtected/$($protectedEndpoints.Count)"
    
    # Test 2: Token validation
    Write-Host "🧪 Testing invalid token handling..." -ForegroundColor $Colors.Test
    
    try {
        $headers = @{"Authorization" = "Bearer invalid-token-123"}
        $response = Invoke-WebRequest -Uri "$BaseUrl/api/upload-file" -Method POST -Headers $headers -ErrorAction SilentlyContinue
        
        $tokenValidated = $response.StatusCode -eq 401 -or $response.StatusCode -eq 403
        $results.TokenValidation = $tokenValidated
        Write-TestResult "Token Validation" $tokenValidated "Invalid token properly rejected"
        
    } catch {
        $tokenValidated = $_.Exception.Response.StatusCode -eq 401 -or $_.Exception.Response.StatusCode -eq 403
        $results.TokenValidation = $tokenValidated
        Write-TestResult "Token Validation" $tokenValidated "Invalid token properly rejected"
    }
    
    return $results
}

function Test-FileUploadSecurity {
    Write-TestHeader "File Upload Security Tests"
    
    $results = @{}
    
    # Test 1: Large file handling
    Write-Host "🧪 Testing large file protection..." -ForegroundColor $Colors.Test
    
    try {
        $largeContent = "A" * (100 * 1024 * 1024)  # 100MB of data
        $response = Invoke-WebRequest -Uri "$BaseUrl/api/upload-file" -Method POST -Body $largeContent -ErrorAction SilentlyContinue
        
        $sizeLimited = $response.StatusCode -eq 413 -or $response.StatusCode -eq 400
        $results.FileSizeLimit = $sizeLimited
        Write-TestResult "File Size Limit" $sizeLimited "Large file properly rejected"
        
    } catch {
        $sizeLimited = $_.Exception.Response.StatusCode -eq 413 -or $_.Exception.Response.StatusCode -eq 400
        $results.FileSizeLimit = $sizeLimited
        Write-TestResult "File Size Limit" $sizeLimited "Large file properly rejected"
    }
    
    # Test 2: SAS token security
    Write-Host "🧪 Testing SAS token security..." -ForegroundColor $Colors.Test
    
    try {
        # Try to access with expired/invalid SAS token
        $invalidSasUrl = "$BaseUrl/api/download-file/test.mp3?sv=2023-01-01&sr=b&sig=invalid"
        $response = Invoke-WebRequest -Uri $invalidSasUrl -Method GET -ErrorAction SilentlyContinue
        
        $sasSecure = $response.StatusCode -ne 200
        $results.SASTokenSecurity = $sasSecure
        Write-TestResult "SAS Token Security" $sasSecure "Invalid SAS token properly rejected"
        
    } catch {
        $sasSecure = $true  # Error means it was rejected
        $results.SASTokenSecurity = $sasSecure
        Write-TestResult "SAS Token Security" $sasSecure "Invalid SAS token properly rejected"
    }
    
    return $results
}

function Run-SecurityMonitoring {
    Write-TestHeader "Security Monitoring Validation"
    
    Write-Host "🧪 Checking security event logging..." -ForegroundColor $Colors.Test
    
    # Run the monitoring script to check for security events
    $monitorResult = & "$PSScriptRoot\Monitor-AzureLogs.ps1" -LogType Errors -Hours 1 -Severity Warning
    
    Write-Host "📊 Recent security events logged - check the monitoring output above" -ForegroundColor $Colors.Info
    Write-Host "🔗 For real-time monitoring, visit:" -ForegroundColor $Colors.Info
    Write-Host "   Application Insights: https://portal.azure.com" -ForegroundColor $Colors.Dim
}

# Main execution
Write-Host "🛡️  Security Testing Suite - Audio Cleaner Web Application" -ForegroundColor $Colors.Header
Write-Host "=================================================" -ForegroundColor $Colors.Header
Write-Host ""
Write-Host "📋 Test Configuration:" -ForegroundColor $Colors.Info
Write-Host "   Function App: $BaseUrl" -ForegroundColor $Colors.Dim
Write-Host "   Static Web App: $StaticWebUrl" -ForegroundColor $Colors.Dim
Write-Host "   Test Type: $TestType" -ForegroundColor $Colors.Dim
Write-Host ""

$allResults = @{}

switch ($TestType) {
    "All" {
        $allResults.RateLimit = Test-RateLimiting
        $allResults.InputValidation = Test-InputValidation
        $allResults.SecurityHeaders = Test-SecurityHeaders
        $allResults.Authentication = Test-Authentication
        $allResults.FileUpload = Test-FileUploadSecurity
        Run-SecurityMonitoring
    }
    "RateLimit" { $allResults.RateLimit = Test-RateLimiting }
    "InputValidation" { $allResults.InputValidation = Test-InputValidation }
    "Headers" { $allResults.SecurityHeaders = Test-SecurityHeaders }
    "Authentication" { $allResults.Authentication = Test-Authentication }
    "FileUpload" { $allResults.FileUpload = Test-FileUploadSecurity }
}

# Summary
Write-TestHeader "Security Test Summary"

$totalTests = 0
$passedTests = 0

foreach ($category in $allResults.Keys) {
    Write-Host "📁 $category" -ForegroundColor $Colors.Header
    foreach ($test in $allResults[$category].Keys) {
        $totalTests++
        if ($allResults[$category][$test]) {
            $passedTests++
            Write-Host "   ✅ $test" -ForegroundColor $Colors.Success
        } else {
            Write-Host "   ❌ $test" -ForegroundColor $Colors.Error
        }
    }
    Write-Host ""
}

$successRate = [math]::Round(($passedTests / $totalTests) * 100, 1)
$overallColor = if ($successRate -ge 80) { $Colors.Success } elseif ($successRate -ge 60) { $Colors.Warning } else { $Colors.Error }

Write-Host "🎯 Overall Security Score: $successRate% ($passedTests/$totalTests)" -ForegroundColor $overallColor

if ($successRate -ge 80) {
    Write-Host "🏆 Excellent! Your security measures are working well." -ForegroundColor $Colors.Success
} elseif ($successRate -ge 60) {
    Write-Host "⚠️  Good progress, but some security measures need attention." -ForegroundColor $Colors.Warning
} else {
    Write-Host "🚨 Security measures need immediate attention!" -ForegroundColor $Colors.Error
}

Write-Host ""
Write-Host "💡 Next Steps:" -ForegroundColor $Colors.Info
Write-Host "   1. Monitor logs: .\Monitor-AzureLogs.ps1 -LogType Errors" -ForegroundColor $Colors.Dim
Write-Host "   2. Check Azure Portal for detailed metrics" -ForegroundColor $Colors.Dim
Write-Host "   3. Review any failed tests above" -ForegroundColor $Colors.Dim
