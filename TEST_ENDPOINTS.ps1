$ErrorActionPreference = "Stop"
$base = "http://127.0.0.1:8000"

function Test-Get($path) {
    try {
        $r = Invoke-RestMethod "$base$path" -TimeoutSec 60
        if ($r.status -eq "error") { throw $r.error }
        Write-Host ("PASS  GET  " + $path) -ForegroundColor Green
        return $r
    } catch {
        Write-Host ("FAIL  GET  " + $path + " :: " + $_.Exception.Message) -ForegroundColor Red
        return $null
    }
}

function Test-PostJson($path, $body) {
    try {
        $json = $body | ConvertTo-Json -Depth 8 -Compress
        $r = Invoke-RestMethod "$base$path" -Method Post -ContentType "application/json" -Body $json -TimeoutSec 60
        if ($r.status -eq "error") { throw $r.error }
        Write-Host ("PASS  POST " + $path) -ForegroundColor Green
        return $r
    } catch {
        Write-Host ("FAIL  POST " + $path + " :: " + $_.Exception.Message) -ForegroundColor Red
        return $null
    }
}

$null = Test-Get "/health"
$null = Test-Get "/data_sources"
$demo = Test-Get "/demo_sequence"
if ($demo -and $demo.frames) {
    $pred = Test-PostJson "/predict" @{ frames = $demo.frames }
    if ($pred) {
        Write-Host ("      mode=" + $pred.mode + " source=" + $pred.source) -ForegroundColor Cyan
    }
} else {
    Write-Host "SKIP  POST /predict (synthetic demo unavailable)" -ForegroundColor Yellow
}

$insat = Test-Get "/insat_demo"
if ($insat) {
    $null = Test-Get "/insat_preview/tir1"
    $null = Test-Get "/insat_preview/wv"
} else {
    Write-Host "SKIP  INSAT preview checks (no processed sample)" -ForegroundColor Yellow
}

$null = Test-Get "/metrics"
$null = Test-Get "/evaluation_summary"
$null = Test-Get "/track"
Write-Host "Smoke test complete." -ForegroundColor Cyan
