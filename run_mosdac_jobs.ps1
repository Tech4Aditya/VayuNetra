$ErrorActionPreference = 'Stop'

if (-not $env:MOSDAC_USERNAME -or -not $env:MOSDAC_PASSWORD) {
    throw 'Set MOSDAC_USERNAME and MOSDAC_PASSWORD in the current PowerShell session before running this script.'
}

$root = (Get-Location).Path

$apiZip = Join-Path $root 'backend\data\mosdac_jobs\mdapi.zip'
$apiDir = Join-Path $root 'backend\data\mosdac_jobs\_mdapi'

# ============================================================
# DOWNLOAD OFFICIAL MOSDAC CLIENT
# ============================================================

if (-not (Test-Path $apiZip)) {
    Write-Host 'Downloading official MOSDAC mdapi...'

    Invoke-WebRequest `
        -Uri 'https://mosdac.gov.in/software/mdapi.zip' `
        -OutFile $apiZip
}

if (Test-Path $apiDir) {
    Remove-Item $apiDir -Recurse -Force
}

New-Item `
    -ItemType Directory `
    -Path $apiDir |
    Out-Null

Write-Host 'Extracting MOSDAC mdapi...'

Expand-Archive `
    -Path $apiZip `
    -DestinationPath $apiDir `
    -Force

$mdapi = Get-ChildItem `
    $apiDir `
    -Filter 'mdapi.py' `
    -Recurse |
    Select-Object -First 1

if (-not $mdapi) {
    throw 'mdapi.py not found inside downloaded mdapi.zip'
}

Write-Host "Using mdapi: $($mdapi.FullName)"


# ============================================================
# MOSDAC CLIENT COMPATIBILITY FIX
# ============================================================

$mdapiText = Get-Content `
    $mdapi.FullName `
    -Raw

$oldLine = 'username = user_creds.get("username/email", "")'

$newLine = 'username = user_creds.get("username", "") or user_creds.get("username/email", "")'

if ($mdapiText.Contains($oldLine)) {

    $mdapiText = $mdapiText.Replace(
        $oldLine,
        $newLine
    )

    [System.IO.File]::WriteAllText(
        $mdapi.FullName,
        $mdapiText,
        [System.Text.UTF8Encoding]::new($false)
    )

    Write-Host 'MOSDAC client compatibility patch: APPLIED'
}
else {
    Write-Host 'MOSDAC client compatibility patch: not required'
}


# ============================================================
# FANI PAGINATED DOWNLOAD
#
# MOSDAC max count = 100.
#
# We use one-day windows so the half-hourly INSAT product
# remains comfortably below the limit.
# ============================================================

$jobsRoot = Join-Path `
    $root `
    'backend\data\mosdac_jobs'

$fanniJob = Join-Path `
    $jobsRoot `
    'FANI'

$template = Join-Path `
    $fanniJob `
    'config.template.json'

if (-not (Test-Path $template)) {
    throw "FANI template not found: $template"
}

$templateText = Get-Content `
    $template `
    -Raw


# ============================================================
# FANI DATE WINDOWS
# ============================================================

$startDate = [datetime]'2019-04-25'
$endDate   = [datetime]'2019-05-07'

$currentDate = $startDate

while ($currentDate -lt $endDate) {

    # One-day window.
    #
    # The next iteration begins on the following date.
    $windowStart = $currentDate
    $windowEnd = $currentDate.AddDays(1)

    $startString = $windowStart.ToString('yyyy-MM-dd')
    $endString = $windowEnd.ToString('yyyy-MM-dd')

    Write-Host ''
    Write-Host '============================================================'
    Write-Host "FANI MOSDAC WINDOW: $startString -> $endString"
    Write-Host '============================================================'


    # --------------------------------------------------------
    # Build configuration from template
    # --------------------------------------------------------

    $cfg = $templateText

    $cfg = $cfg.Replace(
        '${MOSDAC_USERNAME}',
        $env:MOSDAC_USERNAME
    ).Replace(
        '${MOSDAC_PASSWORD}',
        $env:MOSDAC_PASSWORD
    )


    # --------------------------------------------------------
    # Replace date range
    # --------------------------------------------------------

    $cfg = $cfg.Replace(
        '"startTime": "2019-04-25"',
        "`"startTime`": `"$startString`""
    )

    $cfg = $cfg.Replace(
        '"endTime": "2019-05-05"',
        "`"endTime`": `"$endString`""
    )


    # --------------------------------------------------------
    # Explicitly keep count below MOSDAC maximum
    # --------------------------------------------------------

    $cfg = $cfg.Replace(
        '"count": "100"',
        '"count": "100"'
    )


    # --------------------------------------------------------
    # Write temporary config
    # --------------------------------------------------------

    $configPath = Join-Path `
        $apiDir `
        'config.json'

    [System.IO.File]::WriteAllText(
        $configPath,
        $cfg,
        [System.Text.UTF8Encoding]::new($false)
    )


    # --------------------------------------------------------
    # Validate JSON
    # --------------------------------------------------------

    try {

        Get-Content `
            $configPath `
            -Raw |
            ConvertFrom-Json |
            Out-Null

        Write-Host 'Config JSON: OK'

    }
    catch {

        throw `
            "Generated config.json is invalid JSON: $($_.Exception.Message)"
    }


    # --------------------------------------------------------
    # Verify configuration
    # --------------------------------------------------------

    $testConfig = Get-Content `
        $configPath `
        -Raw |
        ConvertFrom-Json

    $testUsername =
        $testConfig.user_credentials.username

    $testStart =
        $testConfig.search_parameters.startTime

    $testEnd =
        $testConfig.search_parameters.endTime

    $testCount =
        $testConfig.search_parameters.count


    if (-not $testUsername) {
        throw 'Username is missing from generated config.json'
    }


    Write-Host "MOSDAC username: $testUsername"
    Write-Host "Start: $testStart"
    Write-Host "End:   $testEnd"
    Write-Host "Count: $testCount"


    # --------------------------------------------------------
    # Run MOSDAC client
    # --------------------------------------------------------

    Push-Location $apiDir

    try {

        python $mdapi.FullName

    }
    finally {

        Pop-Location
    }


    # --------------------------------------------------------
    # Next window
    # --------------------------------------------------------

    $currentDate = $currentDate.AddDays(1)
}


Write-Host ''
Write-Host '============================================================'
Write-Host 'FANI PAGINATED MOSDAC DOWNLOAD FINISHED'
Write-Host '============================================================'
