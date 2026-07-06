$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$LogDir = Join-Path $Root "data\mrag\logs"
$ResultLog = Join-Path $LogDir "soak_test_6h.jsonl"
$PidFile = Join-Path $LogDir "soak_test_6h.pid"
$StdoutLog = Join-Path $LogDir "soak_test_6h.stdout.log"
$StderrLog = Join-Path $LogDir "soak_test_6h.stderr.log"

New-Item -ItemType Directory -Force $LogDir | Out-Null

if (Test-Path $PidFile) {
    $ExistingPid = Get-Content $PidFile -Raw
    $Existing = Get-Process -Id $ExistingPid -ErrorAction SilentlyContinue
    if ($Existing) {
        Write-Host "Six-hour soak test is already running. PID: $ExistingPid"
        exit 0
    }
}

foreach ($Path in ($ResultLog, $PidFile, $StdoutLog, $StderrLog)) {
    if (Test-Path $Path) {
        Remove-Item -LiteralPath $Path
    }
}

$Process = Start-Process `
    -FilePath $Python `
    -ArgumentList @(
        "scripts\soak_test.py",
        "--duration-seconds", "21600",
        "--interval-seconds", "30",
        "--mcp-every", "20",
        "--output", "data\mrag\logs\soak_test_6h.jsonl"
    ) `
    -WorkingDirectory $Root `
    -WindowStyle Hidden `
    -RedirectStandardOutput $StdoutLog `
    -RedirectStandardError $StderrLog `
    -PassThru

Set-Content -LiteralPath $PidFile -Value $Process.Id -Encoding ASCII
Write-Host "Six-hour soak test started. PID: $($Process.Id)"
Write-Host "Run check_soak_test.cmd to inspect progress."
