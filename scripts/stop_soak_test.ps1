$Root = Split-Path -Parent $PSScriptRoot
$PidFile = Join-Path $Root "data\mrag\logs\soak_test_6h.pid"

if (-not (Test-Path $PidFile)) {
    Write-Host "No soak-test PID file found."
    exit 0
}

$ProcessId = (Get-Content $PidFile -Raw).Trim()
$Process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
if ($Process) {
    Stop-Process -Id $ProcessId -Force
    Write-Host "Stopped soak-test process $ProcessId"
} else {
    Write-Host "Soak-test process $ProcessId is not running."
}
Remove-Item -LiteralPath $PidFile
