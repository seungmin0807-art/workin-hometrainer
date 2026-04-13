$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Error "Python virtual environment was not found. Expected: $python"
}

& $python (Join-Path $root "scripts\analyze_videos.py") `
    --correct (Join-Path $root "correct.mp4") `
    --wrong (Join-Path $root "wrong.mp4") `
    --output-dir (Join-Path $root "app")

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $python (Join-Path $root "scripts\serve.py") `
    --root (Join-Path $root "app") `
    --port 8000 `
    --open

