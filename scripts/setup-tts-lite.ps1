param(
  [string]$Python = $(if ($env:TTS_PYTHON_BOOTSTRAP) { $env:TTS_PYTHON_BOOTSTRAP } else { "py" })
)

$ErrorActionPreference = "Stop"

$TtsDir = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvDir = Join-Path $TtsDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

Write-Host "Creating/updating lightweight TTS Python environment..." -ForegroundColor Cyan
& $Python -m venv $VenvDir

Write-Host "Installing FastAPI/STT runtime dependencies..." -ForegroundColor Cyan
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install --upgrade -r (Join-Path $TtsDir "requirements-lite.txt")

Write-Host "TTS lite setup complete." -ForegroundColor Green
Write-Host "Run from backend: npm run dev:all" -ForegroundColor Green
