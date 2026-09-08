$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$environmentPath = Join-Path $projectRoot ".venv-cpu"
$pythonPath = Join-Path $environmentPath "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    py -3.12 -m venv $environmentPath
    if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed." }
}

& $pythonPath -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
& $pythonPath -m pip install -r (Join-Path $projectRoot "requirements-cpu.txt")
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }

$requiredModels = @(
    (Join-Path $projectRoot "engine\models\comic_base.pt")
)
foreach ($model in $requiredModels) {
    if (-not (Test-Path -LiteralPath $model)) {
        throw "Missing model file: $model. Download and extract the model pack first."
    }
}

Push-Location -LiteralPath $projectRoot
try {
& $pythonPath -m PyInstaller `
    --noconfirm `
    --clean `
    --distpath (Join-Path $projectRoot "dist\cpu") `
    --workpath (Join-Path $projectRoot "build\cpu") `
    (Join-Path $projectRoot "ComicEbookConverter.spec")
    if ($LASTEXITCODE -ne 0) { throw "Application build failed." }
} finally {
    Pop-Location
}
