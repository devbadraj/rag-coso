param([switch]$Offline)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:HF_HOME = Join-Path $PSScriptRoot '.local\huggingface'
$env:HF_HUB_OFFLINE = if ($Offline) { '1' } else { '0' }
$env:PYTHONUTF8 = '1'
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Install dependencies first: uv sync --python 3.12 --extra dev --locked'
}
& $pythonPath -m streamlit run app.py
