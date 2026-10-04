$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path '.venv/Scripts/python.exe')) {
    throw 'Create the virtual environment and install requirements first. See README.md.'
}
& '.\.venv\Scripts\python.exe' app.py
