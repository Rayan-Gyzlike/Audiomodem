# Единая команда запуска "Аудиомодема".
# Использование:  powershell -ExecutionPolicy Bypass -File start.ps1
$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\src"
& "$PSScriptRoot\.venv\Scripts\python.exe" run.py
