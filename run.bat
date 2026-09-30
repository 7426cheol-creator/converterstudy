@echo off
rem One-command start on Windows: creates .venv on first use, installs, opens http://127.0.0.1:8765
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  py -3 -m venv .venv || python -m venv .venv
  .venv\Scripts\python.exe -m pip install --upgrade pip
  .venv\Scripts\python.exe -m pip install -e ".[test]"
)
.venv\Scripts\python.exe -m convlab serve %*
