@echo off
setlocal enabledelayedexpansion

:: Get system language via PowerShell (wmic is removed on Windows 11 24H2+)
for /f "usebackq delims=" %%a in (`powershell -NoProfile -Command "(Get-Culture).Name"`) do set "LANG_CODE=%%a"

:: Map language code to app language (zh-* -> zh, everything else -> en)
if "!LANG_CODE:~0,2!"=="zh" (
    set "LANG=zh"
) else (
    set "LANG=en"
)

set SCRIPT_DIR=%~dp0
set PYTHON_EXE=%SCRIPT_DIR%python-embed\python.exe
cd src\webui
%PYTHON_EXE% app.py --lang=%LANG%
pause
