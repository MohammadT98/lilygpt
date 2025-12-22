@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"
cd /d "%REPO_ROOT%" || exit /b 1

set "DEFAULT_ROOT=data\normalized_dataset"
echo LilyPond compile check
set /p TARGET_ROOT=Enter folder to scan [%DEFAULT_ROOT%]: 
if "%TARGET_ROOT%"=="" set "TARGET_ROOT=%DEFAULT_ROOT%"

echo Using root: %TARGET_ROOT%
uv run python scripts\check_lilypond_compile.py --root "%TARGET_ROOT%"
set "EXITCODE=%ERRORLEVEL%"

if not "%EXITCODE%"=="0" (
  echo Finished with failures. Exit code: %EXITCODE%
) else (
  echo All files compiled successfully.
)

pause
exit /b %EXITCODE%
