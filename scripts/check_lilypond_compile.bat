@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"
cd /d "%REPO_ROOT%" || exit /b 1

set "DEFAULT_ROOT=data\normalized_dataset"

echo LilyPond compile check (default folder)
set "TARGET_ROOT=%DEFAULT_ROOT%"
set "TARGET_PATTERN=*.ly"

echo Using root: %TARGET_ROOT%
echo Using pattern: %TARGET_PATTERN%
uv run python scripts\check_lilypond_compile.py --root "%TARGET_ROOT%" --pattern "%TARGET_PATTERN%"
set "EXITCODE=%ERRORLEVEL%"

if not "%EXITCODE%"=="0" (
  echo Finished with failures. Exit code: %EXITCODE%
) else (
  echo All files compiled successfully.
)

pause
exit /b %EXITCODE%
