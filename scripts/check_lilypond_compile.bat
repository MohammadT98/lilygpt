@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"
cd /d "%REPO_ROOT%" || exit /b 1

set "DEFAULT_ROOT=data\normalized_dataset"
set "LOG_DIR=data\logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo LilyPond compile check (default folder)
set "TARGET_ROOT=%DEFAULT_ROOT%"
set "TARGET_PATTERN=*.ly"

set "TS=%DATE%_%TIME%"
set "TS=%TS:/=-%"
set "TS=%TS::=-%"
set "TS=%TS:.=-%"
set "TS=%TS: =_%"
set "LOG_PATH=%LOG_DIR%\compile_check_%TS%.log"
set "TMP_LOG=%TEMP%\compile_check_%TS%.log"

echo Using root: %TARGET_ROOT%
echo Using pattern: %TARGET_PATTERN%

if exist "%LOCALAPPDATA%\uv\uv.exe" (
  set "UV_CMD=%LOCALAPPDATA%\uv\uv.exe"
) else (
  set "UV_CMD=uv"
)

%UV_CMD% --version > "%TMP_LOG%" 2>&1
if "%ERRORLEVEL%"=="0" (
  %UV_CMD% run python scripts\check_lilypond_compile.py --root "%TARGET_ROOT%" --pattern "%TARGET_PATTERN%" >> "%TMP_LOG%" 2>&1
) else (
  python scripts\check_lilypond_compile.py --root "%TARGET_ROOT%" --pattern "%TARGET_PATTERN%" > "%TMP_LOG%" 2>&1
)
set "EXITCODE=%ERRORLEVEL%"

if not "%EXITCODE%"=="0" (
  echo Finished with failures. Exit code: %EXITCODE%
) else (
  echo All files compiled successfully.
)

pause

move /Y "%TMP_LOG%" "%LOG_PATH%" > nul
echo Log saved to: %LOG_PATH%
pause
exit /b %EXITCODE%
