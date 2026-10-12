@REM ----------------------------------------------------------------------------
@REM Maven Start Up Batch script for AxiomCloud Spring Boot Backend
@REM ----------------------------------------------------------------------------
@echo off
setlocal

set "DIR=%~dp0"
set "MAVEN_DIR=%DIR%.mvn\apache-maven-3.9.9"

if exist "%MAVEN_DIR%\bin\mvn.cmd" (
    "%MAVEN_DIR%\bin\mvn.cmd" %*
    exit /b %ERRORLEVEL%
)

where mvn >nul 2>nul
if %ERRORLEVEL% equ 0 (
    mvn %*
    exit /b %ERRORLEVEL%
)

echo [INFO] Maven not found. Running PowerShell wrapper to initialize...
powershell -NoProfile -ExecutionPolicy Bypass -File "%DIR%mvnw.ps1" %*
exit /b %ERRORLEVEL%
