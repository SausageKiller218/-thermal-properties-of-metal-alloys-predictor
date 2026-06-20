@echo off
setlocal
cd /d "%~dp0"

set "EXE=AGLPredictor_web.exe"

if not exist "%EXE%" (
    if exist "agl.exe" (
        set "EXE=agl.exe"
    )
)

if not exist "%EXE%" (
    echo Ошибка: не найден AGLPredictor_web.exe или agl.exe рядом с run_server.cmd
    pause
    exit /b 1
)

if "%HOST%"=="" set "HOST=0.0.0.0"
if "%PORT%"=="" set "PORT=5000"
if "%THREADS%"=="" set "THREADS=4"

echo Запускаю AGL Predictor в режиме веб-сервера...
echo EXE: %EXE%
echo URL: http://%HOST%:%PORT%
echo.

"%EXE%" -server --host "%HOST%" --port "%PORT%" --threads "%THREADS%" %*

endlocal