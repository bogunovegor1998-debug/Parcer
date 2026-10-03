@echo off
setlocal

rem Каталог проекта: по умолчанию — каталог, из которого запущен этот скрипт (..).
rem Можно переопределить переменной окружения PARCER_HOME.
if not defined PARCER_HOME set "PARCER_HOME=%~dp0.."
set "BASE=%PARCER_HOME%"
set "LOGDIR=%BASE%\logs"
set "LOG=%LOGDIR%\ingest_task_stdout.log"

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

echo [%date% %time%] START >> "%LOG%"

cd /d "%BASE%" >> "%LOG%" 2>>&1

echo [%date% %time%] PWD=%cd% >> "%LOG%"
echo [%date% %time%] Running python... >> "%LOG%"

"%BASE%\.venv\Scripts\python.exe" -u "%BASE%\ingest_telegram.py" --with-comments --since-date=2025-01-01 --comments-backfill=200 --batch=300 >> "%LOG%" 2>>&1

echo [%date% %time%] END exit_code=%errorlevel% >> "%LOG%"
exit /b %errorlevel%
