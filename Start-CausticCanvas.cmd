@echo off
setlocal
pushd "%~dp0" || exit /b 1
echo Starting CausticCanvas. Keep this window open while using the app.
call npm.cmd run dev -- --host 127.0.0.1 --open
set "result=%errorlevel%"
popd
if not "%result%"=="0" pause
exit /b %result%
