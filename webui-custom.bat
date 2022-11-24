@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=--medvram --listen --xformers
rem set COMMANDLINE_ARGS=--medvram --listen --xformers --api --cors-allow-origins=http://127.0.0.1:80,http://localhost
rem set COMMANDLINE_ARGS=--medvram --xformers --api --cors-allow-origins=http://127.0.0.1
rem set COMMANDLINE_ARGS=--medvram --xformers
rem set COMMANDLINE_ARGS=--medvram --listen

git fetch
git status --untracked-files=no
rem git pull
echo.

call webui.bat
