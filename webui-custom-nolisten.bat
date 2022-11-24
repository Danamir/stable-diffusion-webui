@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=--medvram --xformers

git fetch
git status --untracked-files=no
rem git pull
echo.

call webui.bat
