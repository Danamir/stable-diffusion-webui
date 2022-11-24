@echo off

set PYTHON=
set GIT=
set VENV_DIR=
rem set COMMANDLINE_ARGS=--listen --no-half --precision full
rem set COMMANDLINE_ARGS=--listen
set COMMANDLINE_ARGS=--listen --xformers

call webui.bat
