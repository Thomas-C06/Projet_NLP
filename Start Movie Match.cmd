@echo off
cd /d "%~dp0"
py scripts\launch.py
if errorlevel 1 pause
