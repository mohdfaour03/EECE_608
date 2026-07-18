@echo off
rem Relocates EECE_608 out of OneDrive. Non-destructive. See move_project.ps1.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0move_project.ps1"
timeout /t 20
