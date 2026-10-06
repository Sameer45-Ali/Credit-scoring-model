@echo off
title FinTech Credit Scoring Microservice
cd /d "%~dp0"
echo Starting Credit Scoring Microservice...
echo.
python run_server.py
pause
