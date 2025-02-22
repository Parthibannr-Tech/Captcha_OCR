@echo off
cd /d "%~dp0"  REM
call venv\Scripts\activate.bat
python captcha_ocr.py
deactivate
pause
