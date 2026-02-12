@echo off
TITLE Sentinel AI - Facial Recognition System
echo [INFO] Launching from Portable Python Environment (F: Drive)...

:: Set the path to our local python
set PYTHON_BIN=.\python_bin\python.exe

:: Run the GPU check and then the App
"%PYTHON_BIN%" -m streamlit run app.py

pause