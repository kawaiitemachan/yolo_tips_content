@echo off
echo Starting detection with mosaic effect...

REM Activate virtual environment
call venv\Scripts\activate

REM Run custom detection script
echo Running custom detection script
python detect.py

echo Processing completed! Check the results folder.
pause