@echo off
echo Starting segmentation with mosaic effect...

REM Activate virtual environment
call venv\Scripts\activate

REM Install required dependencies if not already installed
pip install opencv-python-headless numpy

REM Run custom segmentation script
echo Running custom detection and mosaic application script
python detect.py

echo Processing completed! Check the results in runs\segment\mosaic_results
pause