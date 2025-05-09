@echo off
echo Starting YOLO11n training...

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate
    pip install ultralytics
) else (
    call venv\Scripts\activate
)

REM Check if config file exists and use it, otherwise use default parameters
if exist "config.yaml" (
    echo Using configuration from config.yaml
    yolo train cfg=config.yaml
) else (
    echo Config file not found, using default parameters
    yolo train model=yolo11n-seg.pt data=dataset/data.yaml epochs=100 imgsz=640 batch=4 name=yolov11n_custom_model
)

echo Training completed
pause