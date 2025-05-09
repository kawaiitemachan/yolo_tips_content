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

REM Check GPU availability
python -c "import torch; print('CUDA_AVAILABLE=' + str(torch.cuda.is_available()))" > temp.txt
set /p CUDA_CHECK=<temp.txt
del temp.txt

REM Check if config file exists and use it, otherwise use default parameters
if exist "config.yaml" (
    echo Using configuration from config.yaml

    REM Create temporary config file with fallback to CPU if CUDA not available
    if "%CUDA_CHECK%"=="CUDA_AVAILABLE=False" (
        echo CUDA not available, falling back to CPU...
        python -c "import yaml; cfg = yaml.safe_load(open('config.yaml')); cfg['device'] = 'cpu'; yaml.dump(cfg, open('config_temp.yaml', 'w'))"
        yolo train cfg=config_temp.yaml
        del config_temp.yaml
    ) else (
        yolo train cfg=config.yaml
    )
) else (
    echo Config file not found, using default parameters

    REM Add CPU device parameter if CUDA not available
    if "%CUDA_CHECK%"=="CUDA_AVAILABLE=False" (
        echo CUDA not available, falling back to CPU...
        yolo train model=yolo11n-seg.pt data=dataset/data.yaml epochs=100 imgsz=640 batch=4 name=yolov11n_custom_model device=cpu
    ) else (
        yolo train model=yolo11n-seg.pt data=dataset/data.yaml epochs=100 imgsz=640 batch=4 name=yolov11n_custom_model
    )
)

echo Training completed
pause