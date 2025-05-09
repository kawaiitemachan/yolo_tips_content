@echo off
echo Starting prediction with trained model...

REM Activate virtual environment
call venv\Scripts\activate

REM Default paths - change if needed
set MODEL_PATH=runs\detect\yolov11n_custom_model\weights\best.pt
set SOURCE_PATH=dataset\valid\images

REM Check if model exists
if not exist "%MODEL_PATH%" (
    echo Model not found at %MODEL_PATH%
    echo Please train the model first or specify correct model path
    pause
    exit /b
)

REM Run prediction
python -m ultralytics predict model=%MODEL_PATH% source=%SOURCE_PATH% conf=0.25 save=True

echo Prediction completed! Results saved in runs\predict
pause