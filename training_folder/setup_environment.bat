@echo off
echo Setting up environment for YOLO training...

REM Create virtual environment
python -m venv venv
call venv\Scripts\activate

REM Install required packages
pip install -U pip
pip install ultralytics matplotlib opencv-python pytorch-lightning

echo Environment setup complete!
echo You can now run train_yolov11n.bat to start training
pause