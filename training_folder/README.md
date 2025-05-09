# YOLOv11n Training Project

This project is set up for training YOLOv11n models with your custom dataset.

## Project Structure
- `dataset/` - Contains your training data and data.yaml configuration
- `config.yaml` - Configuration file for training parameters
- `train_yolov11n.bat` - Windows batch file to start training with one click
- `setup_environment.bat` - Script to set up the Python environment
- `predict.bat` - Script to run inference with trained model

## How to Use

### First-time Setup
1. Make sure Python is installed on your Windows machine
2. Double-click `setup_environment.bat` to create a virtual environment and install required packages

### Training
1. Double-click `train_yolov11n.bat` to start the training process
2. The script will:
   - Activate the virtual environment
   - Use `config.yaml` if it exists, otherwise use default parameters
   - Start training using your dataset

### Prediction
1. After training completes, double-click `predict.bat` to run inference on validation images
2. Results will be saved in the `runs/predict` directory

## Training Results
Training results will be saved in the `runs/detect/yolov11n_custom_model` directory, including:
- Best weights: `best.pt`
- Last weights: `last.pt`
- Training metrics and charts
- Validation results

## Customization
To customize training parameters, edit the `config.yaml` file.

Key parameters:
- `model: yolov11n.pt` - Base model
- `epochs: 100` - Number of training epochs
- `imgsz: 640` - Image size for training
- `batch: 16` - Batch size
- `name: yolov11n_custom_model` - Name of the output model
- Various augmentation parameters

The system automatically handles different numbers of classes or class names defined in `data.yaml`.