import os
import cv2
import glob
from ultralytics import YOLO

def apply_mosaic(image, box, mosaic_size):
    """Apply mosaic to a detected object in the image."""
    x1, y1, x2, y2 = map(int, box)
    
    # Extract the region to apply mosaic
    region = image[y1:y2, x1:x2]
    
    # Apply mosaic effect
    height, width = region.shape[:2]
    if height > 0 and width > 0:  # Ensure region is valid
        # Calculate mosaic block sizes
        mosaic_h = max(1, int(height / mosaic_size))
        mosaic_w = max(1, int(width / mosaic_size))
        
        # Resize down, then resize up to create mosaic effect
        small = cv2.resize(region, (mosaic_w, mosaic_h), interpolation=cv2.INTER_LINEAR)
        mosaic = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
        
        # Place mosaic region back into the image
        image[y1:y2, x1:x2] = mosaic
    
    return image

def main():
    # Get model path
    model_path = os.path.join("runs", "detect", "yolov11n_custom_model", "weights", "best.pt")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        alternative_path = input("Enter alternative model path (or press enter to exit): ")
        if alternative_path.strip():
            model_path = alternative_path
        else:
            print("Exiting program.")
            return
    
    # Load YOLO model
    model = YOLO(model_path)
    
    # Get folder path from user
    folder_path = input("Enter folder path containing images to process: ")
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return
    
    # Get mosaic size from user
    mosaic_size = input("Enter mosaic pixel size (smaller = more pixelated, recommended 8-20): ")
    try:
        mosaic_size = int(mosaic_size)
        if mosaic_size <= 0:
            print("Mosaic size must be positive. Using default value of 10.")
            mosaic_size = 10
    except ValueError:
        print("Invalid input. Using default mosaic size of 10.")
        mosaic_size = 10
    
    # Get confidence threshold from user
    confidence = input("Enter detection confidence threshold (0.1-1.0, default=0.25): ")
    try:
        confidence = float(confidence)
        if confidence <= 0 or confidence > 1:
            print("Confidence must be between 0.1 and 1.0. Using default value of 0.25.")
            confidence = 0.25
    except ValueError:
        print("Invalid input. Using default confidence of 0.25.")
        confidence = 0.25
    
    # Create output directory
    output_dir = os.path.join("runs", "detect", "mosaic_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all images in the specified folder
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    if not image_files:
        print(f"No images found in {folder_path}")
        return
    
    print(f"Processing {len(image_files)} images...")
    
    # Process each image
    for img_path in image_files:
        # Get base filename
        base_name = os.path.basename(img_path)
        output_path = os.path.join(output_dir, f"mosaic_{base_name}")
        
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read image: {img_path}")
            continue
        
        # Run detection
        results = model(img, conf=confidence)
        
        # Process results and apply mosaic
        for result in results:
            if len(result.boxes) > 0:
                # Make a copy of the image to modify
                img_with_mosaic = img.copy()
                
                # Apply mosaic to each detected object
                for box in result.boxes.xyxy.cpu().numpy():
                    img_with_mosaic = apply_mosaic(img_with_mosaic, box, mosaic_size)
                
                # Save the processed image
                cv2.imwrite(output_path, img_with_mosaic)
                print(f"Processed and saved: {output_path}")
            else:
                print(f"No objects detected in {img_path}")
                # Save a copy of the original image
                cv2.imwrite(output_path, img)
    
    print(f"Processing complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()