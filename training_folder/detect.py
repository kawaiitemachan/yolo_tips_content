import os
import cv2
import glob
import numpy as np
from ultralytics import YOLO

def apply_mosaic_to_mask(image, mask, mosaic_size):
    """Apply mosaic effect to areas defined by the segmentation mask."""
    # Create a copy of the image to modify
    img_with_mosaic = image.copy()

    # Find contours in the mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a binary mask for all contours
    full_mask = np.zeros_like(mask)
    cv2.drawContours(full_mask, contours, -1, 1, -1)

    # Get the coordinates where the mask is True
    y_coords, x_coords = np.where(full_mask > 0)

    if len(y_coords) > 0 and len(x_coords) > 0:
        # Find the bounding box of the mask
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        min_x, max_x = np.min(x_coords), np.max(x_coords)

        # Extract the region to apply mosaic
        region = image[min_y:max_y+1, min_x:max_x+1].copy()

        # Get the mask for this region
        region_mask = full_mask[min_y:max_y+1, min_x:max_x+1]

        if region.size > 0 and region_mask.size > 0:
            # Calculate dimensions
            height, width = region.shape[:2]

            # Calculate mosaic block sizes
            mosaic_h = max(1, int(height / mosaic_size))
            mosaic_w = max(1, int(width / mosaic_size))

            # Resize down, then resize up to create mosaic effect
            small = cv2.resize(region, (mosaic_w, mosaic_h), interpolation=cv2.INTER_LINEAR)
            mosaic = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)

            # Create a masked version of the mosaic region
            masked_mosaic = np.zeros_like(region)
            for c in range(0, 3):  # For each color channel
                masked_mosaic[:, :, c] = np.where(region_mask > 0, mosaic[:, :, c], region[:, :, c])

            # Place mosaic region back into the image
            img_with_mosaic[min_y:max_y+1, min_x:max_x+1] = masked_mosaic

    return img_with_mosaic

def main():
    # Get model path
    model_path = os.path.join("runs", "segment", "yolov11n_seg_custom_model", "weights", "best.pt")

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        alternative_path = input("Enter alternative model path (or press enter to exit): ")
        if alternative_path.strip():
            model_path = alternative_path
        else:
            print("Exiting program.")
            return

    # Load YOLO model with automatic device selection
    try:
        # Try to use GPU first
        import torch
        if torch.cuda.is_available():
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            print("Using GPU for inference...")
            model = YOLO(model_path).to('cuda:0')
        else:
            print("No GPU detected. Using CPU for inference...")
            model = YOLO(model_path).to('cpu')
    except Exception as e:
        print(f"Error setting device: {e}")
        print("Falling back to CPU...")
        model = YOLO(model_path).to('cpu')

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
    output_dir = os.path.join("runs", "segment", "mosaic_results")
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

        # Run segmentation
        results = model(img, conf=confidence)

        # Process results and apply mosaic
        processed = False
        for result in results:
            # Check if there are any masks
            if hasattr(result, 'masks') and result.masks is not None and len(result.masks) > 0:
                # Make a copy of the image to modify
                img_with_mosaic = img.copy()

                # Apply mosaic to each segmentation mask
                for mask in result.masks.data.cpu().numpy():
                    # Convert mask from [0,1] to binary
                    binary_mask = (mask > 0.5).astype(np.uint8)

                    # Apply mosaic using the mask
                    img_with_mosaic = apply_mosaic_to_mask(img_with_mosaic, binary_mask, mosaic_size)

                # Save the processed image
                cv2.imwrite(output_path, img_with_mosaic)
                print(f"Processed and saved: {output_path}")
                processed = True

        if not processed:
            print(f"No objects detected in {img_path}")
            # Save a copy of the original image
            cv2.imwrite(output_path, img)

    print(f"Processing complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()