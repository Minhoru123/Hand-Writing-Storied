import cv2
import numpy as np
from pathlib import Path
import random

class SimplePreprocessor:
    def __init__(self):
        pass

    def convert_to_grayscale(self, image):
        # Convert RGB image to grayscale
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def enhance_contrast(self, image):
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(image)

    def preprocess_sample_images(self, input_dir, num_samples=5):
        # Select a limited number of images
        input_path = Path(input_dir)
        images = list(input_path.glob('*.png'))
        selected_images = random.sample(images, min(num_samples, len(images)))
        
        # Define the output path for the samples inside the output folder
        output_dir = Path("output/image_processor_samples")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for image_path in selected_images:
            # Process each selected image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Could not read image: {image_path}")
                continue

            # Apply preprocessing steps and save each stage
            gray = self.convert_to_grayscale(image)
            enhanced = self.enhance_contrast(gray)

            # Save original, grayscale, and enhanced images inside the 'output/image_processor_samples' folder
            cv2.imwrite(str(output_dir / f"{image_path.stem}_original.png"), image)
            cv2.imwrite(str(output_dir / f"{image_path.stem}_grayscale.png"), gray)
            cv2.imwrite(str(output_dir / f"{image_path.stem}_enhanced.png"), enhanced)
        
        print(f"Sample images saved to {output_dir}")

# Usage example
preprocessor = SimplePreprocessor()
preprocessor.preprocess_sample_images('data/snippets', num_samples=100)  # Adjust the directory as needed
