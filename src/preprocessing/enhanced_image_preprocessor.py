import cv2
import numpy as np
from pathlib import Path
import random
import logging

class EnhancedImagePreprocessor:
    def __init__(self):
        # Initialize logging to track preprocessing progress and errors
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    # Define the preprocessing steps
    # Convert the image to grayscale
    def convert_to_grayscale(self, image):
        # Convert the image to grayscale
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Remove noise using Gaussian blur
    def remove_noise(self, image):
        # Apply Gaussian blur to remove noise
        return cv2.GaussianBlur(image, (5, 5), 0)

    # Threshold the image using adaptive thresholding
    def threshold_image(self, image):
        # Convert the image to binary using adaptive thresholding
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

    # Deskew the image if necessary
    def deskew_image(self, image):
        # Deskew the image if necessary (use Hough Transform or similar)
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    # Enhance the image contrast using CLAHE
    def enhance_contrast(self, image):
        # Enhance the image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    # Preprocess a set of sample images
    def preprocess_sample_images(self, input_dir, num_samples=5):
        # Select a limited number of images
        input_path = Path(input_dir)
        # List all the images in the input directory
        images = list(input_path.glob('*.png'))
        # Select a subset of images randomly
        selected_images = random.sample(images, min(num_samples, len(images)))
        
        output_dir = Path("output/enhanced_image_preprocessor_samples")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each selected image and save the intermediate stages
        for image_path in selected_images:
            # Read the image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Could not read image: {image_path}")
                continue

            # Process each image using the enhanced preprocessing steps
            gray = self.convert_to_grayscale(image)
            denoised = self.remove_noise(gray)
            thresholded = self.threshold_image(denoised)
            deskewed = self.deskew_image(thresholded)
            enhanced = self.enhance_contrast(deskewed)

            # Save each stage as a sample
            cv2.imwrite(str(output_dir / f"{image_path.stem}_original.png"), image)
            cv2.imwrite(str(output_dir / f"{image_path.stem}_grayscale.png"), gray)
            cv2.imwrite(str(output_dir / f"{image_path.stem}_denoised.png"), denoised)
            cv2.imwrite(str(output_dir / f"{image_path.stem}_thresholded.png"), thresholded)
            cv2.imwrite(str(output_dir / f"{image_path.stem}_deskewed.png"), deskewed)
            cv2.imwrite(str(output_dir / f"{image_path.stem}_enhanced.png"), enhanced)

        print(f"Sample images saved to {output_dir}")

# Usage
preprocessor = EnhancedImagePreprocessor()
preprocessor.preprocess_sample_images('data/snippets', num_samples=100)  
