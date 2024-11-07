import pytesseract
import cv2
import logging
import numpy as np
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import pandas as pd

class TesseractHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            # Test if Tesseract is installed
            pytesseract.get_tesseract_version()
            print("Tesseract initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing Tesseract: {str(e)}")
            print("Please ensure Tesseract is installed and added to PATH")
            print("Installation guide: https://github.com/UB-Mannheim/tesseract/wiki")
            raise

    def preprocess_image(self, image) -> np.ndarray:
        try:
            # Convert to grayscale if not already
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Apply thresholding to handle variations in text intensity
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            return binary
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            return image

    def process_single_image(self, image_path: str) -> Dict:
        """
        Process a single image and return OCR results
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")

            # Preprocess
            processed_image = self.preprocess_image(image)

            # Configure Tesseract parameters
            custom_config = r'--oem 3 --psm 6'  # Assume uniform text block
            
            # Perform OCR
            text = pytesseract.image_to_string(processed_image, config=custom_config)
            
            # Get confidence score
            data = pytesseract.image_to_data(processed_image, config=custom_config, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if conf != '-1']
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            return {
                'filename': Path(image_path).name,
                'text': text.strip(),
                'confidence': avg_confidence / 100,  # Convert to 0-1 scale to match EasyOCR
                'success': True
            }

        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {str(e)}")
            return {
                'filename': Path(image_path).name,
                'text': '',
                'confidence': 0,
                'success': False,
                'error': str(e)
            }

    def process_directory(self, input_dir: str, output_dir: str) -> pd.DataFrame:
        """
        Process all images in a directory and save results to CSV
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get all PNG files
        image_files = list(input_path.glob('*.png'))
        results = []

        print(f"Processing {len(image_files)} images...")
        for image_file in tqdm(image_files):
            result = self.process_single_image(str(image_file))
            results.append(result)

        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        csv_path = output_path / 'tesseract_results.csv'
        df.to_csv(csv_path, index=False)
        
        # Generate summary report
        summary_path = output_path / 'tesseract_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("Tesseract OCR Processing Summary\n")
            f.write("==============================\n\n")
            f.write(f"Total images processed: {len(results)}\n")
            f.write(f"Successful: {sum(1 for r in results if r['success'])}\n")
            f.write(f"Failed: {sum(1 for r in results if not r['success'])}\n")
            f.write(f"Average confidence: {df['confidence'].mean():.2f}\n")
            
            # Add some example results
            f.write("\nSample Results (first 5 successful):\n")
            successful = df[df['success']].head()
            for _, row in successful.iterrows():
                f.write(f"\nFile: {row['filename']}\n")
                f.write(f"Text: {row['text'][:100]}{'...' if len(row['text']) > 100 else ''}\n")
                f.write(f"Confidence: {row['confidence']:.2f}\n")

        print(f"\nResults saved to {csv_path}")
        print(f"Summary saved to {summary_path}")
        
        return df

def main():
    try:
        # Initialize Tesseract Handler
        ocr = TesseractHandler()
        
        # Process directory
        current_dir = Path(__file__).parent.parent
        input_dir = current_dir / 'data' / 'snippets'
        output_dir = current_dir / 'output'
        
        results = ocr.process_directory(str(input_dir), str(output_dir))
        
        # Print quick summary
        print("\nQuick Summary:")
        print(f"Total images processed: {len(results)}")
        print(f"Average confidence score: {results['confidence'].mean():.2f}")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()