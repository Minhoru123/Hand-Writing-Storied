import easyocr
import cv2
import logging
import numpy as np
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import pandas as pd

class OCRHandler:
    # Initialize EasyOCR
    def __init__(self, model_lang=['en']):
        self.logger = logging.getLogger(__name__)
        try:
            print("Initializing EasyOCR. This might take a moment...")
            self.reader = easyocr.Reader(model_lang)
            print("EasyOCR initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing EasyOCR: {str(e)}")
            raise
    
    # Preprocess image
    def preprocess_image(self, image) -> np.ndarray:
        """
        Preprocess image for OCR
        - Convert to grayscale
        """
        try:
            # Convert to grayscale if not already
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            return gray
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            return image

    # Process single image
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

            # Perform OCR
            results = self.reader.readtext(processed_image)

            # Extract text and confidence
            text_results = []
            confidence_scores = []
            for result in results:
                text_results.append(result[1])  # text
                confidence_scores.append(result[2])  # confidence

            return {
                'filename': Path(image_path).name,
                'text': ' '.join(text_results),
                'confidence': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
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
            
    # Process directory
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
        csv_path = output_path / 'easyocr_results.csv'
        df.to_csv(csv_path, index=False)
        
        # Generate summary report
        summary_path = output_path / 'easyocr_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("OCR Processing Summary\n")
            f.write("====================\n\n")
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
        # Initialize OCR Handler
        ocr = OCRHandler()
        
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