import easyocr
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging


class HandwritingRecognition:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            print("Initializing EasyOCR for handwriting recognition...")
            self.reader = easyocr.Reader(['en'])
            print("Initialization complete")
        except Exception as e:
            self.logger.error(f"Error initializing OCR: {str(e)}")
            raise

    def preprocess_image(self, image):
        """
        Preprocess image for better handwriting recognition
        """
        try:
            # Check if image is empty
            if image is None:
                raise ValueError("Input image is None")
                
            # Handle different image formats
            if len(image.shape) == 2:
                gray = image
            elif len(image.shape) == 3:
                if image.shape[2] == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                elif image.shape[2] == 4:
                    rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                else:
                    raise ValueError(f"Unexpected number of channels: {image.shape[2]}")
            else:
                raise ValueError(f"Unexpected image shape: {image.shape}")
            
            # Resize to standard height while maintaining aspect ratio
            target_height = 64
            aspect_ratio = gray.shape[1] / gray.shape[0]
            target_width = int(target_height * aspect_ratio)
            resized = cv2.resize(gray, (target_width, target_height), interpolation=cv2.INTER_AREA)
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(resized)
            
            # Remove dotted lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            detect_horizontal = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            for c in cnts:
                cv2.drawContours(enhanced, [c], -1, (255, 255, 255), 2)
            
            # Binarization
            _, binary = cv2.threshold(enhanced, 160, 255, cv2.THRESH_BINARY_INV)
            
            # Clean up noise
            kernel = np.ones((2, 2), np.uint8)
            clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Final enhancements
            processed = cv2.dilate(clean, kernel, iterations=1)
            processed = cv2.copyMakeBorder(processed, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=0)
            
            return processed

        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            return image

    def process_snippet(self, image_path):
        """
        Process a single image snippet and return recognition results
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                return '', 0.0

            # Preprocess
            processed_image = self.preprocess_image(image)

            # Perform OCR
            results = self.reader.readtext(
                processed_image,
                paragraph=False,
                batch_size=1,
                contrast_ths=0.2,
                text_threshold=0.6,
                low_text=0.3,
                width_ths=0.8,
                height_ths=0.8,
                add_margin=0.3
            )

            # Process results
            if results:
                # Get the highest confidence result
                best_result = max(results, key=lambda x: x[2])
                text = best_result[1].strip()
                confidence = best_result[2]
                return text, confidence
            else:
                return '', 0.0

        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {str(e)}")
            return '', 0.0

    def process_snippets(self, input_dir, output_file):
        """
        Process all snippets and create the final CSV
        """
        input_path = Path(input_dir)
        
        # Get all PNG files, excluding debug images and any other temporary files
        snippets = [
            f for f in input_path.glob('*.png') 
            if not any(x in f.name.lower() for x in ['debug', 'temp', 'processed'])
        ]

        # Sort snippets for consistent processing
        snippets.sort()

        # Keep track of processed files to avoid duplicates
        processed_files = set()
        results = []

        print(f"Processing {len(snippets)} original snippets...")
        for snippet_path in tqdm(snippets):
            # Skip if we've already processed this file
            if snippet_path.name in processed_files:
                continue

            text, confidence = self.process_snippet(snippet_path)
            
            # Add to results and mark as processed
            results.append({
                'snippet_name': snippet_path.name,
                'label': text if text else '',
                'confidence_score': round(confidence, 3) if confidence > 0 else ''
            })
            processed_files.add(snippet_path.name)

        # Create and save DataFrame
        df = pd.DataFrame(results)
        
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Save results
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
        return df


def cleanup_directory(directory):
    """
    Clean up any debug or temporary files
    """
    dir_path = Path(directory)
    for pattern in ['debug_*', 'temp_*', 'processed_*']:
        for file in dir_path.glob(pattern):
            try:
                file.unlink()
                print(f"Cleaned up: {file}")
            except Exception as e:
                print(f"Could not delete {file}: {e}")


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        # Clean up any existing debug files first
        input_dir = 'data/snippets'
        cleanup_directory(input_dir)
        
        # Initialize recognizer
        recognizer = HandwritingRecognition()
        
        # Set up paths
        output_file = 'results/handwriting_recognition_results.csv'
        
        # Process all snippets
        results = recognizer.process_snippets(input_dir, output_file)
        
        # Print summary
        total_snippets = len(results)
        recognized = len(results[results['label'].str.len() > 0])
        print(f"\nProcessing Summary:")
        print(f"Total snippets processed: {total_snippets}")
        print(f"Snippets with recognized text: {recognized}")
        print(f"Empty snippets: {total_snippets - recognized}")
        
        # Print first few results as sample
        print("\nSample Results (first 5):")
        print(results.head().to_string())
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
