import pytesseract
import cv2
import logging
import numpy as np
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import pandas as pd
import os


class TesseractHandler:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            print("Testing Tesseract installation...")
            pytesseract.get_tesseract_version()
            print("Tesseract initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing Tesseract: {str(e)}")
            raise
    
    def preprocess_image(self, image) -> np.ndarray:
        """
        Enhanced preprocessing pipeline with explicit RGB/grayscale handling
        """
        try:
            # Check if image is empty
            if image is None:
                raise ValueError("Input image is None")
                
            # Get number of channels
            if len(image.shape) == 2:
                # Image is already grayscale
                gray = image
            elif len(image.shape) == 3:
                if image.shape[2] == 3:
                    # Convert BGR to grayscale
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                elif image.shape[2] == 4:
                    # Handle RGBA images
                    # First convert RGBA to RGB, then to grayscale
                    rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                else:
                    raise ValueError(f"Unexpected number of channels: {image.shape[2]}")
            else:
                raise ValueError(f"Unexpected image shape: {image.shape}")

            # Print debug info about image
            print(f"Original image shape: {image.shape}")
            print(f"Grayscale image shape: {gray.shape}")
            
            # Resize to standard height while maintaining aspect ratio
            target_height = 64  # Standard height for document processing
            aspect_ratio = gray.shape[1] / gray.shape[0]
            target_width = int(target_height * aspect_ratio)
            resized = cv2.resize(gray, (target_width, target_height), interpolation=cv2.INTER_AREA)
            
            # Aggressive contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(resized)
            
            # Remove dotted lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            detect_horizontal = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            for c in cnts:
                cv2.drawContours(enhanced, [c], -1, (255, 255, 255), 2)
                
            # Thresholding with high contrast
            _, binary = cv2.threshold(enhanced, 160, 255, cv2.THRESH_BINARY_INV)
            
            # Remove small noise
            kernel = np.ones((2, 2), np.uint8)
            clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Thicken signature strokes (more dilation for Tesseract)
            dilated = cv2.dilate(clean, kernel, iterations=2)
            
            # Add padding (more padding for Tesseract)
            padded = cv2.copyMakeBorder(dilated, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=0)
            
            print(f"Final processed image shape: {padded.shape}")
            return padded

        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            return image

    def process_single_image(self, image_path: str) -> Dict:
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")

            # Preprocess
            processed_image = self.preprocess_image(image)
            
            # Configure Tesseract for handwriting
            custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" -c textord_heavy_nr=1 -c textord_min_linesize=3'

            # Get confidence data
            data = pytesseract.image_to_data(processed_image, config=custom_config, output_type=pytesseract.Output.DICT)
            
            # Get text
            text = pytesseract.image_to_string(processed_image, config=custom_config).strip()
            
            # Calculate confidence
            confidences = [conf for conf in data['conf'] if conf > 0]
            confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Convert confidence to 0-1 scale
            confidence = confidence / 100
            
            # Only save debug image if confidence is above threshold
            debug_path = None
            if confidence > 0.1:  # You can adjust this threshold
                debug_path = Path(image_path).parent / f"debug_tesseract_{Path(image_path).name}"
                cv2.imwrite(str(debug_path), processed_image)

            result_dict = {
                'filename': Path(image_path).name,
                'text': text,
                'confidence': confidence,
                'success': bool(text and confidence > 0.1),
                'processed_path': str(debug_path) if debug_path else None
            }

            print(f"Final result: {result_dict}")
            return result_dict

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return {
                'filename': Path(image_path).name,
                'text': '',
                'confidence': 0,
                'success': False,
                'error': str(e)
            }
    
    def clean_debug_files(self, directory: Path):
        """Clean up existing debug files in the directory."""
        print("Cleaning up existing debug files...")
        count = 0
        for debug_file in directory.glob("debug_tesseract_*.png"):
            try:
                debug_file.unlink()
                count += 1
            except Exception as e:
                print(f"Error deleting {debug_file}: {e}")
        print(f"Cleaned up {count} debug files")
    
    def process_directory(self, input_dir: str, output_dir: str, clean_existing: bool=True) -> pd.DataFrame:
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Clean existing debug files if requested
        if clean_existing:
            self.clean_debug_files(input_path)

        # Get all PNG files (excluding debug files)
        image_files = [f for f in input_path.glob('*.png') 
                      if not f.name.startswith('debug_')]
        results = []

        print(f"Processing {len(image_files)} images with Tesseract...")
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
            f.write("OCR Processing Summary (Tesseract)\n")
            f.write("==============================\n\n")
            f.write(f"Total images processed: {len(results)}\n")
            f.write(f"Successful: {sum(1 for r in results if r['success'])}\n")
            f.write(f"Failed: {sum(1 for r in results if not r['success'])}\n")
            
            if 'confidence' in df.columns and not df.empty:
                f.write(f"Average confidence: {df['confidence'].mean():.3f}\n")
            
            # Add example results
            if not df.empty:
                successful = df[df['success']].sort_values('confidence', ascending=False).head()
                f.write("\nTop 5 Results by Confidence:\n")
                for _, row in successful.iterrows():
                    f.write(f"\nFile: {row['filename']}\n")
                    f.write(f"Text: {row['text']}\n")
                    f.write(f"Confidence: {row['confidence']:.3f}\n")

        print(f"\nResults saved to {csv_path}")
        print(f"Summary saved to {summary_path}")
        
        return df


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        # Initialize OCR Handler
        ocr = TesseractHandler()
        
        # Set up input/output directories
        script_dir = Path(__file__).parent.parent.parent.resolve()
        input_dir = script_dir / 'data' / 'snippets'
        output_dir = script_dir / 'output'
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify input directory exists
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        print(f"Processing images from: {input_dir}")
        print(f"Saving results to: {output_dir}")
        
        # Process all images
        results = ocr.process_directory(str(input_dir), str(output_dir))
        
        # Print final summary
        if not results.empty:
            print("\nProcessing Complete!")
            print(f"Total images processed: {len(results)}")
            print(f"Successful: {sum(results['success'])}")
            print(f"Failed: {len(results) - sum(results['success'])}")
            if 'confidence' in results.columns:
                print(f"Average confidence: {results['confidence'].mean():.3f}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
