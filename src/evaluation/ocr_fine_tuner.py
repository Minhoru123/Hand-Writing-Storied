import easyocr
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import pandas as pd
import time


class EnhancedOCRTuner:

    def __init__(self, input_dir, output_dir):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.input_dir = Path(r"C:\Users\mijay\OneDrive\Desktop\AIProjects\Hand-Writing-Storied\data\snippets")  # Update the input directory path
        
        # Update the output directory path to point to 'ocr_results' inside the existing 'output' folder
        self.output_dir = Path('output/ocr_results/ocr_fine_tuning')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Expanded configurations
        self.readers = {
            'aggressive_recall': {
                'reader': easyocr.Reader(['en']),
                'params': {
                    'paragraph': False,
                    'min_size': 5,  # Reduced to catch smaller text
                    'contrast_ths': 0.05,  # More aggressive contrast threshold
                    'adjust_contrast': 1.5,  # Increased contrast adjustment
                    'text_threshold': 0.3,  # Lower text confidence threshold
                    'link_threshold': 0.2,  # Lower linking threshold
                    'mag_ratio': 2,  # Increase image size for detection
                }
            },
            'balanced': {
                'reader': easyocr.Reader(['en']),
                'params': {
                    'paragraph': False,
                    'min_size': 8,
                    'contrast_ths': 0.1,
                    'adjust_contrast': 1.2,
                    'text_threshold': 0.4,
                    'link_threshold': 0.3,
                    'mag_ratio': 1.5,
                }
            },
            'handwriting_focused': {
                'reader': easyocr.Reader(['en']),
                'params': {
                    'paragraph': False,
                    'min_size': 10,
                    'contrast_ths': 0.15,
                    'adjust_contrast': 1.3,
                    'text_threshold': 0.35,
                    'link_threshold': 0.25,
                    'mag_ratio': 1.8,
                    'slope_ths': 0.5,  # More tolerant to slanted text
                }
            }
        }

    def preprocess_for_ocr(self, image):
        """Additional preprocessing specific to OCR."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced)

        return denoised

    def process_image(self, image_path, reader_config):
        try:
            # Read and preprocess image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")

            # Apply preprocessing
            processed_image = self.preprocess_for_ocr(image)

            # Get OCR results
            reader = reader_config['reader']
            params = reader_config['params']
            
            start_time = time.time()
            results = reader.readtext(processed_image, **params)
            processing_time = time.time() - start_time

            # Extract text and confidence
            if results:
                text = ' '.join([result[1] for result in results])
                confidence = np.mean([result[2] for result in results])
                max_conf = max([result[2] for result in results]) if results else 0
            else:
                text = ''
                confidence = 0.0
                max_conf = 0.0

            return {
                'filename': image_path.name,
                'text': text,
                'avg_confidence': confidence,
                'max_confidence': max_conf,
                'processing_time': processing_time,
                'num_detections': len(results),
                'configuration': reader_config  # Ensure that the configuration is included here
            }

        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {e}")
            return {
                'filename': image_path.name,
                'text': '',
                'avg_confidence': 0.0,
                'max_confidence': 0.0,
                'processing_time': 0.0,
                'num_detections': 0,
                'error': str(e),
                'configuration': None  # Add configuration to the error result as well
            }

    def write_results_to_txt(self, file_path, results_df, summary):
        with open(file_path, 'w') as file:
            file.write("Enhanced OCR Configuration Summary\n")
            file.write("=" * 50 + "\n")

            for config in summary.index:
                file.write(f"\nConfiguration: {config}\n")
                file.write(f"Average Confidence: {summary.loc[config, ('avg_confidence', 'mean')]:.3f}\n")
                file.write(f"Max Confidence: {summary.loc[config, ('max_confidence', 'max')]:.3f}\n")
                file.write(f"Average Processing Time: {summary.loc[config, ('processing_time', 'mean')]:.3f}s\n")
                file.write(f"Average Detections: {summary.loc[config, ('num_detections', 'mean')]:.1f}\n")
                file.write(f"Max Detections: {summary.loc[config, ('num_detections', 'max')]:.0f}\n")
                file.write("-" * 50 + "\n")

            file.write("\nOCR Results Details:\n")
            file.write("=" * 50 + "\n")
            for _, row in results_df.iterrows():
                file.write(f"\nFile: {row['filename']}\n")
                file.write(f"Text Detected: {row['text']}\n")
                file.write(f"Avg Confidence: {row['avg_confidence']:.3f}\n")
                file.write(f"Max Confidence: {row['max_confidence']:.3f}\n")
                file.write(f"Processing Time: {row['processing_time']:.3f}s\n")
                file.write(f"Number of Detections: {row['num_detections']}\n")
                file.write("-" * 50 + "\n")
    
    def evaluate_configurations(self, test_size=20):  # Increased test size
        image_files = list(self.input_dir.glob('*.png'))
        if not image_files:
            self.logger.error("No images found in the input directory.")
            return None, None
    
        test_images = image_files[:test_size]
        self.logger.info(f"Testing on {len(test_images)} images.")
    
        results = []
    
        for config_name, reader_config in self.readers.items():
            self.logger.info(f"\nTesting configuration: {config_name}")
    
            for img_path in tqdm(test_images, desc=f"Processing with {config_name}"):
                self.logger.debug(f"Processing image: {img_path}")
                result = self.process_image(img_path, reader_config)
            
                if result['text'] == '':
                    self.logger.warning(f"No text detected in {img_path}.")
                
                result['configuration'] = config_name
                results.append(result)
    
        if not results:
            self.logger.error("No results obtained from the evaluation.")
            return None, None

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
    
        # Ensure configuration column exists
        if 'configuration' not in results_df.columns:
            self.logger.error("Configuration column is missing in the results DataFrame.")
            return None, None

        # Save results to CSV
        results_df.to_csv(self.output_dir / 'enhanced_ocr_tuning_results.csv', index=False)
    
        # Summary of results
        summary = results_df.groupby('configuration').agg({
            'avg_confidence': ['mean', 'std'],
            'max_confidence': ['mean', 'max'],
            'processing_time': 'mean',
            'num_detections': ['mean', 'max']
        }).round(3)
    
        summary.to_csv(self.output_dir / 'enhanced_ocr_tuning_summary.csv')
    
        # Write to the text file for easier access to results
        txt_file_path = self.output_dir / 'ocr_tuning_results.txt'
        self.write_results_to_txt(txt_file_path, results_df, summary)

        print("\nEnhanced OCR Configuration Summary:")
        print("=" * 50)
        for config in summary.index:
            print(f"\nConfiguration: {config}")
            print(f"Average Confidence: {summary.loc[config, ('avg_confidence', 'mean')]:.3f}")
            print(f"Max Confidence: {summary.loc[config, ('max_confidence', 'max')]:.3f}")
            print(f"Average Processing Time: {summary.loc[config, ('processing_time', 'mean')]:.3f}s")
            print(f"Average Detections: {summary.loc[config, ('num_detections', 'mean')]:.1f}")
            print(f"Max Detections: {summary.loc[config, ('num_detections', 'max')]:.0f}")
    
        return results_df, summary


def main():
    current_dir = Path(__file__).parent.parent
    input_dir = current_dir / 'output' / 'preprocessed'  # Keep this as is
    output_dir = current_dir / 'output'  # Point to the 'output' folder
    
    tuner = EnhancedOCRTuner(input_dir, output_dir)
    results_df, summary = tuner.evaluate_configurations()

    if results_df is not None:
        print("Results DataFrame created successfully.")
    else:
        print("Error: Results DataFrame was not created.")

if __name__ == "__main__":
    main()
