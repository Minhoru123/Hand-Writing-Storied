import logging
import os
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from dataclasses import dataclass
import cv2 as cv
from typing import List
from tqdm import tqdm
import pandas as pd

# Updated dataclass to store the image and its label
@dataclass
class Image:
    filename: str
    width: int
    height: int
    aspect_ratio: float
    file_size: int
    is_grayscale: bool
    mean_intensity: float
    std_intensity: float
    min_intensity: float
    max_intensity: float
    empty_percent: float
    text_area_percent: float    # New field for OCR preparation

class DataLoader:
    # Constructor to initialize the DataLoader class
    def __init__(self, data_dir: str, output_dir: str = "Output"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        # Add directory verification
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir.absolute()}\n"
                f"Current working directory: {Path.cwd()}"
            )
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.supported_formats = ('.png')
        self.image_files = []
        self.image_stats = []
        self.total_images = 0
        
        #Error logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    # Function to scan the data directory and return the number of images in the dataset    
    def scan_data(self) -> Dict[str, int]:
        stats = {
            'total_files': 0,
            'valid_images': 0,
            'invalid_images': 0
        }
    
        try:
            # Print directory being checked
            print(f"Checking directory: {self.data_dir}")
        
            # Get all PNG files
            self.image_files = list(self.data_dir.glob('*.png'))
            stats['total_files'] = len(self.image_files)
        
            # Basic validation of each image
            for img_path in self.image_files:
                try:
                    img = cv.imread(str(img_path))
                    if img is not None:
                        stats['valid_images'] += 1
                    else:
                        stats['invalid_images'] += 1
                except Exception as e:
                    stats['invalid_images'] += 1
                
            # Log results
            if stats['total_files'] == 0:
                print("No PNG files found!")
            else:
                print(f"Found {stats['valid_images']} valid PNG files")
            
        except Exception as e:
            print(f"Error scanning directory: {e}")
        
        return stats
    
    # function to load a single image from the dataset with error handling
    def load_image(self, image_file: Path) -> Optional[np.ndarray]:
        try:
            # Load the image using PIL
            img = cv.imread(str(image_file))
            if img is None:
                self.logger.error(f"Error loading image: {image_file}")
                return None
            return img
        except Exception as e:
            self.logger.error(f"Error loading image: {str(e)}")
            return None
    
    # Function to check images stats
    def compute_image_stats(self, img: np.ndarray, filename: str) -> Image:
        # Compute the image statistics
        height, width, _ = img.shape
        is_grayscale = len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1)
        
        # Convert to gray for intensity analysis
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if not is_grayscale else img
        
        # Calculate emptiness
        empty_percent = np.sum(gray > 250) / gray.size * 100
        
        # Calculate text area (new)
        text_threshold = np.mean(gray) - np.std(gray)
        text_area_percent = np.sum(gray < text_threshold) / gray.size * 100
        
        return Image(
            filename=filename,
            width=width,
            height=height,
            aspect_ratio=width/height,
            file_size=os.path.getsize(self.data_dir / filename),
            is_grayscale=is_grayscale,
            mean_intensity=np.mean(gray),
            std_intensity=np.std(gray),
            min_intensity=np.min(gray),
            max_intensity=np.max(gray),
            empty_percent=empty_percent,
            text_area_percent=text_area_percent  # New field
        )
    
    # Function to process the images in the dataset
    def analyze_images(self) -> List[Image]:
        self.image_stats = []
        failed_images = []
        
        for image_file in tqdm(self.image_files):
            img = self.load_image(image_file)
            if img is not None:
                try:
                    stats = self.compute_image_stats(img, image_file.name)
                    self.image_stats.append(stats)
                except Exception as e:
                    self.logger.error(f"Error processing image: {str(e)}")
                    failed_images.append(image_file.name)
            else:
                failed_images.append(image_file.name)
                
        if failed_images:
            self.logger.error(f"Failed to load {len(failed_images)} images: {failed_images}")
        
        return self.image_stats
    
    # Function to generate the report
    def generate_report(self):
        if not self.image_stats:
            self.logger.error("No statistics available for report generation")
            return
            
        df = pd.DataFrame([vars(s) for s in self.image_stats])
        
        report = {
            # Basic Dataset Statistics
            'Dataset Summary': {
                'total_images': len(self.image_stats),
                'grayscale_images': sum(s.is_grayscale for s in self.image_stats),
                'color_images': sum(not s.is_grayscale for s in self.image_stats),
            },
            
            # Dimension Analysis
            'Dimension Statistics': {
                'mean_width': df['width'].mean(),
                'width_range': f"{df['width'].min():.0f}-{df['width'].max():.0f}",
                'mean_height': df['height'].mean(),
                'height_range': f"{df['height'].min():.0f}-{df['height'].max():.0f}",
                'mean_aspect_ratio': df['aspect_ratio'].mean(),
                'aspect_ratio_range': f"{df['aspect_ratio'].min():.2f}-{df['aspect_ratio'].max():.2f}",
            },
            
            # Image Quality Metrics
            'Image Quality': {
                'mean_file_size_kb': df['file_size'].mean() / 1024,
                'mean_intensity': df['mean_intensity'].mean(),
                'std_intensity': df['std_intensity'].mean(),
                'average_empty_space': df['empty_percent'].mean(),
                'average_text_coverage': df['text_area_percent'].mean(),  # New metric
            }
        }
        
        # Save report with sections
        with open(self.output_dir / 'analysis_report.txt', 'w') as f:
            f.write("Image Dataset Analysis Report\n")
            f.write("===========================\n\n")
            
            # Write each section
            for section_name, metrics in report.items():
                f.write(f"{section_name}\n")
                f.write("-" * len(section_name) + "\n")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.2f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
                f.write("\n")
            
            # Add key findings
            f.write("Key Findings\n")
            f.write("-----------\n")
            f.write(f"- Dataset contains {report['Dataset Summary']['total_images']} images\n")
            f.write(f"- Images are primarily {'grayscale' if report['Dataset Summary']['grayscale_images'] > report['Dataset Summary']['color_images'] else 'color'}\n")
            f.write(f"- Average image dimensions: {report['Dimension Statistics']['mean_width']:.0f}x{report['Dimension Statistics']['mean_height']:.0f} pixels\n")
            f.write(f"- Mean intensity: {report['Image Quality']['mean_intensity']:.2f} (0-255 scale)\n")
            f.write(f"- Average empty space: {report['Image Quality']['average_empty_space']:.2f}%\n")
            f.write(f"- Average text coverage: {report['Image Quality']['average_text_coverage']:.2f}%\n")
        
        # Save detailed stats to CSV
        df.to_csv(self.output_dir / 'detailed_stats.csv', index=False)
        
        return report
    
    # Function to generate OCR readiness report
    def generate_ocr_readiness_report(self):
        try:
            if not self.image_stats:
                self.logger.error("No statistics available for report generation")
                return
            
            print("Starting OCR readiness report generation...")
        
            df = pd.DataFrame([vars(s) for s in self.image_stats])
        
            # Create the report path
            report_path = self.output_dir / 'ocr_preparation_report.txt'
            print(f"Attempting to write report to: {report_path}")
        
            # Save simple report
            with open(report_path, 'w') as f:
                f.write("OCR Preparation Analysis\n")
                f.write("=======================\n\n")
                f.write(f"Total Images Analyzed: {len(df)}\n")
                f.write(f"Image Dimensions: {df['width'].mean():.0f}x{df['height'].mean():.0f} pixels\n")
                f.write(f"Average Text Coverage: {df['text_area_percent'].mean():.1f}%\n")
                f.write(f"Grayscale Images: {sum(df['is_grayscale'])}/{len(df)}\n")
            
                # Add a simple recommendation
                f.write("\nRecommendations for OCR:\n")
                if df['text_area_percent'].mean() < 10:
                    f.write("- Consider image enhancement as text coverage is low\n")
                if not all(df['is_grayscale']):
                    f.write("- Convert all images to grayscale before OCR\n")
        
            print("OCR readiness report generated successfully")
        
        except Exception as e:
            self.logger.error(f"Error generating OCR readiness report: {str(e)}")
            print(f"Error generating OCR readiness report: {str(e)}")
            raise
        
# Main execution function
def main():
    try:
        # Initialize with explicit paths
        current_dir = Path(__file__).parent.parent  # Go up one level from src
        data_dir = current_dir / 'data' / 'snippets'
        output_dir = current_dir / 'Output'
        
        print(f"Working directory: {Path.cwd()}")
        print(f"Looking for images in: {data_dir.absolute()}")
        
        data_loader = DataLoader(
            data_dir=str(data_dir),
            output_dir=str(output_dir)
        )
        
        # Scan the data directory
        data_stats = data_loader.scan_data()
        print("File scanning complete:", data_stats)
        
        # Analyze the images
        image_stats = data_loader.analyze_images()
        print(f"\nAnalyzed {len(image_stats)} images successfully")
        
        # Generate reports
        print("\nGenerating reports...")
        
        data_loader.generate_report()
        print("General report generated")
        
        data_loader.generate_ocr_readiness_report()
        print("OCR readiness report generated")
        
        # Verify files exist
        output_dir = Path(output_dir)
        report_files = {
            'analysis_report.txt': output_dir / 'analysis_report.txt',
            'ocr_preparation_report.txt': output_dir / 'ocr_preparation_report.txt',
            'detailed_stats.csv': output_dir / 'detailed_stats.csv'
        }
        
        print("\nChecking generated files:")
        for name, path in report_files.items():
            if path.exists():
                print(f"✓ {name} - Generated successfully")
            else:
                print(f"✗ {name} - Not found")
                
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise
if __name__ == '__main__':
    main()