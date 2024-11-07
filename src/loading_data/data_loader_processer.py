import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from dataclasses import dataclass
from typing import List
from tqdm import tqdm

@dataclass
class ImageStats:
    """Basic image statistics."""
    filename: str
    width: int
    height: int
    is_grayscale: bool
    text_coverage: float
    mean_intensity: float

class DataLoader:
    def __init__(self, data_dir: str, target_width: int = 800, target_height: int = 600):
        """Initialize data loader with paths and optional resize parameters."""
        self.data_dir = Path(data_dir)
        self.output_dir = Path("output/data_analysis_report")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.target_width = target_width
        self.target_height = target_height
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def analyze_images(self) -> List[ImageStats]:
        """Analyze all images in directory, including optional resizing."""
        image_stats = []
        image_files = list(self.data_dir.glob("*.png"))
        
        self.logger.info(f"Looking for images in: {self.data_dir}")
        self.logger.info(f"Found {len(image_files)} images")
        
        if not image_files:
            self.logger.warning("No PNG files found in the specified directory!")
            return []
        
        for img_path in tqdm(image_files, desc="Analyzing images"):
            try:
                # Read and analyze image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # Resize the image if the target dimensions are specified
                img_resized = cv2.resize(img, (self.target_width, self.target_height))
                
                # Get basic stats from the resized image
                height, width = img_resized.shape[:2]
                is_grayscale = len(img_resized.shape) == 2
                gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY) if not is_grayscale else img_resized
                
                # Calculate text coverage
                threshold = np.mean(gray) - np.std(gray)
                text_coverage = np.sum(gray < threshold) / gray.size * 100
                
                # Store stats
                image_stats.append(ImageStats(
                    filename=img_path.name,
                    width=width,
                    height=height,
                    is_grayscale=is_grayscale,
                    text_coverage=text_coverage,
                    mean_intensity=np.mean(gray)
                ))

            except Exception as e:
                self.logger.error(f"Error processing {img_path.name}: {e}")
                
        return image_stats

    def generate_report(self, stats: List[ImageStats]):
        """Generate analysis report."""
        # Convert to DataFrame for easy analysis
        df = pd.DataFrame([vars(s) for s in stats])
        
        # Generate and save report
        with open(self.output_dir / "analysis_report.txt", "w") as f:
            f.write("Image Dataset Analysis\n")
            f.write("===================\n\n")
            f.write(f"Total Images: {len(df)}\n")
            f.write(f"Average Dimensions: {df['width'].mean():.0f}x{df['height'].mean():.0f}\n")
            f.write(f"Grayscale Images: {df['is_grayscale'].sum()}\n")
            f.write(f"Average Text Coverage: {df['text_coverage'].mean():.1f}%\n")
            f.write(f"Average Intensity: {df['mean_intensity'].mean():.1f}\n")
        
        # Save detailed stats
        df.to_csv(self.output_dir / "detailed_stats.csv", index=False)

def main():
    try:
        # Setup paths
        current_dir = Path(__file__).parent.parent.parent  # Go up to project root
        data_dir = current_dir / "data" / "snippets"
        print(f"Current directory: {current_dir}")
        
        # Run analysis with resizing to 800x600 by default
        loader = DataLoader(data_dir, target_width=800, target_height=600)
        stats = loader.analyze_images()
        loader.generate_report(stats)
        
        print("Analysis complete! Check output/data_analysis_report/ for results")
        
    except Exception as e:
        logging.error(f"Analysis failed: {e}")

if __name__ == "__main__":
    main()
