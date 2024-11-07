import easyocr
import pytesseract
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

class EnhancedOCRComparison:
    def __init__(self):
        # Initialize OCR engines
        self.easyocr_reader = easyocr.Reader(['en'])
        self.output_dir = Path('output/ocr_results/comparison')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_image(self, image_path):
        """Process single image with both engines and collect detailed metrics."""
        image = cv2.imread(str(image_path))
        
        # EasyOCR processing
        easy_start = time.time()
        easy_results = self.easyocr_reader.readtext(image) # Default language is English
        easy_time = time.time() - easy_start
        
        # Extract text and confidence
        easy_text = ' '.join([result[1] for result in easy_results]) if easy_results else ''
        easy_conf = sum([result[2] for result in easy_results])/len(easy_results) if easy_results else 0
        
        # Tesseract processing
        tess_start = time.time()
        tess_text = pytesseract.image_to_string(image) # Default language is English
        tess_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME)
        tess_time = time.time() - tess_start
        
        # Extract text and confidence
        tess_conf = tess_data['conf'].mean() / 100  # Normalize to 0-1 scale

        return {
            'filename': image_path.name,
            'easyocr_text': easy_text,
            'easyocr_conf': easy_conf,
            'easyocr_time': easy_time,
            'easyocr_detections': len(easy_results) if easy_results else 0,
            'tesseract_text': tess_text,
            'tesseract_conf': tess_conf,
            'tesseract_time': tess_time,
            'tesseract_detections': len(tess_data[tess_data['conf'] > 0]),
            'has_text': bool(easy_text.strip() or tess_text.strip())
        }

    def analyze_results(self, results_df):
        """Analyze results and make recommendation."""
        metrics = {
            'EasyOCR': {
                'avg_confidence': results_df['easyocr_conf'].mean(),
                'success_rate': (results_df['easyocr_text'].str.len() > 0).mean() * 100,
                'avg_time': results_df['easyocr_time'].mean(),
                'total_detections': results_df['easyocr_detections'].sum()
            },
            'Tesseract': {
                'avg_confidence': results_df['tesseract_conf'].mean(),
                'success_rate': (results_df['tesseract_text'].str.len() > 0).mean() * 100,
                'avg_time': results_df['tesseract_time'].mean(),
                'total_detections': results_df['tesseract_detections'].sum()
            }
        }
        
        # Calculate overall scores (weighted metrics)
        for engine in metrics:
            metrics[engine]['overall_score'] = (
                metrics[engine]['avg_confidence'] * 0.4 +  # Confidence importance
                metrics[engine]['success_rate'] * 0.4 +   # Success rate importance
                (1 / metrics[engine]['avg_time']) * 0.2   # Speed importance
            )
        
        return metrics

    def compare_engines(self, data_dir):
        """Run comparison and generate detailed report."""
        image_files = list(Path(data_dir).glob('*.png'))
        print(f"Found {len(image_files)} images to process")
        
        # Process images
        results = []
        for img_path in tqdm(image_files, desc="Processing images"):
            result = self.process_image(img_path)
            results.append(result)
        
        results_df = pd.DataFrame(results)
        metrics = self.analyze_results(results_df)
        
        # Save detailed results
        results_df.to_csv(self.output_dir / 'detailed_comparison.csv', index=False)
        
        # Generate visualizations
        self.create_comparison_plots(results_df, metrics)
        
        # Generate recommendation report
        self.generate_recommendation(metrics)
        
        return metrics

    def create_comparison_plots(self, df, metrics):
        """Create comprehensive comparison visualizations."""
        # Remove seaborn style, use default instead
        plt.style.use('default')
    
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
        # 1. Confidence Distribution
        axes[0,0].hist(df['easyocr_conf'], alpha=0.5, label='EasyOCR', 
                   bins=20, color='blue')
        axes[0,0].hist(df['tesseract_conf'], alpha=0.5, label='Tesseract', 
                   bins=20, color='red')
        axes[0,0].set_title('Confidence Score Distribution')
        axes[0,0].legend()
    
        # 2. Success Rates
        success_rates = [metrics[engine]['success_rate'] 
                    for engine in ['EasyOCR', 'Tesseract']]
        axes[0,1].bar(['EasyOCR', 'Tesseract'], success_rates, 
                  color=['blue', 'red'])
        axes[0,1].set_title('Success Rate (%)')
    
        # 3. Processing Times
        times = [metrics[engine]['avg_time'] 
            for engine in ['EasyOCR', 'Tesseract']]
        axes[1,0].bar(['EasyOCR', 'Tesseract'], times, 
                  color=['blue', 'red'])
        axes[1,0].set_title('Average Processing Time (s)')
    
        # 4. Overall Scores
        scores = [metrics[engine]['overall_score'] 
             for engine in ['EasyOCR', 'Tesseract']]
        axes[1,1].bar(['EasyOCR', 'Tesseract'], scores, 
                  color=['blue', 'red'])
        axes[1,1].set_title('Overall Performance Score')
    
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comparison_analysis.png')
        plt.close()

    def generate_recommendation(self, metrics):
        """Generate detailed recommendation based on analysis."""
        # Determine best engine
        easy_score = metrics['EasyOCR']['overall_score']
        tess_score = metrics['Tesseract']['overall_score']
        
        recommendation = "EasyOCR" if easy_score > tess_score else "Tesseract"
        
        report = f"""
OCR Engine Comparison Report
===========================

Detailed Metrics
---------------
EasyOCR:
- Average Confidence: {metrics['EasyOCR']['avg_confidence']:.3f}
- Success Rate: {metrics['EasyOCR']['success_rate']:.1f}%
- Processing Time: {metrics['EasyOCR']['avg_time']:.3f}s
- Total Detections: {metrics['EasyOCR']['total_detections']}
- Overall Score: {metrics['EasyOCR']['overall_score']:.3f}

Tesseract:
- Average Confidence: {metrics['Tesseract']['avg_confidence']:.3f}
- Success Rate: {metrics['Tesseract']['success_rate']:.1f}%
- Processing Time: {metrics['Tesseract']['avg_time']:.3f}s
- Total Detections: {metrics['Tesseract']['total_detections']}
- Overall Score: {metrics['Tesseract']['overall_score']:.3f}

Recommendation
-------------
Based on the analysis, {recommendation} is recommended for this project because:
1. {'Higher confidence scores' if metrics[recommendation]['avg_confidence'] > metrics['Tesseract' if recommendation == 'EasyOCR' else 'EasyOCR']['avg_confidence'] else 'Better processing speed'}
2. {'Better success rate' if metrics[recommendation]['success_rate'] > metrics['Tesseract' if recommendation == 'EasyOCR' else 'EasyOCR']['success_rate'] else 'More efficient processing'}
3. {'More text detections' if metrics[recommendation]['total_detections'] > metrics['Tesseract' if recommendation == 'EasyOCR' else 'EasyOCR']['total_detections'] else 'Better overall performance'}
"""
        
        # Save recommendation
        with open(self.output_dir / 'ocr_recommendation.txt', 'w') as f:
            f.write(report)
        
        print("\nComparison Summary:")
        print(f"Recommended Engine: {recommendation}")
        print(f"Confidence: {metrics[recommendation]['avg_confidence']:.3f}")
        print(f"Success Rate: {metrics[recommendation]['success_rate']:.1f}%")

def main():
    import time
    data_dir = Path('data/snippets')
    
    print("Starting OCR engine comparison...")
    comparator = EnhancedOCRComparison()
    comparator.compare_engines(data_dir)
    print("\nComparison complete! Check output/ocr_results/comparison/ for detailed analysis")

if __name__ == "__main__":
    main()
