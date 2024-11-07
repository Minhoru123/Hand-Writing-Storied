import pandas as pd
from pathlib import Path
import logging
from datetime import datetime


def compare_ocr_results(easyocr_csv: str, tesseract_csv: str, output_path: str):
    """
    Compare OCR results from EasyOCR and Tesseract, ignoring debug files
    """
    # Load results
    easyocr_df = pd.read_csv(easyocr_csv)
    tesseract_df = pd.read_csv(tesseract_csv)
    
    # Standardize column names if needed
    easyocr_df = easyocr_df.rename(columns={
        'text': 'label',
        'confidence': 'confidence_score'
    })
    tesseract_df = tesseract_df.rename(columns={
        'text': 'label',
        'confidence': 'confidence_score'
    })
    
    # Filter out debug files from analysis
    easyocr_df = easyocr_df[~easyocr_df['filename'].str.contains('debug', case=False)]
    tesseract_df = tesseract_df[~tesseract_df['filename'].str.contains('debug', case=False)]
    
    # Calculate metrics for each implementation
    metrics = {
        'EasyOCR': {
            'total_processed': len(easyocr_df),
            'successful_recognitions': sum(easyocr_df['label'].str.len() > 0),
            'empty_results': sum(easyocr_df['label'].str.len() == 0),
            'avg_confidence': easyocr_df[easyocr_df['confidence_score'] > 0]['confidence_score'].mean(),
            'recognition_rate': (sum(easyocr_df['label'].str.len() > 0) / len(easyocr_df) * 100),
            'high_confidence_results': sum(easyocr_df['confidence_score'] > 0.7)
        },
        'Tesseract': {
            'total_processed': len(tesseract_df),
            'successful_recognitions': sum(tesseract_df['label'].str.len() > 0),
            'empty_results': sum(tesseract_df['label'].str.len() == 0),
            'avg_confidence': tesseract_df[tesseract_df['confidence_score'] > 0]['confidence_score'].mean(),
            'recognition_rate': (sum(tesseract_df['label'].str.len() > 0) / len(tesseract_df) * 100),
            'high_confidence_results': sum(tesseract_df['confidence_score'] > 0.7)
        }
    }
    
    # Determine which implementation performed better
    easyocr_score = (
        0.4 * metrics['EasyOCR']['recognition_rate'] + 
        0.4 * (metrics['EasyOCR']['avg_confidence'] * 100) + 
        0.2 * (metrics['EasyOCR']['high_confidence_results'] / metrics['EasyOCR']['total_processed'] * 100)
    )
    
    tesseract_score = (
        0.4 * metrics['Tesseract']['recognition_rate'] + 
        0.4 * (metrics['Tesseract']['avg_confidence'] * 100) + 
        0.2 * (metrics['Tesseract']['high_confidence_results'] / metrics['Tesseract']['total_processed'] * 100)
    )
    
    recommended_engine = "EasyOCR" if easyocr_score > tesseract_score else "Tesseract"
    score_diff = abs(easyocr_score - tesseract_score)
    confidence_level = "strongly" if score_diff > 10 else "moderately" if score_diff > 5 else "slightly"
    
    # Generate comparison report
    with open(output_path, 'w') as f:
        f.write("OCR Results Comparison Report\n")
        f.write("===========================\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Recommendation\n")
        f.write("--------------\n")
        f.write(f"Based on the analysis, {recommended_engine} is {confidence_level} recommended ")
        f.write(f"for this handwriting recognition task.\n")
        f.write(f"Overall scores: EasyOCR: {easyocr_score:.2f}, Tesseract: {tesseract_score:.2f}\n\n")
        
        f.write("Performance Metrics\n")
        f.write("-----------------\n")
        
        for engine in ['EasyOCR', 'Tesseract']:
            f.write(f"\n{engine}:\n")
            m = metrics[engine]
            f.write(f"- Total Images Processed: {m['total_processed']}\n")
            f.write(f"- Successful Recognitions: {m['successful_recognitions']}\n")
            f.write(f"- Empty Results: {m['empty_results']}\n")
            f.write(f"- Recognition Rate: {m['recognition_rate']:.1f}%\n")
            f.write(f"- Average Confidence: {m['avg_confidence']:.3f}\n")
            f.write(f"- High Confidence Results (>0.7): {m['high_confidence_results']}\n")
        
        f.write("\nConclusion\n")
        f.write("----------\n")
        f.write(f"{recommended_engine} shows better overall performance for this specific task, ")
        f.write(f"with a {confidence_level} higher combined score considering recognition rate, ")
        f.write("confidence scores, and successful recognitions.\n")


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        # Set up paths
        output_dir = Path("output")
        easyocr_results = output_dir / "easyocr_results.csv"
        tesseract_results = output_dir / "tesseract_results.csv"
        comparison_report = output_dir / "ocr_comparison_report.txt"
        
        # Compare results and generate report
        compare_ocr_results(
            str(easyocr_results),
            str(tesseract_results),
            str(comparison_report)
        )
        
        print(f"\nComparison report generated: {comparison_report}")
        
    except Exception as e:
        logger.error(f"Error in comparison: {str(e)}")
        raise


if __name__ == "__main__":
    main()
