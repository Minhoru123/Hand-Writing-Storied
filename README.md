# Handwriting OCR Project

This project implements an OCR (Optical Character Recognition) system for analyzing handwritten text snippets. The goal is to accurately detect and extract text from handwritten samples, providing confidence scores for each recognition.

## Technical Approach & Reasoning

### Why Two OCR Engines?

1. **EasyOCR**
   * Modern deep learning approach
   * Better suited for handwriting variation
   * More configurable parameters
   * Higher accuracy potential
   * Slower processing speed
2. **Tesseract**
   * Industry standard baseline
   * Faster processing
   * Good for comparison
   * Less accurate with handwriting

## Performance Metrics

### EasyOCR

* Recognition Rate: ~30%
* Average Confidence: 0.101
* Processing Time: ~0.3s per image

### Tesseract

* Recognition Rate: ~22%
* Average Confidence: 0.009
* Processing Time: ~0.2s per image

## Confidence Score Interpretation

* 0.7-1.0: High confidence (likely correct)
* 0.4-0.7: Medium confidence (needs verification)
* <0.4: Low confidence (likely incorrect)

## Implementation Details

### Image Preprocessing Pipeline

* Grayscale conversion
* Contrast enhancement using CLAHE
* Noise removal
* Adaptive thresholding
* Stroke enhancement
* Border removal

## Usage

1. Place input images in `data/snippets/` directory
2. Run EasyOCR analysis:python easyocr_processor.py
3. Run Tesseract analysis:python tesseractocr_processor.py
4. Compare results:

    python src/ocr/comparing_ocrs.py

## Output Format

The system generates CSV files with the following columns:

* `snippet_name`: Original image filename
* `label`: Recognized text
* `confidence_score`: Recognition confidence (0-1)
