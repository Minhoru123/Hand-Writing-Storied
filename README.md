# Handwriting OCR Project

This project implements an OCR (Optical Character Recognition) system for analyzing handwritten text snippets. The goal is to accurately detect and extract text from handwritten samples, providing confidence scores for each recognition.

### Key Features

* Dual OCR engine comparison (EasyOCR and Tesseract)
* Advanced image preprocessing pipeline
* Optimized parameter configurations
* Confidence score metrics
* CSV output format for results

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

### Understanding Confidence Scores

* **Scale** : 0-1 (0% to 100% confidence)
* **Calculation Method** : Probability of correct text recognition
* **Interpretation** :

  `0.7-1.0: High confidence (likely correct) 0.4-0.7: Medium confidence (needs verification) <0.4: Low confidence (likely incorrect)`
* **Project Results** :

  `Average Confidence: 0.101 (10.1%) Best Cases: 0.963 (96.3%)`

## Development Process

### Step 1: Initial Data Analysis

Key findings from the analysis
Image Characteristics:

- Dimensions: 515x46 pixels (average)
- Format: RGB-stored grayscale
- Challenges: Background artifacts, variable text positioning

### Step 2: Preprocessing Pipeline Implementation

`def preprocess_image(self, image):
    # 1.Convert to grayscale

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Enhance contrast (improve text visibility)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # 3. Remove noise (clean up image)
    denoised = cv2.medianBlur(enhanced, 3)

    # 4. Threshold (separate text from background)
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    return binary

### Step 3: OCR Engine Optimization

##### EasyOCR Configuration Evolution

` # Initial Parameters`

`default_params = {     'paragraph': False,     'min_size': 10,     'text_threshold': 0.7,     'link_threshold': 0.4,     'contrast_ths': 0.2 }`

`Results: 28 % success rate`

`# Optimized Parameters`

`optimized_params = {      'paragraph': False,      'min_size': 5,          # Smaller text detection      'contrast_ths': 0.05,   # More aggressive contrast      'adjust_contrast': 1.5, # Enhanced contrast      'text_threshold': 0.3,  # Lower detection threshold      'link_threshold': 0.2,  # Lenient text linking      'mag_ratio': 2         # Increased image size }`

`Results: 30% success rate`

##### Tesseract Configuration Evolution

`# Initial Config default_config = {     'lang': 'eng',     'config': '--psm 6' }`

`Results: 21% success rate`

` #Optimized Config optimized_config = {     'lang': 'eng',     'config': '--psm 8 --oem 1',     'custom_oem_psm_config': True }`

`Results: 22% success rate`

### Step 4: Output Format Implementation

`# CSV Structure`
`columns = [     'snippet_name',      # Original filename     'label',            # Recognized text     'confidence_score'  # 0-1 value ]`

`Example Output snippet_name,label,confidence_score image1.png,detected_text,0.78 image2.png, blank, 0.01`

## Implementation Details

### Key Components

1. **Image Preprocessor**
   * Handles image cleaning and enhancement
   * Implements complete preprocessing pipeline
   * Optimized for OCR input
2. **OCR Fine-tuner**
   * Manages OCR engine parameters
   * Implements testing configurations
   * Tracks performance metrics
3. **CSV Generator**
   * Creates standardized output format
   * Handles empty cases appropriately
   * Implements error handling

## Results & Analysis

### Performance Metrics

EasyOCR Performance:

- Initial Success Rate: 28%`
- Optimized Success Rate: 30%`
- Average Confidence: 0.101
- Best Case Confidence: 0.963
- Processing Time: ~0.3s per image`

Tesseract Performance:

- Initial Success Rate: 21%
- Final Success Rate: 22%
- Average Confidence: 0.009
- Processing Time: ~0.2s per image

### Challenges & Solutions

1. **Low Confidence Scores**
   * Challenge: Poor initial recognition rates
   * Solution: Enhanced preprocessing and parameter tuning
   * Result: Improved high-confidence cases
2. **Processing Speed**
   * Challenge: Slow EasyOCR processing
   * Solution: Optimized parameters and preprocessing
   * Result: Reduced processing time

## Installation & Usage

### Dependencies

`# Create virtual environment`
`python -m venv myenv myenv\Scripts\activate  # Windows source myenv/bin/activate # Linux/Mac`

# Install required packages

`pip install easyocr opencv-python numpy pandas`

### Running the Project

`#Preprocess images python src/enhanced_image_preprocessor.py`

`#Generate final CSV python src/csv_output_report.py`
