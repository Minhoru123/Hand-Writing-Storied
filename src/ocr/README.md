## OCR Results:

Both OCR models are used to extract text from a set of images. Below is a summary of their performance:

### EasyOCR Results:

* **Total images processed** : 100
* **Successful** : 100
* **Failed** : 0
* **Average confidence** : 0.07 (indicating generally low confidence)

Sample Results from EasyOCR (first 5 successful):

1. **File** : 007625911_007625911_00105_22.png
   **Text** : (empty)
   **Confidence** : 0.00
2. **File** : 007625911_007625911_00621_22.png
   **Text** : (empty)
   **Confidence** : 0.00
3. **File** : 007625911_007625911_01452_22.png
   **Text** : U 1L
   **Confidence** : 0.09
4. **File** : 007625911_007625911_02101_22.png
   **Text** : (empty)
   **Confidence** : 0.00
5. **File** : 007625911_007625911_03861_22.png
   **Text** : (empty)
   **Confidence** : 0.00

### Tesseract Results:

* **Total images processed** : 100
* **Successful** : 100
* **Failed** : 0
* **Average confidence** : 0.01 (very low confidence)

Sample Results from Tesseract (first 5 successful):

1. **File** : 007625911_007625911_00105_22.png
   **Text** : ���
   **Confidence** : 0.03
2. **File** : 007625911_007625911_00621_22.png
   **Text** : (empty)
   **Confidence** : -0.01
3. **File** : 007625911_007625911_01452_22.png
   **Text** : (empty)
   **Confidence** : -0.01
4. **File** : 007625911_007625911_02101_22.png
   **Text** : (empty)
   **Confidence** : -0.01
5. **File** : 007625911_007625911_03861_22.png
   **Text** : (empty)
   **Confidence** : -0.01

## Notes:

* Both OCR models performed poorly on this dataset with low confidence values.
* EasyOCR seemed to capture very limited text (e.g., "U 1L" with low confidence).
* Tesseract struggled even more with reading the images, often outputting random characters or no text at all.
* The low confidence may be due to the quality of the images or misalignment in the text, suggesting a need for better preprocessing or image enhancement.
* Future improvements may include experimenting with different OCR settings, training on more diverse data, or enhancing the image preprocessing pipeline.
