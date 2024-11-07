This is the processing folder and it has: 

### `The image_processor.py`

This script performs basic preprocessing steps:

- **Resizing:** Adjusts the image dimensions to a consistent size.
- **Grayscale Conversion:** Converts images to grayscale.
- **Contrast Enhancement:** Enhances image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
- **Sample Generation:** Creates sample output images showcasing the preprocessing steps applied to selected images.

### `The                                                                                                                                            enhanced_image_preprocessor.py`

This script applies more advanced preprocessing steps to prepare images for OCR:

- **Noise Reduction:** Applies techniques like Gaussian blur to reduce image noise.
- **Thresholding:** Enhances text regions in the image by applying thresholding methods like Otsu’s binarization.
- **Edge Detection:** Detects edges in the image, which can help identify text more clearly.
- **Sample Generation:** Displays all preprocessing steps applied to selected images, helping visualize the changes.
