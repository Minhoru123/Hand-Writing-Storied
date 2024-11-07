### Preprocessing Folder

This folder contains scripts responsible for various image preprocessing techniques that help enhance the quality of the images before performing OCR. The following scripts are included:

### `image_processor.py`

This script performs basic preprocessing steps:

* **Resizing:** Adjusts the image dimensions to a consistent size.
* **Grayscale Conversion:** Converts the image to grayscale to simplify the analysis.
* **Contrast Enhancement:** Enhances the image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization), which makes text stand out more clearly against the background.
* **Sample Generation:** Creates output images that showcase how each preprocessing step affects the image, allowing for better visualization of the changes.

### `enhanced_image_preprocessor.py`

This script applies more advanced preprocessing steps aimed at preparing the images for more accurate OCR:

* **Noise Reduction:** Reduces unwanted noise using methods like Gaussian blur, which smooths the image and removes minor disturbances that might interfere with OCR recognition.
* **Deskewing:** Corrects the alignment of any skewed or rotated text, ensuring that the text is in a horizontal alignment, which improves the performance of OCR algorithms.
* **Thresholding:** Applies techniques like Otsu’s binarization to separate the foreground (text) from the background, converting the image into a binary (black-and-white) format. This helps OCR models focus on detecting the characters clearly.
* **Edge Detection:** Uses methods like the Canny edge detector to identify and highlight the edges of letters and words. This can help OCR engines recognize text more effectively, especially with difficult handwriting.
* **Sample Generation:** Displays all preprocessing steps applied to selected images, enabling you to visually compare the results before and after the various preprocessing techniques.
