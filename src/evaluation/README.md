## Evaluation Folder


The **evaluation** folder contains key scripts used to assess and fine-tune the performance of different OCR engines on the handwritten dataset. The two main scripts in this folder are:

1. **ocr_comparison.py** : This script compares the performance of Tesseract and EasyOCR on a set of handwritten images, providing insights into each model's accuracy and confidence.
2. **ocr_fine_tuner.py** : This script is designed to fine-tune the EasyOCR model by adjusting hyperparameters and evaluating its performance to improve accuracy and reliability.

These scripts serve as the foundation for evaluating OCR models and fine-tuning them to achieve better results on real-world handwriting datasets.


## ocr_comparison.py

### Purpose

Compares Tesseract and EasyOCR based on accuracy and confidence scores, providing insights into which model performs better on the dataset.

### Key Steps

1. **OCR Predictions** : Runs both Tesseract and EasyOCR on the same set of images.
2. **Evaluation** : Calculates accuracy and confidence scores for both models.
3. **Results** : Outputs a comparison of accuracy and confidence, with visuals showing OCR predictions for each model.

### Purpose of Comparison

The comparison helps justify choosing EasyOCR for this project due to its superior ability to handle diverse handwriting styles compared to Tesseract.

## ocr_fine_tuner.py

### Purpose

Fine-tunes the EasyOCR model to improve its accuracy with handwritten text, especially non-linear and skewed samples.

### Key Steps

1. **Loading Model** : Starts with a pre-trained EasyOCR model.
2. **Data Augmentation** : Applies transformations like rotation and noise to enhance model robustness.
3. **Hyperparameter Tuning** : Adjusts settings like learning rate and batch size for better performance.
4. **Evaluation** : Assesses the fine-tuned model's performance on a validation set.

### Purpose of Fine-Tuning

Fine-tuning helps EasyOCR perform better on challenging handwriting styles, improving accuracy and confidence.
