# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("image-to-text", model="microsoft/trocr-base-handwritten")

# Load model directly
from transformers import AutoTokenizer, AutoModelForImageTextToText

tokenizer = AutoTokenizer.from_pretrained("microsoft/trocr-base-handwritten")
model = AutoModelForImageTextToText.from_pretrained("microsoft/trocr-base-handwritten")
