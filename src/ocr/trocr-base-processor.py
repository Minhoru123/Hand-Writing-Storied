from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
from pathlib import Path


class HandwrittenTextDataset(Dataset):

    def __init__(self, data_dir, processor, valid_extensions=('.jpg', '.jpeg', '.png')):
        self.processor = processor
        self.valid_extensions = valid_extensions
        
        # Get all image paths from the snippets folder
        self.image_paths = []
        snippets_dir = Path(data_dir) / 'output'
        
        if not snippets_dir.exists():
            raise ValueError(f"Directory not found: {snippets_dir}")
            
        for ext in valid_extensions:
            self.image_paths.extend(list(snippets_dir.glob(f'*{ext}')))
            
        print(f"Found {len(self.image_paths)} images in {snippets_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            # Load and convert image
            image = Image.open(image_path).convert("RGB")
            
            # Process image
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
            return {
                'pixel_values': pixel_values.squeeze(),
                'image_path': str(image_path)
            }
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None


def process_handwritten_text(data_dir, batch_size=16, output_file='results.txt'):
    # Initialize model and processor
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()
    
    # Create dataset and dataloader
    dataset = HandwrittenTextDataset(data_dir, processor)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: [item for item in x if item is not None]  # Filter out None values
    )
    
    results = []
    
    # Process batches
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing images"):
            if not batch:  # Skip empty batches
                continue
                
            # Separate pixel values and paths
            pixel_values = torch.stack([item['pixel_values'] for item in batch]).to(device)
            image_paths = [item['image_path'] for item in batch]
            
            # Generate text
            generated_ids = model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Store results with file names
            for path, text in zip(image_paths, generated_text):
                results.append((path, text))
    
    # Save results to file
    output_path = Path(data_dir) / output_file
    with open(output_path, 'w', encoding='utf-8') as f:
        for path, text in results:
            f.write(f"{path}\t{text}\n")
    
    print(f"\nResults saved to {output_path}")
    return results


if __name__ == "__main__":
    # Process all images in the data/snippets directory
    data_dir = "enhanced_image_preprocessor_samples"
    
    try:
        results = process_handwritten_text(
            data_dir=data_dir,
            batch_size=16,
            output_file='ocr_results.txt'
        )
        
        # Print first few results as preview
        print("\nFirst few results:")
        for path, text in results[:5]:
            print(f"{Path(path).name}: {text}")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
