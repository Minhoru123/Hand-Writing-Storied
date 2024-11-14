from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import json
from difflib import SequenceMatcher
import numpy as np
from tqdm import tqdm


class OCRValidator:

    def __init__(self, model_name='microsoft/trocr-base-handwritten'):
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
    def analyze_single_prediction(self, image_path, ground_truth):
        """Analyze a single prediction in detail"""
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
        
        # Get prediction
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)
            prediction = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Calculate similarity
        similarity = SequenceMatcher(None, prediction.lower(), ground_truth.lower()).ratio()
        
        # Analyze character-level differences
        char_analysis = self._analyze_character_differences(prediction, ground_truth)
        
        # Get confidence scores
        confidence_info = self._get_confidence_scores(pixel_values, generated_ids)
        
        return {
            'prediction': prediction,
            'ground_truth': ground_truth,
            'similarity_score': similarity,
            'character_analysis': char_analysis,
            'confidence_scores': confidence_info
        }
    
    def _analyze_character_differences(self, prediction, ground_truth):
        """Analyze character-level differences between prediction and ground truth"""
        pred_chars = list(prediction.lower())
        true_chars = list(ground_truth.lower())
        
        analysis = {
            'missed_chars': [],
            'extra_chars': [],
            'substitutions': [],
            'correct_chars': []
        }
        
        # Use dynamic programming to find alignment
        m, n = len(pred_chars), len(true_chars)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill dp table
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0:
                    dp[i][j] = j
                elif j == 0:
                    dp[i][j] = i
                elif pred_chars[i - 1] == true_chars[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j],  # deletion
                                     dp[i][j - 1],  # insertion
                                     dp[i - 1][j - 1])  # substitution
        
        # Backtrack to find differences
        i, j = m, n
        while i > 0 and j > 0:
            if pred_chars[i - 1] == true_chars[j - 1]:
                analysis['correct_chars'].append(pred_chars[i - 1])
                i -= 1
                j -= 1
            else:
                if dp[i][j] == dp[i - 1][j] + 1:
                    analysis['extra_chars'].append(pred_chars[i - 1])
                    i -= 1
                elif dp[i][j] == dp[i][j - 1] + 1:
                    analysis['missed_chars'].append(true_chars[j - 1])
                    j -= 1
                else:
                    analysis['substitutions'].append((true_chars[j - 1], pred_chars[i - 1]))
                    i -= 1
                    j -= 1
        
        return analysis
    
    def _get_confidence_scores(self, pixel_values, generated_ids):
        """Get confidence scores for the prediction"""
        outputs = self.model(pixel_values, generated_ids)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        confidence_scores = probs.max(dim=-1)[0]
        
        return {
            'mean_confidence': float(confidence_scores.mean()),
            'min_confidence': float(confidence_scores.min()),
            'max_confidence': float(confidence_scores.max())
        }
    
    def visualize_analysis(self, result, save_path=None):
        """Create visualization of the analysis"""
        plt.figure(figsize=(15, 10))
        
        # Text comparison
        plt.subplot(2, 1, 1)
        plt.axis('off')
        comparison_text = (
            f"Prediction: {result['prediction']}\n"
            f"Ground Truth: {result['ground_truth']}\n"
            f"Similarity Score: {result['similarity_score']:.2f}\n\n"
            f"Confidence Scores:\n"
            f"  Mean: {result['confidence_scores']['mean_confidence']:.2f}\n"
            f"  Min: {result['confidence_scores']['min_confidence']:.2f}\n"
            f"  Max: {result['confidence_scores']['max_confidence']:.2f}"
        )
        plt.text(0.1, 0.5, comparison_text, fontsize=12, verticalalignment='center')
        
        # Character analysis
        plt.subplot(2, 1, 2)
        plt.axis('off')
        analysis = result['character_analysis']
        analysis_text = (
            f"Character Analysis:\n"
            f"Correct Characters: {', '.join(analysis['correct_chars'])}\n"
            f"Missed Characters: {', '.join(analysis['missed_chars'])}\n"
            f"Extra Characters: {', '.join(analysis['extra_chars'])}\n"
            f"Substitutions: {', '.join([f'{t}->{p}' for t, p in analysis['substitutions']])}"
        )
        plt.text(0.1, 0.5, analysis_text, fontsize=12, verticalalignment='center')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


def validate_model(data_dir, ground_truth_file, output_dir="validation_results"):
    """Run complete validation on a dataset"""
    validator = OCRValidator()
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load ground truth
    with open(ground_truth_file, 'r') as f:
        ground_truth = json.load(f)
    
    all_results = []
    summary_stats = {
        'similarity_scores': [],
        'confidence_scores': [],
        'error_patterns': {
            'missed_chars': {},
            'extra_chars': {},
            'substitutions': {}
        }
    }
    
    print("\nValidating images...")
    for item in tqdm(ground_truth):
        image_path = Path(data_dir) / item['image_name']
        result = validator.analyze_single_prediction(image_path, item['text'])
        
        # Save individual analysis
        validator.visualize_analysis(
            result,
            save_path=output_dir / f"analysis_{item['image_name']}.png"
        )
        
        # Update summary statistics
        summary_stats['similarity_scores'].append(result['similarity_score'])
        summary_stats['confidence_scores'].append(
            result['confidence_scores']['mean_confidence']
        )
        
        # Update error patterns
        analysis = result['character_analysis']
        for char in analysis['missed_chars']:
            summary_stats['error_patterns']['missed_chars'][char] = \
                summary_stats['error_patterns']['missed_chars'].get(char, 0) + 1
                
        for char in analysis['extra_chars']:
            summary_stats['error_patterns']['extra_chars'][char] = \
                summary_stats['error_patterns']['extra_chars'].get(char, 0) + 1
                
        for true_char, pred_char in analysis['substitutions']:
            key = f"{true_char}->{pred_char}"
            summary_stats['error_patterns']['substitutions'][key] = \
                summary_stats['error_patterns']['substitutions'].get(key, 0) + 1
        
        all_results.append(result)
    
    # Calculate final statistics
    final_stats = {
        'average_similarity': np.mean(summary_stats['similarity_scores']),
        'average_confidence': np.mean(summary_stats['confidence_scores']),
        'common_errors': {
            'missed': dict(sorted(
                summary_stats['error_patterns']['missed_chars'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]),
            'extra': dict(sorted(
                summary_stats['error_patterns']['extra_chars'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]),
            'substitutions': dict(sorted(
                summary_stats['error_patterns']['substitutions'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5])
        }
    }
    
    # Save summary report
    with open(output_dir / 'validation_summary.json', 'w') as f:
        json.dump(final_stats, f, indent=2)
    
    return final_stats


if __name__ == "__main__":
    # Example usage
    data_dir = "data/snippets"
    ground_truth_file = "data/ground_truth.json"
    
    # Run validation
    stats = validate_model(data_dir, ground_truth_file)
    
    print("\nValidation Summary:")
    print(f"Average Similarity Score: {stats['average_similarity']:.2f}")
    print(f"Average Confidence Score: {stats['average_confidence']:.2f}")
    
    print("\nMost Common Errors:")
    for error_type, errors in stats['common_errors'].items():
        print(f"\n{error_type.title()}:")
        for item, count in errors.items():
            print(f"  {item}: {count} times")
