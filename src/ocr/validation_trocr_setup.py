import json
from pathlib import Path


def create_ground_truth_file():
    """Create a ground truth JSON file from user input"""
    ground_truth = []
    
    # Create data directory structure if it doesn't exist
    data_dir = Path("data")
    snippets_dir = data_dir / "snippets"
    
    # Create directories if they don't exist
    snippets_dir.mkdir(parents=True, exist_ok=True)
    
    print("Available images in snippets directory:")
    image_files = list(snippets_dir.glob("*.[jp][pn][g]"))  # Match .jpg, .jpeg, .png
    
    if not image_files:
        print("No images found in data/snippets directory!")
        print("Please add your images to the data/snippets directory first.")
        return
    
    print("\nFound these images:")
    for idx, image_path in enumerate(image_files, 1):
        print(f"{idx}. {image_path.name}")
    
    print("\nLet's create the ground truth file.")
    print("For each image, enter the actual text that appears in it.")
    print("Press Enter without text to finish.\n")
    
    for image_path in image_files:
        text = input(f"Enter the text for {image_path.name} (or press Enter to skip): ").strip()
        if not text:
            continue
            
        ground_truth.append({
            "image_name": image_path.name,
            "text": text
        })
    
    # Save the ground truth file
    ground_truth_path = data_dir / "ground_truth.json"
    with open(ground_truth_path, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)
    
    print(f"\nGround truth file created at: {ground_truth_path}")
    return str(ground_truth_path)


def run_validation_pipeline():
    """Run the complete validation pipeline"""
    # First, create ground truth file if needed
    if not Path("data/ground_truth.json").exists():
        print("No ground truth file found. Let's create one.")
        ground_truth_file = create_ground_truth_file()
        if not ground_truth_file:
            return
    else:
        ground_truth_file = "data/ground_truth.json"
        
    # Import and run the validator
    from trocr_validator import validate_model  # This is from the previous code
    
    print("\nRunning validation...")
    stats = validate_model(
        data_dir="data/snippets",
        ground_truth_file=ground_truth_file
    )
    
    # Print results in a user-friendly format
    print("\n=== Validation Results ===")
    print(f"\nOverall Scores:")
    print(f"Average Similarity: {stats['average_similarity']:.2%}")
    print(f"Average Confidence: {stats['average_confidence']:.2%}")
    
    print("\nCommon Error Patterns:")
    for error_type, errors in stats['common_errors'].items():
        print(f"\n{error_type.title()}:")
        for item, count in errors.items():
            print(f"  {item}: {count} times")
    
    # Provide recommendations based on results
    print("\n=== Recommendations ===")
    
    if stats['average_similarity'] < 0.8:
        print("\n⚠️ Low similarity scores detected:")
        print("- Check image quality and preprocessing")
        print("- Consider adding more training examples")
        print("- Review commonly confused characters")
    
    if stats['average_confidence'] < 0.7:
        print("\n⚠️ Low confidence scores detected:")
        print("- Model is uncertain about predictions")
        print("- Consider fine-tuning on similar examples")
        print("- Add more varied training data")
    
    if len(stats['common_errors']['substitutions']) > 3:
        print("\n⚠️ Multiple character substitutions detected:")
        print("- Focus training on commonly confused characters")
        print("- Add more examples of these specific characters")
    
    print("\nResults have been saved to: data/validation_results/")
    print("Check the generated visualizations for detailed analysis of each image.")


if __name__ == "__main__":
    print("Welcome to the OCR Validation Setup!")
    print("\nThis script will help you:")
    print("1. Create a ground truth file for your images")
    print("2. Run validation against your images")
    print("3. Analyze the results and provide recommendations")
    
    input("\nPress Enter to continue...")
    run_validation_pipeline()
