"""
Simple text generation demo using the N-gram model.
This script trains models on Frankenstein and generates random text.
"""
import random
import sys
from pathlib import Path
from ngram_language_model import create_ngram_model


def main():
    """Generate random text using n-gram models of different orders."""
    
    # Check if data file exists
    data_file = "frankenstein.txt"
    if not Path(data_file).exists():
        print(f"❌ Error: Cannot find {data_file}")
        print("   Please ensure frankenstein.txt is in the current directory.")
        sys.exit(1)
    
    print("=" * 70)
    print(" " * 15 + "N-Gram Language Model")
    print(" " * 18 + "Text Generation Demo")
    print("=" * 70)
    print(f"\nTraining on: {data_file}")
    print(f"Generating 30 tokens per model\n")
    print("=" * 70)
    
    # Train models with different n values
    for n in [1, 2, 3, 4]:
        print(f"\n{n}-gram Model")
        print("-" * 70)
        
        model = create_ngram_model(n, data_file)
        
        random.seed(42)  # For reproducibility
        generated_text = model.random_text(30)
        
        print(f"Generated text:\n{generated_text}\n")
    
    print("=" * 70)
    print("\n💡 Observation:")
    print("   As n increases, generated text becomes more coherent")
    print("   but may also become more repetitive (memorization).")
    print()


if __name__ == "__main__":
    main()