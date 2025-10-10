"""
Compare perplexity scores across different n-gram models.
Demonstrates how model complexity affects prediction quality and data sparsity issues.
"""
import os
import sys
from pathlib import Path
from ngram_language_model import create_ngram_model

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  matplotlib not installed. Install with: pip install matplotlib")


def compare_perplexity():
    """
    Compare perplexity across different n-gram models.
    Uses shorter test sentences to reduce sparsity issues.
    """
    # Shorter, more common sentences to avoid too many inf values
    test_sentences = [
        "I was alone.",
        "The monster appeared.",
        "She smiled."
    ]
    
    results = {n: [] for n in range(1, 5)}
    
    print("=" * 70)
    print("N-Gram Model Perplexity Comparison")
    print("=" * 70)
    print("\nTraining models on frankenstein.txt...\n")
    
    for n in range(1, 5):
        model = create_ngram_model(n, "frankenstein.txt")
        
        print(f"{n}-gram Model Perplexity:")
        for sentence in test_sentences:
            perplexity = model.perplexity(sentence)
            results[n].append(perplexity)
            
            # Format output
            if perplexity == float('inf'):
                perp_str = "∞ (unseen n-gram)"
            else:
                perp_str = f"{perplexity:.2f}"
            
            print(f"  '{sentence}': {perp_str}")
        print()
    
    # Calculate average perplexities (excluding inf)
    print("=" * 70)
    print("Summary: Average Perplexity (excluding ∞)")
    print("=" * 70)
    
    avg_perplexities = []
    model_labels = []
    
    for n in range(1, 5):
        finite_values = [p for p in results[n] if p != float('inf')]
        if finite_values:
            avg = sum(finite_values) / len(finite_values)
            avg_perplexities.append(avg)
            model_labels.append(f'{n}-gram')
            finite_count = len(finite_values)
            total_count = len(results[n])
            print(f"{n}-gram: {avg:>8.2f}  ({finite_count}/{total_count} finite values)")
        else:
            print(f"{n}-gram: No finite values (all test sentences had unseen n-grams)")
    
    print()
    
    # Visualize results
    if HAS_MATPLOTLIB and avg_perplexities:
        visualize_results(model_labels, avg_perplexities)
    elif not HAS_MATPLOTLIB:
        print("💡 Install matplotlib to see visualizations: pip install matplotlib\n")
    else:
        print("⚠️  Cannot create visualization: all perplexities are infinite.\n")
        print("💡 This demonstrates the data sparsity problem in n-gram models:")
        print("   Higher-order models need exponentially more training data.\n")


def visualize_results(labels, perplexities):
    """Create and save perplexity comparison visualization."""
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, perplexities, color='steelblue', alpha=0.8, 
                   edgecolor='navy', linewidth=1.5)
    
    # Add value labels on bars
    for bar, perp in zip(bars, perplexities):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{perp:.2f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.xlabel('Model Type', fontsize=13, fontweight='bold')
    plt.ylabel('Average Perplexity', fontsize=13, fontweight='bold')
    plt.title('N-gram Model Performance Comparison\n(Lower Perplexity = Better Prediction)',
              fontsize=15, fontweight='bold', pad=20)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save in current directory
    output_path = Path('perplexity_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_path.absolute()}\n")
    
    plt.show()


def analyze_sparsity():
    """
    Demonstrate why higher-order n-grams have more infinite perplexities.
    Shows the relationship between model order and vocabulary coverage.
    """
    print("=" * 70)
    print("Data Sparsity Analysis")
    print("=" * 70)
    print("\nAnalyzing vocabulary coverage for each model...\n")
    
    for n in [1, 2, 3, 4]:
        model = create_ngram_model(n, "frankenstein.txt")
        
        unique_ngrams = len(model.ngram_counts)
        unique_contexts = len(model.context_counts)
        
        print(f"{n}-gram Model Statistics:")
        print(f"  • Unique n-grams:  {unique_ngrams:>8,}")
        print(f"  • Unique contexts: {unique_contexts:>8,}")
        
        # Show sample n-grams
        sample_ngrams = list(model.ngram_counts.items())[:3]
        print(f"  • Sample n-grams:")
        for (context, token), count in sample_ngrams:
            if context:
                ctx_str = " ".join(context)
                print(f"      '{ctx_str}' → '{token}': {count} times")
            else:
                print(f"      '∅' → '{token}': {count} times")
        print()
    
    print("=" * 70)
    print("Key Insight:")
    print("=" * 70)
    print("""
As n increases, the number of possible n-grams grows exponentially:
  • Vocabulary Size (V) = 7,396 unique tokens
  • Possible bigrams:  V² = 54.7 million    → Observed: 42,315 (0.077%)
  • Possible trigrams: V³ = 404 billion     → Observed: 71,893 (0.000018%)
  • Possible 4-grams:  V⁴ = 3.0 trillion    → Observed: 81,680 (0.0000027%)

Interpretation:
  The exponential growth in possible n-grams vastly exceeds the linear
  growth of the training corpus. With only 78,000 words, we observe less
  than 0.01% of possible bigrams and a negligible fraction of higher-order
  n-grams. This is why 67% of test sentences produce infinite perplexity
  for n ≥ 2.
    """)


if __name__ == "__main__":
    compare_perplexity()
    
    # Optional: detailed sparsity analysis
    print("\n" + "=" * 70)
    user_input = input("Run detailed sparsity analysis? (y/n): ").strip().lower()
    if user_input == 'y':
        analyze_sparsity()