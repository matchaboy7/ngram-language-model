"""
Standalone perplexity visualization tool.
Creates publication-quality plots from perplexity data.
"""
import matplotlib.pyplot as plt
from pathlib import Path


def plot_perplexity_comparison(models, perplexities, output_file='perplexity_comparison.png'):
    """
    Create a bar chart comparing perplexity across different n-gram models.
    
    Args:
        models: List of model names (e.g., ['Unigram', 'Bigram'])
        perplexities: List of corresponding perplexity values
        output_file: Path to save the output figure
    """
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, perplexities, color='steelblue', alpha=0.8,
                   edgecolor='navy', linewidth=1.5)
    
    # Add value labels on bars
    for bar, perp in zip(bars, perplexities):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{perp:.2f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Model Type', fontsize=12, fontweight='bold')
    plt.ylabel('Perplexity', fontsize=12, fontweight='bold')
    plt.title('N-gram Model Performance Comparison\n(Lower is Better)',
              fontsize=14, fontweight='bold', pad=15)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {Path(output_file).absolute()}")
    plt.show()


def main():
    """Example usage with sample data."""
    # Example data from actual runs
    models = ['Unigram', 'Bigram', 'Trigram', '4-gram']
    perplexities = [148.94, 23.56, 8.42, 3.17]  # Example values
    
    plot_perplexity_comparison(models, perplexities)


if __name__ == "__main__":
    main()