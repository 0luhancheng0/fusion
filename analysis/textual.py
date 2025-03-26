from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from .base import AbstractAnalyzer

class TextualEmbeddingsAnalyzer(AbstractAnalyzer):
    """Analyzer for textual embeddings evaluation results."""
    
    def __init__(self, dpi=300, cmap="viridis", figsize=(6.4, 4.8)):
        super().__init__("/home/lcheng/oz318/fusion/logs/TextualEmbeddings", dpi, cmap, figsize)
    
    def post_process(self):
        """Process the DataFrame to extract model names and dimensions."""
        if not self.df.empty:
            # Model name is directly in the path
            self.df['model'] = self.df['path'].apply(lambda p: Path(p).name)
            
            # Extract embedding dimension from the path or from the config
            if 'embedding_dim' in self.df.columns:
                # Use existing column if available
                pass
            else:
                # Try to extract from path or embedding_path
                if 'embedding_path' in self.df.columns:
                    self.df['embedding_dim'] = self.df['embedding_path'].apply(
                        lambda p: int(Path(p).stem) if Path(p).stem.isdigit() else None
                    )

    
    def scatter_plot(self):
        """Create a scatter plot with lines comparing test accuracy and link prediction metrics."""

        
        # Prepare data for plotting
        plot_data = self.df.copy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Get unique models and sort them for consistent display
        models = sorted(plot_data['model'].unique())
        
        # Calculate x-positions (evenly spaced)
        x_positions = range(len(models))
        
        # Get accuracy and link prediction values for each model
        test_accuracies = [plot_data[plot_data['model'] == model]['acc/test'].mean() for model in models]
        lp_aucs = [plot_data[plot_data['model'] == model]['lp_uniform/auc'].mean() for model in models]
        
        # Check if hard link prediction data is available
        has_hard_lp = 'lp_hard/auc' in self.df.columns
        if has_hard_lp:
            lp_hard_aucs = [plot_data[plot_data['model'] == model]['lp_hard/auc'].mean() for model in models]
        
        # Create a second y-axis for link prediction
        ax2 = ax.twinx()
        
        # Plot test accuracy on left y-axis
        line1, = ax.plot(x_positions, test_accuracies, '-', color='#1f77b4', linewidth=2, alpha=0.8, label='Test Accuracy')
        ax.scatter(x_positions, test_accuracies, s=120, color='#1f77b4', 
                  edgecolor='white', linewidth=2, zorder=5)
        
        # Plot standard link prediction on right y-axis
        line2, = ax2.plot(x_positions, lp_aucs, '-', color='#ff7f0e', linewidth=2, alpha=0.8, label='LP AUC')
        ax2.scatter(x_positions, lp_aucs, s=120, color='#ff7f0e', 
                   edgecolor='white', linewidth=2, marker='s', zorder=5)
                   
        # Plot hard link prediction if available
        if has_hard_lp:
            line3, = ax2.plot(x_positions, lp_hard_aucs, '-', color='#2ca02c', linewidth=2, alpha=0.8, label='Hard LP AUC')
            ax2.scatter(x_positions, lp_hard_aucs, s=120, color='#2ca02c', 
                       edgecolor='white', linewidth=2, marker='^', zorder=5)
        
        # Add value labels for test accuracy (above)
        for x, y, model in zip(x_positions, test_accuracies, models):
            # Add test accuracy value
            ax.annotate(
                f'{y:.3f}',
                (x, y),
                xytext=(0, 7),  # 7 points above
                textcoords='offset points',
                ha='center',
                fontsize=9,
                color='#1f77b4',
                fontweight='bold'
            )
            
            # Add dimension label below if available
            if 'embedding_dim' in plot_data.columns:
                dim = plot_data[plot_data['model'] == model]['embedding_dim'].iloc[0]
                ax.annotate(
                    f'd={dim}',
                    (x, y),
                    xytext=(0, -15),  # 15 points below
                    textcoords='offset points',
                    ha='center',
                    fontsize=8,
                    alpha=0.7
                )
        
        # Add value labels for standard link prediction
        for x, y in zip(x_positions, lp_aucs):
            ax2.annotate(
                f'{y:.3f}',
                (x, y),
                xytext=(0, -20),  # 20 points below
                textcoords='offset points',
                ha='center',
                fontsize=9,
                color='#ff7f0e',
                fontweight='bold'
            )
            
        # Add value labels for hard link prediction if available
        if has_hard_lp:
            for x, y in zip(x_positions, lp_hard_aucs):
                ax2.annotate(
                    f'{y:.3f}',
                    (x, y),
                    xytext=(0, -35),  # 35 points below (further down than standard LP)
                    textcoords='offset points',
                    ha='center',
                    fontsize=9,
                    color='#2ca02c',
                    fontweight='bold'
                )
        
        # Set labels and title
        ax.set_title('Textual Embeddings Performance Comparison', fontsize=14)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Test Accuracy', color='#1f77b4', fontsize=12)
        ax2.set_ylabel('Link Prediction AUC', color='#ff7f0e', fontsize=12)
        
        # Set tick parameters
        ax.tick_params(axis='y', colors='#1f77b4')
        ax2.tick_params(axis='y', colors='#ff7f0e')
        
        # Set x-axis ticks to model names
        ax.set_xticks(x_positions)
        ax.set_xticklabels(models, rotation=45, ha='right')
        
        # Set appropriate y-axis limits with padding
        ax.set_ylim(max(0, min(test_accuracies) - 0.03), min(1, max(test_accuracies) + 0.03))
        
        # Calculate min and max for link prediction y-axis
        lp_values = lp_aucs
        if has_hard_lp:
            lp_values = lp_values + lp_hard_aucs
        ax2.set_ylim(max(0, min(lp_values) - 0.03), min(1, max(lp_values) + 0.03))
        
        # Add grid lines for better readability (only for the left y-axis)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add combined legend with all available metrics
        lines = [line1, line2]
        labels = [line1.get_label(), line2.get_label()]
        if has_hard_lp:
            lines.append(line3)
            labels.append(line3.get_label())
        ax.legend(lines, labels, loc='upper right', frameon=True, framealpha=0.9)
        
        plt.tight_layout()
        return self.save_and_return(fig, "combined_performance")
    
    def bar_plot(self):
        """Create a grouped bar chart comparing different metrics across models."""

        # Determine which metrics to include
        # metrics = ['acc/valid', 'acc/test', 'lp_uniform/auc', 'lp_hard/auc']

        
        # Melt the DataFrame to create a long format for grouped bars
        plot_data = self.df.copy()
        metrics_data = pd.melt(
            plot_data,
            id_vars=['model', 'embedding_dim'],
            value_vars=self.metrics,
            var_name='metric', 
            value_name='score'
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Create grouped bar chart
        sns.barplot(x='model', y='score', hue='metric', data=metrics_data, ax=ax)
        
        ax.set_title('Textual Embeddings Performance Metrics by Model')
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return self.save_and_return(fig, "metrics_comparison")
    
    


    def run(self):
        """Run all available visualizations for textual embeddings analysis."""
        results = {}
        results['scatterplot'] = self.scatter_plot()
        results["barplot"] = self.bar_plot()
            
        return results
