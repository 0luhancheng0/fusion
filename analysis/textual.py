from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from .base import AbstractAnalyzer

class TextualEmbeddingsAnalyzer(AbstractAnalyzer):
    """Analyzer for textual embeddings evaluation results."""
    
    def __init__(self, dpi=300, cmap="viridis", figsize=(6.4, 4.8), 
                 remove_outliers=False, outlier_params=None):
        super().__init__("/home/lcheng/oz318/fusion/logs/TextualEmbeddings", dpi, cmap, figsize, 
                        remove_outliers=remove_outliers, outlier_params=outlier_params)
    
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
    
    def analyze(self):
        """Analyze textual embeddings results."""
        if self.df.empty:
            print("No data to analyze.")
            return None
        
        # Group by model and embedding dimension
        try:
            # Add lp_hard/auc to the aggregation if it exists
            agg_dict = {
                "acc/valid": ["mean", "std"],
                "acc/test": ["mean", "std"],
                "lp_uniform/auc": ["mean", "std"]
            }
            
            # Include hard link prediction if available
            if "lp_hard/auc" in self.df.columns:
                agg_dict["lp_hard/auc"] = ["mean", "std"]
                
            result = (
                self.df.groupby(["model", "embedding_dim"])
                .agg(agg_dict)
                .reset_index()
            )
            
            # Format column names
            result.columns = ["_".join(col).strip() for col in result.columns.values]
            return result
        except KeyError as e:
            print(f"Could not analyze data: {e}")
            return self.df
    
    def visualize(self):
        """Create a scatter plot with lines comparing test accuracy and link prediction metrics."""
        if self.df.empty:
            print("No data to visualize.")
            return plt.figure()
        
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
    
    def visualize_metrics(self):
        """Create a grouped bar chart comparing different metrics across models."""
        if self.df.empty:
            print("No data to visualize.")
            return plt.figure()
        
        # Determine which metrics to include
        metrics = ['acc/valid', 'acc/test', 'lp_uniform/auc']
        if 'lp_hard/auc' in self.df.columns:
            metrics.append('lp_hard/auc')
        
        # Melt the DataFrame to create a long format for grouped bars
        plot_data = self.df.copy()
        metrics_data = pd.melt(
            plot_data,
            id_vars=['model', 'embedding_dim'],
            value_vars=metrics,
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
    
    def visualize_dimension_impact(self):
        """Create a scatter plot showing relationship between dimension and performance."""
        if self.df.empty:
            print("No data to visualize.")
            return plt.figure()
        
        if 'embedding_dim' not in self.df.columns:
            print("No dimension information available.")
            return plt.figure()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create scatter plot with model-based coloring
        models = self.df['model'].unique()
        colors = sns.color_palette(self.cmap.name, len(models))
        
        for i, model in enumerate(models):
            model_data = self.df[self.df['model'] == model]
            ax.scatter(
                model_data['embedding_dim'], 
                model_data['acc/test'],
                c=[colors[i]],
                s=100, 
                label=model,
                alpha=0.7
            )
        
        # Add best-fit line if enough data points
        if len(self.df) > 2:
            try:
                # Use only models that have multiple dimensions
                dimension_counts = self.df.groupby('model')['embedding_dim'].nunique()
                multi_dim_models = dimension_counts[dimension_counts > 1].index.tolist()
                
                if multi_dim_models:
                    for model in multi_dim_models:
                        model_data = self.df[self.df['model'] == model]
                        if len(model_data) > 1:
                            # Add trend line for this model
                            x = model_data['embedding_dim']
                            y = model_data['acc/test']
                            z = np.polyfit(x, y, 1)
                            p = np.poly1d(z)
                            x_sorted = np.sort(x)
                            ax.plot(x_sorted, p(x_sorted), '--', color=colors[list(models).index(model)], alpha=0.7)
            except Exception as e:
                print(f"Could not create trend lines: {e}")
        
        ax.set_title('Impact of Embedding Dimension on Test Accuracy')
        ax.set_xlabel('Embedding Dimension')
        ax.set_ylabel('Test Accuracy')
        ax.grid(True, alpha=0.3)
        ax.legend(title='Model')
        
        plt.tight_layout()
        return self.save_and_return(fig, "dimension_impact")
    
    def find_best_model(self, metric='acc/test'):
        """Find the best performing model according to the specified metric."""
        if self.df.empty:
            print("No data to analyze.")
            return None
        
        best_idx = self.df[metric].idxmax()
        best_model = self.df.iloc[best_idx]
        
        print(f"Best model: {best_model['model']} with {metric}={best_model[metric]:.4f}")
        if 'embedding_dim' in best_model:
            print(f"Dimension: {best_model['embedding_dim']}")
            
        return best_model

    def run(self):
        """Run all available visualizations for textual embeddings analysis."""
        results = super().run()
        
        try:
            results["metrics"] = self.visualize_metrics()
            print("Metrics comparison visualization created.")
        except Exception as e:
            print(f"Error creating metrics comparison: {e}")
            
        try:
            results["dimension_impact"] = self.visualize_dimension_impact()
            print("Dimension impact visualization created.")
        except Exception as e:
            print(f"Error creating dimension impact visualization: {e}")
            
        return results
