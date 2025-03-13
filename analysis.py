from pathlib import Path
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class Node2VecAnalyzer:
    def __init__(self, base_path="/home/lcheng/oz318/fusion/logs/Node2VecLightning", dpi=300, cmap="viridis"):
        self.base_path = base_path
        self.embeddings_paths = list(Path(self.base_path).glob("**/*.pt"))
        self.df = self._create_results_df()
        self.dpi = dpi
        self.cmap = sns.color_palette(cmap, as_cmap=True)
    
    def _create_results_df(self):
        """Load embeddings and create a DataFrame with results."""
        seeds = [i.parent.stem for i in self.embeddings_paths]
        dims = [i.parent.parent.stem for i in self.embeddings_paths]
        embeddings = [torch.load(i, weights_only=False) for i in self.embeddings_paths]
        results = [i['metadata']['results'] for i in embeddings]
        
        df = pd.DataFrame(results)
        df['seeds'] = seeds
        df['dims'] = dims
        return df
    
    def get_aggregated_results(self):
        """Aggregate results by dimension."""
        return self.df.groupby(['dims']).agg(
            {
                'acc/val': 'mean',
                'acc/test': 'mean',
                'lp/auc': 'mean'
            }
        )
    
    def create_performance_plot(self, metric='acc/test', figsize=(10, 6)):
        """Create a plot showing performance across different dimensions."""
        # Aggregate by dimension
        agg_df = self.df.groupby(['dims'])[metric].agg(['mean', 'std']).reset_index()
        # Convert dims to int for proper sorting
        agg_df['dims'] = agg_df['dims'].astype(int)
        agg_df = agg_df.sort_values('dims')
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)
        
        # Plot mean with error bars
        ax.errorbar(
            agg_df['dims'], 
            agg_df['mean'], 
            yerr=agg_df['std'],
            marker='o',
            linestyle='-',
            capsize=5
        )
        
        # Add labels and title
        ax.set_xlabel('Embedding Dimension')
        ax.set_ylabel(f'{metric.replace("/", " ")} Score')
        ax.set_title(f'Node2Vec Performance by Embedding Dimension')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return fig
    
    def create_heatmap(self, figsize=(12, 8)):
        """Create a heatmap showing results for all metrics across dimensions."""
        # Pivot the data for the heatmap
        metrics = ['acc/val', 'acc/test', 'lp/auc']
        agg_results = self.df.groupby(['dims'])[metrics].mean().reset_index()
        
        # Convert dims to int and sort
        agg_results['dims'] = agg_results['dims'].astype(int)
        agg_results = agg_results.sort_values('dims')
        
        # Reshape for heatmap
        pivot_df = agg_results.melt(
            id_vars=['dims'],
            value_vars=metrics,
            var_name='Metric',
            value_name='Score'
        )
        pivot_df = pivot_df.pivot(index='dims', columns='Metric', values='Score')
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)
        
        # Plot heatmap
        sns.heatmap(
            pivot_df,
            annot=True,
            fmt='.3f',
            cmap=self.cmap,
            ax=ax
        )
        
        ax.set_title('Node2Vec Performance by Dimension and Metric')
        ax.set_ylabel('Embedding Dimension')
        
        return fig
    
    def visualize_seed_variation(self, metric='acc/test', figsize=(10, 6)):
        """Visualize the variation across different seeds for each dimension."""
        # Prepare data
        dims = sorted([int(d) for d in self.df['dims'].unique()])
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)
        
        # Create box plot
        sns.boxplot(
            x='dims',
            y=metric,
            data=self.df,
            ax=ax
        )
        
        # Add individual points for better visualization
        sns.stripplot(
            x='dims',
            y=metric,
            data=self.df,
            color='black',
            size=4,
            alpha=0.5,
            ax=ax
        )
        
        # Add labels and title
        ax.set_xlabel('Embedding Dimension')
        ax.set_ylabel(f'{metric.replace("/", " ")} Score')
        ax.set_title(f'Node2Vec Performance Distribution by Dimension')
        
        return fig

# Example usage
# if __name__ == "__main__":
#     analyzer = Node2VecAnalyzer(dpi=300)
#     print(analyzer.get_aggregated_results())
    
#     # Create and save plots
#     fig1 = analyzer.create_performance_plot(metric='acc/test')
#     fig1.savefig("/home/lcheng/oz318/fusion/logs/figures/node2vec_test_accuracy.png")
    
#     fig2 = analyzer.create_heatmap()
#     fig2.savefig("/home/lcheng/oz318/fusion/logs/figures/node2vec_metrics_heatmap.png")
    
#     fig3 = analyzer.visualize_seed_variation()
#     fig3.savefig("/home/lcheng/oz318/fusion/logs/figures/node2vec_seed_variation.png")

