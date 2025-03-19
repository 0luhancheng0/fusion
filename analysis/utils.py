import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Import analyzers for loading baseline data
from .textual import TextualEmbeddingsAnalyzer
from .node2vec import Node2VecAnalyzer
from .asgc import ASGCAnalyzer

# analysis/utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_heatmap(data, x_col, y_col, value_col, ax=None, cmap="viridis", 
                   annot=True, title=None, xlabel=None, ylabel=None):
    """Create a standard heatmap."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()
        
    # Create pivot table
    pivot_data = data.pivot(index=y_col, columns=x_col, values=value_col)
    
    # Create heatmap
    sns.heatmap(pivot_data, cmap=cmap, annot=annot, fmt=".3f", ax=ax)
    
    # Set labels
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
        
    return fig, ax

def create_boxplot(data, x_col, y_col, ax=None, title=None, xlabel=None, ylabel=None):
    """Create a standard boxplot."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()
        
    # Create boxplot
    sns.boxplot(x=x_col, y=y_col, data=data, ax=ax)
    
    # Set labels
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
        
    return fig, ax

def create_performance_comparison(models, metrics, values, ax=None, title=None):
    """Create a bar chart comparing metrics across models."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()
        
    # Calculate positions for bars
    x = np.arange(len(models))
    width = 0.8 / len(metrics)
    
    # Create grouped bars
    for i, metric in enumerate(metrics):
        pos = x + width * (i - (len(metrics) - 1) / 2)
        bars = ax.bar(pos, values[i], width, label=metric)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Set labels
    if title:
        ax.set_title(title)
    ax.set_xlabel('Model')
    ax.set_ylabel('Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    return fig, ax

def load_baseline_metrics(
    dpi: int = 300,
    cmap: str = "viridis", 
    figsize: Tuple[float, float] = (6.4, 4.8),
    verbose: bool = False
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Load baseline metrics for textual and relational embeddings.
    
    Args:
        dpi: DPI for figures (passed to analyzers)
        cmap: Colormap name for visualizations (passed to analyzers)
        figsize: Figure size tuple (passed to analyzers) 
        verbose: Whether to print debug information
        
    Returns:
        dict: Dictionary containing baseline metrics for textual and relational embeddings
             with the structure:
             {
                 'textual': {
                     'model_name/dim': {'acc/test': value, 'lp_uniform/auc': value, ...},
                     ...
                 },
                 'relational': {
                     'dim': {'acc/test': value, 'lp_uniform/auc': value, ...},
                     ...
                 }
             }
    """
    baselines = {
        'textual': {},
        'relational': {}
    }
    
    # Load textual baselines from TextualEmbeddings
    textual_path = "/home/lcheng/oz318/fusion/logs/TextualEmbeddings"
    if Path(textual_path).exists():
        textual_analyzer = TextualEmbeddingsAnalyzer(dpi=dpi, cmap=cmap, figsize=figsize)
        for _, row in textual_analyzer.df.iterrows():
            if 'model' in row and 'embedding_dim' in row:
                # Create a unique key combining model name and dimension
                key = f"{row['model']}/{row['embedding_dim']}"
                baselines['textual'][key] = {}
                
                # Add all available metrics
                for metric in ['acc/test', 'acc/valid', 'lp_uniform/auc', 'lp_hard/auc']:
                    if metric in row and not pd.isna(row[metric]):
                        baselines['textual'][key][metric] = row[metric]
    
    # Load Node2Vec relational baselines
    node2vec_path = "/home/lcheng/oz318/fusion/logs/Node2VecLightning"
    if Path(node2vec_path).exists():
        node2vec_analyzer = Node2VecAnalyzer(dpi=dpi, cmap=cmap, figsize=figsize)
        # Group by dimension and average the metrics
        if not node2vec_analyzer.df.empty:
            metrics = ['acc/test', 'acc/val', 'lp_uniform/auc', 'lp_hard/auc']
            dim_metrics = node2vec_analyzer.df.groupby('dim')[metrics].mean().reset_index()
            
            for _, row in dim_metrics.iterrows():
                dim = str(int(row['dim']))  # Convert to string for key
                if dim not in baselines['relational']:
                    baselines['relational'][dim] = {}
                    
                # Add all available metrics
                for metric in metrics:
                    if metric in row and not pd.isna(row[metric]):
                        baselines['relational'][dim][metric] = row[metric]
    
    # Load ASGC relational baselines
    asgc_path = "/home/lcheng/oz318/fusion/logs/ASGC"
    if Path(asgc_path).exists():
        asgc_analyzer = ASGCAnalyzer(dpi=dpi, cmap=cmap, figsize=figsize)
        # Group by dimension and find best regularization and k values
        if not asgc_analyzer.df.empty:
            # Get the best performing config for each dimension
            best_configs = []
            for dim in asgc_analyzer.df['dim'].unique():
                dim_data = asgc_analyzer.df[asgc_analyzer.df['dim'] == dim]
                best_idx = dim_data['acc/test'].idxmax()
                best_configs.append(dim_data.loc[best_idx])
            
            best_configs_df = pd.DataFrame(best_configs)
            
            for _, row in best_configs_df.iterrows():
                dim = str(int(row['dim']))  # Convert to string for key
                if dim not in baselines['relational']:
                    baselines['relational'][dim] = {}
                
                # Add metrics, with "asgc_" prefix to distinguish from Node2Vec
                for metric in ['acc/test', 'acc/valid', 'lp_uniform/auc', 'lp_hard/auc']:
                    if metric in row and not pd.isna(row[metric]):
                        baselines['relational'][f"asgc_{dim}"] = {}
                        baselines['relational'][f"asgc_{dim}"][metric] = row[metric]
    
    if verbose:
        print(f"Textual baselines: {list(baselines['textual'].keys())[:5]}") # First 5 keys
        print(f"Relational baselines: {list(baselines['relational'].keys())[:5]}") # First 5 keys

    return baselines

if __name__ == "__main__":
    # When run directly, load and print the baselines with verbose output
    baselines = load_baseline_metrics(verbose=True)
    
    # Print statistics about the loaded baselines
    print("\nBaseline Statistics:")
    print(f"Number of textual baseline models: {len(baselines['textual'])}")
    print(f"Number of relational baseline models: {len(baselines['relational'])}")
    
    # Print example metric values for a few models
    if baselines['textual']:
        model_key = next(iter(baselines['textual']))
        print(f"\nExample textual model ({model_key}):")
        print(baselines['textual'][model_key])
        
    if baselines['relational']:
        model_key = next(iter(baselines['relational']))
        print(f"\nExample relational model ({model_key}):")
        print(baselines['relational'][model_key])
