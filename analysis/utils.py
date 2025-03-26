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

def load_baseline_metrics():

    textual_analyzer = TextualEmbeddingsAnalyzer()
    node2vec_analyzer = Node2VecAnalyzer()
    asgc_analyzer = ASGCAnalyzer()

    baselines = []
    for _, row in textual_analyzer.df.iterrows():
        print(row)
        results = {
            "type": "textual",
            "name": row["model_name"],
            "dim": row["embedding_dim"],
            "acc/test": row["acc/test"],
            "acc/valid": row["acc/valid"],
            "lp_uniform/auc": row["lp_uniform/auc"],
            "lp_hard/auc": row["lp_hard/auc"],
        }
        baselines.append(results)

    for _, row in node2vec_analyzer.df.iterrows():
        results = {
            "type": "relational",
            "name": "node2vec",
            "dim": row["embedding_dim"],
            "acc/test": row["acc/test"],
            "acc/valid": row["acc/valid"],
            "lp_uniform/auc": row["lp_uniform/auc"],
            "lp_hard/auc": row["lp_hard/auc"],
        }
        baselines.append(results)
        
    for _, row in asgc_analyzer.df.iterrows():
        results = {
            "type": "relational",
            "name": "asgc",
            "dim": row["dim"],
            "acc/test": row["acc/test"],
            "acc/valid": row["acc/valid"],
            "lp_uniform/auc": row["lp_uniform/auc"],
            "lp_hard/auc": row["lp_hard/auc"],
        }
        baselines.append(results)
    return pd.DataFrame(baselines)

