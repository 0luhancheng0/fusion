from .base import AbstractAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Node2VecAnalyzer(AbstractAnalyzer):
    """Analyzer for Node2Vec embeddings results."""
    
    def __init__(self, dpi=300, cmap="viridis", figsize=(6.4, 4.8), 
                 remove_outliers=False, outlier_params=None):
        super().__init__("/home/lcheng/oz318/fusion/logs/Node2VecLightning", dpi, cmap, figsize, 
                        remove_outliers=remove_outliers, outlier_params=outlier_params)
    
    def post_process(self):
        """Extract dimension and seed from the directory structure."""
        if not self.df.empty:
            # Directory structure is /path/to/logs/Node2VecLightning/128/0/
            # Extract dimension from the second-to-last directory
            # Extract seed from the last directory
            self.df['dim'] = self.df['path'].apply(lambda p: int(p.split('/')[-2]))
            self.df['seed'] = self.df['path'].apply(lambda p: int(p.split('/')[-1]))
    
    def analyze(self):
        """Analyze Node2Vec results, grouping by dimension."""
        if self.df.empty:
            print("No data to analyze.")
            return None
        
        # Group by dimension
        result = (
            self.df.groupby(["dim"])
            .agg({
                "acc/val": ["mean", "std", "min", "max"],
                "acc/test": ["mean", "std", "min", "max"],
                "lp/auc": ["mean", "std", "min", "max"],
                "lp_hard/auc": ["mean", "std", "min", "max"],  # Added hard link prediction metric
            })
            .reset_index()
        )
        
        # Format column names for better readability
        result.columns = ["_".join(col).strip() for col in result.columns.values]
        return result
    
    def visualize(self):
        """Visualize Node2Vec results - performance by dimension."""
        if self.df.empty:
            print("No data to visualize.")
            return plt.figure()
        
        # Create three subplots - one for node classification, two for link prediction
        fig, axes = plt.subplots(1, 3, figsize=(self.figsize[0]*3, self.figsize[1]), squeeze=False)
        
        # Plot 1: Node classification (test accuracy) by dimension
        sns.boxplot(x="dim", y="acc/test", data=self.df, ax=axes[0, 0])
        axes[0, 0].set_title("Node Classification Accuracy by Dimension")
        axes[0, 0].set_xlabel("Embedding Dimension")
        axes[0, 0].set_ylabel("Test Accuracy")
        
        # Plot 2: Standard link prediction by dimension
        if "lp/auc" in self.df.columns:
            sns.boxplot(x="dim", y="lp/auc", data=self.df, ax=axes[0, 1])
            axes[0, 1].set_title("Link Prediction AUC by Dimension")
            axes[0, 1].set_xlabel("Embedding Dimension")
            axes[0, 1].set_ylabel("AUC")
        else:
            axes[0, 1].text(0.5, 0.5, "Link prediction data not available", 
                          ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_xticks([])
            axes[0, 1].set_yticks([])
            
        # Plot 3: Hard link prediction by dimension
        if "lp_hard/auc" in self.df.columns:
            sns.boxplot(x="dim", y="lp_hard/auc", data=self.df, ax=axes[0, 2])
            axes[0, 2].set_title("Hard Link Prediction AUC by Dimension")
            axes[0, 2].set_xlabel("Embedding Dimension")
            axes[0, 2].set_ylabel("AUC")
        else:
            axes[0, 2].text(0.5, 0.5, "Hard link prediction data not available", 
                          ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].set_xticks([])
            axes[0, 2].set_yticks([])
        
        plt.tight_layout()
        return self.save_and_return(fig, "performance_by_dimension")
    
    def visualize_task_comparison(self):
        """Compare node classification and link prediction performance."""
        if self.df.empty:
            print("No data for task comparison.")
            return plt.figure()
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Calculate average metrics by dimension
        agg_metrics = ['acc/test']
        if 'lp/auc' in self.df.columns:
            agg_metrics.append('lp/auc')
        if 'lp_hard/auc' in self.df.columns:
            agg_metrics.append('lp_hard/auc')
            
        dim_metrics = self.df.groupby('dim')[agg_metrics].mean().reset_index()
        
        # Plot comparison
        x = np.arange(len(dim_metrics))
        width = 0.8 / len(agg_metrics)  # Adjust bar width based on number of metrics
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, orange, green
        labels = ['Node Classification', 'Link Prediction', 'Hard Link Prediction']
        
        for i, metric in enumerate(agg_metrics):
            pos = x + width * (i - (len(agg_metrics) - 1) / 2)
            bars = ax.bar(pos, dim_metrics[metric], width, label=labels[i], color=colors[i])
            
            # Add value labels on bars
            for bar_idx, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8, color=colors[i])
        
        ax.set_xlabel('Embedding Dimension')
        ax.set_ylabel('Performance')
        ax.set_title('Node2Vec: Task Performance Comparison by Dimension')
        ax.set_xticks(x)
        ax.set_xticklabels(dim_metrics['dim'])
        ax.legend()
        
        plt.tight_layout()
        return self.save_and_return(fig, "task_comparison")
    
    def visualize_lp_comparison(self):
        """Compare standard and hard link prediction performance."""
        if self.df.empty or 'lp/auc' not in self.df.columns or 'lp_hard/auc' not in self.df.columns:
            print("No data for link prediction comparison.")
            return plt.figure()
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Calculate average metrics by dimension
        dim_metrics = self.df.groupby('dim')[['lp/auc', 'lp_hard/auc']].mean().reset_index()
        
        # Create line plot with markers
        ax.plot(dim_metrics['dim'], dim_metrics['lp/auc'], 'o-', label='Standard LP', linewidth=2, color='#ff7f0e')
        ax.plot(dim_metrics['dim'], dim_metrics['lp_hard/auc'], 's-', label='Hard LP', linewidth=2, color='#2ca02c')
        
        # Add value labels
        for i, row in dim_metrics.iterrows():
            ax.text(row['dim'], row['lp/auc'] + 0.01, f'{row["lp/auc"]:.3f}', 
                   ha='center', fontsize=9, color='#ff7f0e')
            ax.text(row['dim'], row['lp_hard/auc'] - 0.02, f'{row["lp_hard/auc"]:.3f}', 
                   ha='center', fontsize=9, color='#2ca02c')
        
        ax.set_xlabel('Embedding Dimension')
        ax.set_ylabel('AUC Score')
        ax.set_title('Comparison of Standard vs Hard Link Prediction')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self.save_and_return(fig, "lp_comparison")
        
    def find_best_dimension(self, metric="acc/test"):
        """Find the best performing dimension for specified metric."""
        # ...existing code...
    
    def run(self):
        """Run all available visualizations for Node2Vec analysis."""
        results = super().run()
            
        try:
            results["task_comparison"] = self.visualize_task_comparison()
            print("Task performance comparison created.")
        except Exception as e:
            print(f"Error creating task performance comparison: {e}")
            
        try:
            results["lp_comparison"] = self.visualize_lp_comparison()
            print("Link prediction comparison created.")
        except Exception as e:
            print(f"Error creating link prediction comparison: {e}")
        
        return results

