from .base import AbstractAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Node2VecAnalyzer(AbstractAnalyzer):
    """Analyzer for Node2Vec embeddings results."""
    
    def __init__(self, dpi=300, cmap="viridis", figsize=(6.4, 4.8)):
        super().__init__("/home/lcheng/oz318/fusion/logs/Node2VecLightning", dpi, cmap, figsize)
    
    def post_process(self):
        if not self.df.empty:
            self.df['dim'] = self.df['path'].apply(lambda p: int(p.split('/')[-2]))
            self.df['seed'] = self.df['path'].apply(lambda p: int(p.split('/')[-1]))
    
    def analyze(self):
        result = (
            self.df.groupby(["dim"])
            .agg({
                "acc/valid": ["mean", "std", "min", "max"],
                "acc/test": ["mean", "std", "min", "max"],
                "lp_uniform/auc": ["mean", "std", "min", "max"],
                "lp_hard/auc": ["mean", "std", "min", "max"],  # Added hard link prediction metric
            })
            .reset_index()
        )
        
        result.columns = ["_".join(col).strip() for col in result.columns.values]
        return result
    
    
    
    def performance_by_dimension(self):

        # Create three subplots - one for node classification, two for link prediction
        fig, axes = plt.subplots(1, 3, figsize=(self.figsize[0]*3, self.figsize[1]), squeeze=False)
        
        # Plot 1: Node classification (test accuracy) by dimension
        sns.boxplot(x="dim", y="acc/test", data=self.df, ax=axes[0, 0])
        axes[0, 0].set_title("Node Classification Accuracy by Dimension")
        axes[0, 0].set_xlabel("Embedding Dimension")
        axes[0, 0].set_ylabel("Test Accuracy")
        
        # Plot 2: Standard link prediction by dimension
        sns.boxplot(x="dim", y="lp_uniform/auc", data=self.df, ax=axes[0, 1])
        axes[0, 1].set_title("Link Prediction AUC by Dimension")
        axes[0, 1].set_xlabel("Embedding Dimension")
        axes[0, 1].set_ylabel("AUC")

            
        # Plot 3: Hard link prediction by dimension
        sns.boxplot(x="dim", y="lp_hard/auc", data=self.df, ax=axes[0, 2])
        axes[0, 2].set_title("Hard Link Prediction AUC by Dimension")
        axes[0, 2].set_xlabel("Embedding Dimension")
        axes[0, 2].set_ylabel("AUC")

        
        plt.tight_layout()
        return self.save_and_return(fig, "performance_by_dimension")
    
    def visualize_task_comparison(self):
        """Compare node classification and link prediction performance."""

        
        fig, ax = plt.subplots(figsize=self.figsize)
        

            
        dim_metrics = self.df.groupby('dim')[self.metrics].mean().reset_index()
        
        # Plot comparison
        x = np.arange(len(dim_metrics))
        width = 0.8 / len(self.metrics)  # Adjust bar width based on number of metrics
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, orange, green
        labels = ['Node Classification', 'Link Prediction', 'Hard Link Prediction']
        
        for i, metric in enumerate(self.metrics):
            pos = x + width * (i - (len(self.metrics) - 1) / 2)
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

        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Calculate average metrics by dimension
        dim_metrics = self.df.groupby('dim')[['lp_uniform/auc', 'lp_hard/auc']].mean().reset_index()
        
        # Create line plot with markers
        ax.plot(dim_metrics['dim'], dim_metrics['lp_uniform/auc'], 'o-', label='Standard LP', linewidth=2, color='#ff7f0e')
        ax.plot(dim_metrics['dim'], dim_metrics['lp_hard/auc'], 's-', label='Hard LP', linewidth=2, color='#2ca02c')
        
        # Add value labels
        for i, row in dim_metrics.iterrows():
            ax.text(row['dim'], row['lp_uniform/auc'] + 0.01, f'{row["lp_uniform/auc"]:.3f}', 
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
        

    
    def run(self):
        """Run all available visualizations for Node2Vec analysis."""
        results = super().run()
        results["performance_by_dimension"] = self.visualize()
        results["task_comparison"] = self.visualize_task_comparison()
        results["lp_comparison"] = self.visualize_lp_comparison()

        
        return results

