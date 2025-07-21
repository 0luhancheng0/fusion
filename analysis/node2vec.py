from .base import AbstractAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Node2VecAnalyzer(AbstractAnalyzer):
    """Analyzer for Node2Vec embeddings results."""
    
    def __init__(self, dpi=300, cmap="viridis", figsize=(6.4, 4.8)):
        super().__init__("/home/lcheng/oz318/fusion/logs/Node2VecLightning", dpi, cmap, figsize)
        self.post_process()
        self._backup_df = self._df.copy()
    def post_process(self):
        if not self._df.empty:
            self._df['dim'] = self._df['path'].apply(lambda p: int(p.split('/')[-2]))
            self._df['seed'] = self._df['path'].apply(lambda p: int(p.split('/')[-1]))
    
    def analyze(self):
        result = (
            self._df.groupby(["dim"])
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
        sns.boxplot(x="dim", y="acc/test", data=self._df, ax=axes[0, 0])
        axes[0, 0].set_title("Node Classification Accuracy by Dimension")
        axes[0, 0].set_xlabel("Embedding Dimension")
        axes[0, 0].set_ylabel("Test Accuracy")
        
        # Plot 2: Standard link prediction by dimension
        sns.boxplot(x="dim", y="lp_uniform/auc", data=self._df, ax=axes[0, 1])
        axes[0, 1].set_title("Link Prediction AUC by Dimension")
        axes[0, 1].set_xlabel("Embedding Dimension")
        axes[0, 1].set_ylabel("AUC")

            
        # Plot 3: Hard link prediction by dimension
        sns.boxplot(x="dim", y="lp_hard/auc", data=self._df, ax=axes[0, 2])
        axes[0, 2].set_title("Hard Link Prediction AUC by Dimension")
        axes[0, 2].set_xlabel("Embedding Dimension")
        axes[0, 2].set_ylabel("AUC")

        
        plt.tight_layout()
        return self.save_and_return(fig, "performance_by_dimension")
    
    def visualize_task_comparison(self):
        """Compare node classification and link prediction performance."""

        
        fig, ax = plt.subplots(figsize=self.figsize)
        

            
        dim_metrics = self._df.groupby('dim')[self.metrics].mean().reset_index()
        
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
    
    def scatter(self):
        """Compare performance across metrics and dimensions."""

        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Calculate average metrics by dimension
        dim_metrics = self._df.groupby('dim')[self.metrics].mean().reset_index()
        
        # Colors and markers for different metric types
        colors = {'acc': '#1f77b4', 'uniform': '#ff7f0e', 'hard': '#2ca02c'}
        markers = {'acc': '^', 'uniform': 'o', 'hard': 's'}
        labels = {'acc': 'Node Classification', 'uniform': 'Standard LP', 'hard': 'Hard LP'}
        
        # Create line plot with markers for each metric
        for metric in self.metrics:
            if '/' in metric:
                metric_type = metric.split('/')[0]
                if metric_type.startswith('lp_'):
                    metric_type = metric_type.split('_')[1]  # Extract 'uniform' or 'hard'
            else:
                metric_type = 'acc'
                
            color = colors.get(metric_type, 'gray')
            marker = markers.get(metric_type, 'x')
            label = labels.get(metric_type, metric)
            
            ax.plot(dim_metrics['dim'], dim_metrics[metric], f'{marker}-', 
                   label=label, linewidth=2, color=color)
            
            # Add value labels
            for i, row in dim_metrics.iterrows():
                # Adjust offset based on metric type to avoid overlapping
                if metric_type == 'acc':
                    offset = 0.015
                elif metric_type == 'uniform':
                    offset = 0.01
                else:
                    offset = -0.02
                    
                ax.text(row['dim'], row[metric] + offset, f'{row[metric]:.3f}', 
                       ha='center', fontsize=9, color=color)
        
        # Set x-axis to log2 scale
        ax.set_xscale('log', base=2)
        
        ax.set_xlabel('Embedding Dimension (log₂ scale)')
        ax.set_ylabel('Performance Score')
        ax.set_title('Performance Comparison Across Tasks')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self.save_and_return(fig, "performance_comparison")

    def run(self):
        """Run all available visualizations for Node2Vec analysis."""
        results = {}
        results["performance_by_dimension"] = self.performance_by_dimension()
        results["task_comparison"] = self.visualize_task_comparison()
        results["scatter"] = self.scatter()        
        return results

