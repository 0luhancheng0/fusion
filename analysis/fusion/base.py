from ..base import AbstractAnalyzer

import seaborn as sns
import matplotlib.pyplot as plt


class FusionAnalyzer(AbstractAnalyzer):
    """Base class for analyzing fusion experiments."""

    def __init__(self, experiment_type, dpi=300, cmap="viridis", figsize=(6.4, 4.8), 
                 remove_outliers=False, outlier_params=None):
        self.experiment_type = experiment_type
        base_path = f"/home/lcheng/oz318/fusion/logs/{experiment_type}"
        super().__init__(base_path, dpi, cmap, figsize, 
                        remove_outliers=remove_outliers, outlier_params=outlier_params)
    def post_process(self):
        self.df[["textual_name", "relational_name", "textual_dim", "relational_dim", "latent_dim"]] = self.df.prefix.str.split(
            "[/_]"
        ).tolist()
        self.df = self.df.drop(columns=["prefix"])
        
    def analyze(self):
        """Analyze fusion results by grouping by relevant parameters."""
        if self.df.empty:
            print("No data to analyze.")
            return None


        result = (
            self.df.groupby(
                [
                    "textual_name",
                    "relational_name",
                    "textual_dim",
                    "relational_dim",
                    "latent_dim",
                ]
            )
            .agg({"acc/test": ["mean", "std", "count"]})
            .reset_index()
        )

        # Format column names
        result.columns = [
            "_".join(col).strip() if col[1] else col[0] for col in result.columns.values
        ]
        return result

    def visualize(self, x_axis='textual_name', y_axis='relational_name', filter_params=None, figsize=None):
        """
        Create heatmap visualizations with configurable x and y axes for all available metrics.
        
        Args:
            x_axis: Parameter to use for the x-axis
            y_axis: Parameter to use for the y-axis
            filter_params: Dictionary of parameters to filter the data (e.g., {"latent_dim": 128})
            figsize: Custom figure size, defaults to self.figsize if None
            
        Returns:
            matplotlib figure with heatmaps for test accuracy and link prediction metrics
        """
        if self.df.empty:
            print("No data to visualize.")
            return plt.figure()
            
        # Filter data if filter parameters are provided
        plot_df = self.df.copy()
        if filter_params:
            for param, value in filter_params.items():
                plot_df = plot_df[plot_df[param] == value]
                
        if plot_df.empty:
            print("No data matches the filter parameters.")
            return plt.figure()
        
        # Determine which metrics are available
        metrics = ["acc/test"]
        if "lp_uniform/auc" in plot_df.columns:
            metrics.append("lp_uniform/auc")
        if "lp_hard/auc" in plot_df.columns:
            metrics.append("lp_hard/auc")
            
        # Create a subplot for each metric - make individual heatmaps skinnier
        n_metrics = len(metrics)
        
        # Calculate dimensions to make skinnier heatmaps
        single_width = 3.5  # Reduced width for each heatmap
        single_height = 5.0  # Taller height to maintain proportions
        
        # Create figure with adjusted dimensions
        fig, axes = plt.subplots(1, n_metrics, 
                               figsize=(single_width * n_metrics, 
                                       figsize[1] if figsize else single_height),
                               squeeze=False)
        
        # Create heatmap for each metric
        for i, metric in enumerate(metrics):
            ax = axes[0, i]
            
            # Create pivot table for this metric
            try:
                pivot_data = (
                    plot_df.groupby([x_axis, y_axis])
                    .agg({metric: "mean"})
                    .reset_index()
                    .pivot(index=x_axis, columns=y_axis, values=metric)
                )
                
                # Create heatmap - using the same colormap (self.cmap) for all metrics
                sns.heatmap(pivot_data, cmap=self.cmap, annot=True, fmt=".3f", ax=ax)
                
                # Set title and labels
                metric_name = metric.replace('/', ' ').replace('_', ' ').title()
                ax.set_title(f"{self.experiment_type}: {metric_name}")
                ax.set_xlabel(y_axis.replace('_', ' ').title())
                if i == 0:
                    ax.set_ylabel(x_axis.replace('_', ' ').title())
                else:
                    ax.set_ylabel('')  # Only show y-label on first subplot
            
            except Exception as e:
                ax.text(0.5, 0.5, f"Could not create heatmap\nfor {metric}\n{str(e)}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
        
        # Add common title if filter parameters are provided
        if filter_params:
            filter_str = ", ".join([f"{k}={v}" for k, v in filter_params.items()])
            plt.suptitle(f"{self.experiment_type} Performance ({filter_str})", fontsize=14, y=0.98)
            plt.subplots_adjust(top=0.85)  # Make room for the suptitle
        
        # Create filename from parameters
        filename = f"{self.experiment_type}_{x_axis}_vs_{y_axis}"
        if filter_params:
            filename += "_" + "_".join([f"{k}-{v}" for k, v in filter_params.items()])
        
        # Adjust spacing between subplots
        plt.tight_layout()
        fig.subplots_adjust(wspace=0.4)  # Add more horizontal space between heatmaps
        
        return self.save_and_return(fig, filename)

    def visualize_dimension_impact(self):
        """
        Visualize the impact of different dimensions on model performance.
        
        Returns:
            matplotlib figure showing performance variation across dimensions
        """
        if self.df.empty:
            print("No data to visualize dimensions impact.")
            return plt.figure()
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), squeeze=False)
        
        # Plot 1: Impact of latent dimension on test accuracy
        if 'latent_dim' in self.df.columns:
            ax1 = axes[0, 0]
            sns.boxplot(x='latent_dim', y='acc/test', data=self.df, ax=ax1)
            ax1.set_title(f'{self.experiment_type}: Test Accuracy by Latent Dimension')
            ax1.set_xlabel('Latent Dimension')
            ax1.set_ylabel('Test Accuracy')
            
        # Plot 2: Impact of textual dimension on test accuracy
        if 'textual_dim' in self.df.columns:
            ax2 = axes[0, 1]
            sns.boxplot(x='textual_dim', y='acc/test', data=self.df, ax=ax2)
            ax2.set_title(f'{self.experiment_type}: Test Accuracy by Textual Dimension')
            ax2.set_xlabel('Textual Dimension')
            ax2.set_ylabel('Test Accuracy')
            
        # Plot 3: Impact of relational dimension on test accuracy
        if 'relational_dim' in self.df.columns:
            ax3 = axes[1, 0]
            sns.boxplot(x='relational_dim', y='acc/test', data=self.df, ax=ax3)
            ax3.set_title(f'{self.experiment_type}: Test Accuracy by Relational Dimension')
            ax3.set_xlabel('Relational Dimension')
            ax3.set_ylabel('Test Accuracy')
            
        # Plot 4: Interaction between textual and relational dimensions
        if 'textual_dim' in self.df.columns and 'relational_dim' in self.df.columns:
            ax4 = axes[1, 1]
            dim_metrics = self.df.groupby(['textual_dim', 'relational_dim'])['acc/test'].mean().reset_index()
            pivot_data = dim_metrics.pivot(index='relational_dim', columns='textual_dim', values='acc/test')
            sns.heatmap(pivot_data, annot=True, fmt=".3f", cmap=self.cmap, ax=ax4)
            ax4.set_title(f'{self.experiment_type}: Accuracy by Dimension Interaction')
            ax4.set_xlabel('Textual Dimension')
            ax4.set_ylabel('Relational Dimension')
            
        plt.tight_layout()
        return self.save_and_return(fig, "dimension_impact")
            
    def run(self):
        """Run all available visualizations for fusion analysis."""
        results = super().run()
        
        # Add dimension impact visualization
        try:
            results["dimension_impact"] = self.visualize_dimension_impact()
            print("Dimension impact visualization created.")
        except Exception as e:
            print(f"Error creating dimension impact visualization: {e}")
        
        return results
