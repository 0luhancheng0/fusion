from pathlib import Path
import torch
from .base import AbstractAnalyzer
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

class ASGCAnalyzer(AbstractAnalyzer):
    def __init__(self, dpi=300, cmap="viridis", figsize=(6.4, 4.8)):
        super().__init__("/home/lcheng/oz318/fusion/logs/ASGC", dpi, cmap, figsize)

    def post_process(self):
        self.df = self.df.drop("prefix", axis=1)
        self.df['dim'] = self.df['path'].apply(lambda p: int(p.split('/')[-3]))
        self.df['reg'] = self.df['path'].apply(lambda p: float(p.split('/')[-2]))
        self.df['k'] = self.df['path'].apply(lambda p: int(p.split('/')[-1]))
        

    def analyze(self):
        """Analyze ASGC results by grouping by dim, k and reg parameters."""

        # Group by dimension, k, and regularization
        result = (
            self.df.groupby(["dim", "k", "reg"])
            .agg({
                "acc/valid": ["mean", "std"],  # Use acc/valid if that's your key
                "acc/test": ["mean", "std"],
                "lp_uniform/auc": ["mean", "std"]
            })
            .reset_index()
        )

        # Format for better readability
        result.columns = ["_".join(col).strip() for col in result.columns.values]
        return result
    
    def heatmap(self):
        """Visualize ASGC results as a heatmap of test accuracy and link prediction vs k and reg for each dimension."""

        dimensions = self.df["dim"].unique()
        n_dims = len(dimensions)
        n_cols = 3  #
        n_rows = n_dims
        
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(self.figsize[0] * n_cols * 1.2, self.figsize[1] * n_rows),
            squeeze=False,
        )

        for i, dim in enumerate(sorted(dimensions)):
            # Filter data for this dimension
            dim_df = self.df[self.df["dim"] == dim]

            # Create pivot tables for both metrics
            acc_pivot = (
                dim_df.groupby(["k", "reg"])
                .agg({"acc/test": "mean"})
                .reset_index()
                .pivot(index="k", columns="reg", values="acc/test")
            )
            
            lpuniform_pivot = (
                dim_df.groupby(["k", "reg"])
                .agg({"lp_uniform/auc": "mean"})
                .reset_index()
                .pivot(index="k", columns="reg", values="lp_uniform/auc")
            )
            
            lphard_pivot = (
                dim_df.groupby(["k", "reg"])
                .agg({"lp_hard/auc": "mean"})
                .reset_index()
                .pivot(index="k", columns="reg", values="lp_hard/auc")
            )
            # Plot accuracy heatmap in first column
            ax1 = axes[i, 0]
            sns.heatmap(acc_pivot, cmap=self.cmap, annot=True, fmt=".3f", ax=ax1)
            ax1.set_title(f"Test Accuracy (dim={dim})")
            ax1.set_xlabel("Regularization")
            ax1.set_ylabel("Number of Hops (k)")
            
            # Plot link prediction heatmap in second column
            ax2 = axes[i, 1]
            sns.heatmap(lpuniform_pivot, cmap=self.cmap, annot=True, fmt=".3f", ax=ax2)
            ax2.set_title(f"Uniform Link Prediction AUC (dim={dim})")
            ax2.set_xlabel("Regularization")
            ax2.set_ylabel("Number of Hops (k)")
            
            ax3 = axes[i, 2]
            sns.heatmap(lphard_pivot, cmap=self.cmap, annot=True, fmt=".3f", ax=ax3)
            ax3.set_title(f"Hard Link Prediction AUC (dim={dim})")
            ax3.set_xlabel("Regularization")
            ax3.set_ylabel("Number of Hops (k)")

        plt.tight_layout()
        return self.save_and_return(fig, "heatmap_combined")


    def visualize_parameter_impact(self):

        dimensions = sorted(self.df['dim'].unique())
        
        # Create a 2x1 subplot for each dimension
        fig, axes = plt.subplots(2, len(dimensions), figsize=(6*len(dimensions), 10), squeeze=False)
        
        for i, dim_value in enumerate(dimensions):
            dim_df = self.df[self.df['dim'] == dim_value]
            
            # Plot 1: Effect of k on test accuracy
            ax1 = axes[0, i]
            sns.boxplot(x='k', y='acc/test', data=dim_df, ax=ax1)
            ax1.set_title(f'Effect of k on Test Accuracy (dim={dim_value})')
            ax1.set_xlabel('Number of Hops (k)')
            ax1.set_ylabel('Test Accuracy')
            
            # Plot 2: Effect of regularization on test accuracy
            ax2 = axes[1, i]
            sns.boxplot(x='reg', y='acc/test', data=dim_df, ax=ax2)
            ax2.set_title(f'Effect of Regularization on Test Accuracy (dim={dim_value})')
            ax2.set_xlabel('Regularization Parameter')
            ax2.set_ylabel('Test Accuracy')
            ax2.set_xticklabels([f'{x:.2f}' for x in sorted(dim_df['reg'].unique())])
            
        plt.tight_layout()
        # Save figure with appropriate filename

        return self.save_and_return(fig, "parameter_impact")

    def load_coefficients_to_cpu(self, path_or_index):
        """
        Load coefficients from a file path or by index in the DataFrame, ensuring they're loaded to CPU.
        
        Args:
            path_or_index: Either a file path or an index in the DataFrame
            
        Returns:
            Tensor of coefficients on CPU or None if file doesn't exist
        """
        if isinstance(path_or_index, int):
            path = Path(self.df.iloc[path_or_index]['path']) / 'coefficients.pt'
        else:
            path = path_or_index
            
        if not Path(path).exists():
            print(f"Coefficients file not found: {path}")
            return None
            
        return torch.load(path, map_location="cpu")
    
    def load_coefficients(self, path_or_index):
        """Load coefficients from a file path or by index in the DataFrame."""
        # Maintain backwards compatibility by calling the new method
        return self.load_coefficients_to_cpu(path_or_index)


    def visualize_coefficient_heatmaps(self, dim=None):
        """
        Visualize ASGC coefficients using heatmaps for different experiments.
        
        Args:
            dim: If provided, only show this dimension, otherwise visualize all dimensions
            k_values: List of k values to include, if None select representative ones
            reg_values: List of regularization values to include, if None select representative ones
            
        Returns:
            If dim is provided: matplotlib figure showing coefficient heatmaps for that dimension
            If dim is None: dictionary of figures for each dimension
        """
            
        # If dim is provided, only create a single figure for that dimension
        # Otherwise, create figures for all available dimensions
        if dim is not None:
            dims_to_visualize = [dim]
            return_single = True
        else:
            dims_to_visualize = sorted(self.df['dim'].unique())
            return_single = False
            
        results = {}
        
        for current_dim in dims_to_visualize:
            # Filter by dimension
            df_filtered = self.df[self.df['dim'] == current_dim].copy()
            
            if df_filtered.empty:
                print(f"No data for dimension {current_dim}.")
                continue
            
            # Filter by k and reg values if provided or select representative ones

            all_k = sorted(df_filtered['k'].unique())
            # Select a representative set (up to 4) across the range
            if len(all_k) > 4:
                indices = np.linspace(0, len(all_k) - 1, 4, dtype=int)
                k_vals = [all_k[i] for i in indices]
            else:
                k_vals = all_k


            all_reg = sorted(df_filtered['reg'].unique())
            # Select a representative set (up to 4) across the range
            if len(all_reg) > 4:
                indices = np.linspace(0, len(all_reg) - 1, 4, dtype=int)
                reg_vals = [all_reg[i] for i in indices]
            else:
                reg_vals = all_reg

            
            # Filter dataframe by selected k and reg values
            df_filtered = df_filtered[df_filtered['k'].isin(k_vals) & df_filtered['reg'].isin(reg_vals)]
            
            # Create a grid of plots - one for each combination of k and regularization
            n_rows = len(reg_vals)
            n_cols = len(k_vals)
            
            fig, axes = plt.subplots(n_rows, n_cols, 
                                    figsize=(3.5 * n_cols, 3 * n_rows),
                                    squeeze=False)
                                    
            # Title for the entire figure
            fig.suptitle(f'ASGC Coefficients Heatmaps (dim={current_dim})', fontsize=16, y=0.98)
            
            # For each combination of regularization (rows) and k (columns)
            for i, reg in enumerate(reg_vals):
                for j, k in enumerate(k_vals):
                    # Find matching experiment
                    matches = df_filtered[(df_filtered['reg'] == reg) & (df_filtered['k'] == k)]
                    
                    ax = axes[i, j]
                    
                    if not matches.empty:
                        # Load coefficients for this experiment
                        coeff_path = os.path.join(matches.iloc[0]['path'], 'coefficients.pt')
                        
                        if os.path.exists(coeff_path):
                            # Use the new method to ensure coefficients are loaded to CPU
                            coeffs = self.load_coefficients_to_cpu(coeff_path)
                            coeffs = coeffs.numpy()  # Convert to NumPy array
                            
                            # For visualization, restrict to a manageable number of features
                            if coeffs.shape[1] > 100:
                                # Sample features randomly
                                sample_indices = np.random.choice(coeffs.shape[1], 100, replace=False)
                                coeffs_sample = coeffs[:, sample_indices]
                            else:
                                coeffs_sample = coeffs
                                
                            # Create heatmap
                            im = ax.imshow(coeffs_sample, aspect='auto', cmap='viridis')
                            ax.set_title(f'reg={reg}, k={k}')
                            ax.set_xlabel('Features (sampled)')
                            ax.set_ylabel('Hop Index')
                            
                            # Add colorbar
                            plt.colorbar(im, ax=ax, shrink=0.8)
                        else:
                            # No coefficients file
                            ax.text(0.5, 0.5, f"No coefficients\nfor reg={reg}, k={k}", 
                                    ha='center', va='center', transform=ax.transAxes)
                            ax.set_xticks([])
                            ax.set_yticks([])
                    else:
                        # No matching experiment
                        ax.text(0.5, 0.5, f"No experiment\nfor reg={reg}, k={k}", 
                                ha='center', va='center', transform=ax.transAxes)
                        ax.set_xticks([])
                        ax.set_yticks([])
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle
            
            # Generate an appropriate filename
            filename = f"coefficient_heatmaps_dim{current_dim}"
            
            # Save the figure and add to results
            results[f"coefficient_heatmaps_dim_{current_dim}"] = self.save_and_return(fig, filename)
        
        # If a specific dimension was requested, return just that figure
        # Otherwise return the dictionary of all figures
        if return_single and results:
            return results[dims_to_visualize[0]]
        return results

    def run(self):
        """Run all available visualizations for ASGC analysis."""
        results = {}
        results["parameter_impact"] = self.visualize_parameter_impact()
        results.update(self.visualize_coefficient_heatmaps())
        results["hyperparameter_sensitivity"] = self.visualize_hyperparameter_sensitivity()
        return results

    def visualize_hyperparameter_sensitivity(self):
        """
        Visualize the sensitivity of models to hyperparameters (k, dim, reg) 
        for three metrics (acc/test, lp_uniform/auc, lp_hard/auc).
        Creates a 3x3 grid of subplots with appropriate log scales.
        """
        # Define parameters and metrics
        params = ['k', 'dim', 'reg']
        param_labels = {'k': 'Number of Hops (k)', 'dim': 'Dimension', 'reg': 'Regularization'}

        metric_labels = {
            'acc/test': 'Test Accuracy', 
            'lp_uniform/auc': 'Uniform Link Prediction AUC', 
            'lp_hard/auc': 'Hard Link Prediction AUC'
        }
        
        # Create figure with 3x3 grid
        fig, axes = plt.subplots(len(params), len(self.metrics), figsize=(15, 12))
        
        # Add figure title
        fig.suptitle('Hyperparameter Sensitivity Analysis', fontsize=16)
        
        # For each parameter (rows)
        for i, param in enumerate(params):
            # For each metric (columns)
            for j, metric in enumerate(self.metrics):
                ax = axes[i, j]
                
                # Calculate statistics
                param_stats = self.df.groupby(param)[metric].agg(['mean', 'std']).reset_index()
                param_stats = param_stats.sort_values(param)
                
                # Plot
                ax.errorbar(
                    x=param_stats[param], 
                    y=param_stats['mean'], 
                    yerr=param_stats['std'],
                    fmt='o-', 
                    capsize=5, 
                    linewidth=2,
                    markersize=8
                )
                
                # Add value labels
                for x, y, std in zip(param_stats[param], param_stats['mean'], param_stats['std']):
                    ax.annotate(
                        f'{y:.3f}±{std:.3f}',
                        (x, y),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        fontsize=8
                    )
                
                # Set scales based on parameter
                if param in ['k', 'dim']:
                    ax.set_xscale('log', base=2)
                elif param == 'reg':
                    ax.set_xscale('log', base=10)
                
                # Set column titles (metrics) for top row only
                if i == 0:
                    ax.set_title(metric_labels[metric])
                
                # Set x-axis label for bottom row only
                if i == len(params) - 1:
                    ax.set_xlabel(param_labels[param])
                
                # Set y-axis label for leftmost column only
                if j == 0:
                    # Add parameter name as ylabel
                    ax.set_ylabel(f"{param_labels[param]}\nPerformance")
                
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
        return self.save_and_return(fig, "hyperparameter_sensitivity")
