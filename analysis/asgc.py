from pathlib import Path
import torch
from .base import AbstractAnalyzer
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

class ASGCAnalyzer(AbstractAnalyzer):
    def __init__(self, dpi=300, cmap="viridis", figsize=(6.4, 4.8), remove_outliers=False, outlier_params=None):
        super().__init__("/home/lcheng/oz318/fusion/logs/ASGC", dpi, cmap, figsize, 
                         remove_outliers=remove_outliers, outlier_params=outlier_params)

    def post_process(self):
        """Extract dimension, regularization, and k values from the directory structure."""
        # Extract dim, reg, and k from the directory structure
        # Paths look like: /home/lcheng/oz318/fusion/logs/ASGC/32/0.1/8/
        if not self.df.empty:
            # Extract values from path
            self.df['dim'] = self.df['path'].apply(lambda p: int(p.split('/')[-3]))
            self.df['reg'] = self.df['path'].apply(lambda p: float(p.split('/')[-2]))
            self.df['k'] = self.df['path'].apply(lambda p: int(p.split('/')[-1]))
            
            # If prefix column exists, drop it (may not be needed in ASGC)
            if 'prefix' in self.df.columns:
                self.df = self.df.drop(columns=["prefix"])

    def analyze(self):
        """Analyze ASGC results by grouping by dim, k and reg parameters."""
        if self.df.empty:
            print("No data to analyze.")
            return None

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
    
    def visualize(self):
        """Visualize ASGC results as a heatmap of test accuracy and link prediction vs k and reg for each dimension."""
        if self.df.empty:
            print("No data to visualize.")
            return plt.figure()

        # Get unique dimensions
        dimensions = self.df["dim"].unique()

        if len(dimensions) > 1:
            # Create a figure with a grid for different dimensions (rows) and metrics (columns)
            n_dims = len(dimensions)
            n_cols = 2  # One for accuracy, one for link prediction
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
                
                lp_pivot = (
                    dim_df.groupby(["k", "reg"])
                    .agg({"lp_uniform/auc": "mean"})
                    .reset_index()
                    .pivot(index="k", columns="reg", values="lp_uniform/auc")
                )

                # Plot accuracy heatmap in first column
                ax1 = axes[i, 0]
                sns.heatmap(acc_pivot, cmap=self.cmap, annot=True, fmt=".3f", ax=ax1)
                ax1.set_title(f"Test Accuracy (dim={dim})")
                ax1.set_xlabel("Regularization")
                ax1.set_ylabel("Number of Hops (k)")
                
                # Plot link prediction heatmap in second column
                ax2 = axes[i, 1]
                sns.heatmap(lp_pivot, cmap=self.cmap, annot=True, fmt=".3f", ax=ax2)
                ax2.set_title(f"Link Prediction AUC (dim={dim})")
                ax2.set_xlabel("Regularization")
                ax2.set_ylabel("Number of Hops (k)")

            plt.tight_layout()
            return self.save_and_return(fig, "heatmap_combined")

        else:
            # Single dimension case
            dim = dimensions[0]
            fig, axes = plt.subplots(1, 2, figsize=(self.figsize[0] * 2.2, self.figsize[1]))
            
            # Test accuracy heatmap
            acc_pivot = (
                self.df.groupby(["k", "reg"])
                .agg({"acc/test": "mean"})
                .reset_index()
                .pivot(index="k", columns="reg", values="acc/test")
            )
            
            # Link prediction heatmap
            lp_pivot = (
                self.df.groupby(["k", "reg"])
                .agg({"lp_uniform/auc": "mean"})
                .reset_index()
                .pivot(index="k", columns="reg", values="lp_uniform/auc")
            )

            # Plot accuracy heatmap
            sns.heatmap(acc_pivot, cmap=self.cmap, annot=True, fmt=".3f", ax=axes[0])
            axes[0].set_title(f"Test Accuracy (dim={dim})")
            axes[0].set_xlabel("Regularization")
            axes[0].set_ylabel("Number of Hops (k)")
            
            # Plot link prediction heatmap
            sns.heatmap(lp_pivot, cmap=self.cmap, annot=True, fmt=".3f", ax=axes[1])  # Different colormap for distinction
            axes[1].set_title(f"Link Prediction AUC (dim={dim})")
            axes[1].set_xlabel("Regularization")
            axes[1].set_ylabel("Number of Hops (k)")
            
            plt.tight_layout()
            return self.save_and_return(fig, f"heatmap_combined_dim{dim}")

    def visualize_parameter_impact(self, dim=None):
        """
        Visualize the impact of k and regularization on performance.
        
        Args:
            dim: If provided, only show this dimension, otherwise create plots for all dimensions
        """
        if self.df.empty:
            print("No data to visualize.")
            return plt.figure()
            
        dimensions = [dim] if dim is not None else sorted(self.df['dim'].unique())
        
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
        if dim is not None:
            return self.save_and_return(fig, f"parameter_impact_dim{dim}")
        else:
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

    def visualize_coefficients(self, path_or_index):
        """Visualize the learned coefficients for a specific experiment."""
        coefficients = self.load_coefficients_to_cpu(path_or_index)
        if coefficients is None:
            return plt.figure()
            
        # Get index if a path was provided
        if not isinstance(path_or_index, int):
            # Find the index with matching path
            matches = self.df[self.df['path'] == str(Path(path_or_index).parent)]
            if matches.empty:
                index = None
                title_info = "unknown"
                filename = f"coefficients_unknown"
            else:
                index = matches.index[0]
                dim = self.df.loc[index, 'dim']
                k = self.df.loc[index, 'k']
                reg = self.df.loc[index, 'reg']
                title_info = f"dim={dim}, k={k}, reg={reg}"
                filename = f"coefficients_dim{dim}_k{k}_reg{reg}"
        else:
            index = path_or_index
            dim = self.df.iloc[index]['dim']
            k = self.df.iloc[index]['k']
            reg = self.df.iloc[index]['reg'] 
            title_info = f"dim={dim}, k={k}, reg={reg}"
            filename = f"coefficients_dim{dim}_k{k}_reg{reg}"
        
        # Plot the coefficients
        fig, ax = plt.subplots(figsize=(10, 6))
        num_coeffs = coefficients.shape[0]
        
        # If there are too many features, sample or aggregate
        if coefficients.shape[1] > 20:
            # Option 1: Sample a subset of features
            sample_size = min(20, coefficients.shape[1])
            sampled_indices = torch.randperm(coefficients.shape[1])[:sample_size]
            coeffs_to_plot = coefficients[:, sampled_indices]  # Already on CPU
            
            for i in range(sample_size):
                ax.plot(range(num_coeffs), coeffs_to_plot[:, i], alpha=0.6, marker='o')
            
            # Also plot the mean coefficient
            mean_coeff = coefficients.mean(dim=1)  # Already on CPU
            ax.plot(range(num_coeffs), mean_coeff, 'k-', linewidth=2, label='Mean')
        else:
            # Plot all features
            for i in range(coefficients.shape[1]):
                ax.plot(range(num_coeffs), coefficients[:, i], alpha=0.6, marker='o')
        
        ax.set_title(f'ASGC Coefficients ({title_info})')
        ax.set_xlabel('Hop (k)')
        ax.set_ylabel('Coefficient Value')
        ax.grid(True, alpha=0.3)
        
        if coefficients.shape[1] > 20:
            ax.legend()
            
        return self.save_and_return(fig, filename)

    def visualize_coefficient_heatmaps(self, dim=None, k_values=None, reg_values=None):
        """
        Visualize ASGC coefficients using heatmaps for different experiments.
        
        Args:
            dim: If provided, only show this dimension, otherwise select a representative dimension
            k_values: List of k values to include, if None select representative ones
            reg_values: List of regularization values to include, if None select representative ones
            
        Returns:
            matplotlib figure showing coefficient heatmaps
        """
        if self.df.empty:
            print("No data to visualize.")
            return plt.figure()
            
        # Filter by dimension if provided
        if dim is not None:
            df_filtered = self.df[self.df['dim'] == dim].copy()
        else:
            # Use the first dimension if there are multiple
            dims = sorted(self.df['dim'].unique())
            if dims:
                df_filtered = self.df[self.df['dim'] == dims[0]].copy()
                dim = dims[0]  # Store the selected dimension
            else:
                df_filtered = self.df.copy()
            
        if df_filtered.empty:
            print(f"No data for dimension {dim}.")
            return plt.figure()
        
        # Filter by k and reg values if provided or select representative ones
        if k_values is None:
            all_k = sorted(df_filtered['k'].unique())
            # Select a representative set (up to 4) across the range
            if len(all_k) > 4:
                indices = np.linspace(0, len(all_k) - 1, 4, dtype=int)
                k_values = [all_k[i] for i in indices]
            else:
                k_values = all_k
                
        if reg_values is None:
            all_reg = sorted(df_filtered['reg'].unique())
            # Select a representative set (up to 4) across the range
            if len(all_reg) > 4:
                indices = np.linspace(0, len(all_reg) - 1, 4, dtype=int)
                reg_values = [all_reg[i] for i in indices]
            else:
                reg_values = all_reg
        
        # Filter dataframe by selected k and reg values
        df_filtered = df_filtered[df_filtered['k'].isin(k_values) & df_filtered['reg'].isin(reg_values)]
        
        if df_filtered.empty:
            print("No data for the selected parameters.")
            return plt.figure()
        
        # Create a grid of plots - one for each combination of k and regularization
        n_rows = len(reg_values)
        n_cols = len(k_values)
        
        fig, axes = plt.subplots(n_rows, n_cols, 
                                figsize=(3.5 * n_cols, 3 * n_rows),
                                squeeze=False)
                                
        # Title for the entire figure
        if dim is not None:
            fig.suptitle(f'ASGC Coefficients Heatmaps (dim={dim})', fontsize=16, y=0.98)
        else:
            fig.suptitle(f'ASGC Coefficients Heatmaps', fontsize=16, y=0.98)
        
        # For each combination of regularization (rows) and k (columns)
        for i, reg in enumerate(reg_values):
            for j, k in enumerate(k_values):
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
        if dim is not None:
            filename = f"coefficient_heatmaps_dim{dim}"
        else:
            filename = f"coefficient_heatmaps"
            
        return self.save_and_return(fig, filename)

    def run(self):
        """Run all available visualizations for ASGC analysis."""
        results = super().run()
        
        # Add ASGC-specific visualizations
        try:
            results["parameter_impact"] = self.visualize_parameter_impact()
            print("Parameter impact visualization created.")
        except Exception as e:
            print(f"Error creating parameter impact visualization: {e}")
            
        try:
            results["coefficient_heatmaps"] = self.visualize_coefficient_heatmaps()
            print("Coefficient heatmaps created.")
        except Exception as e:
            print(f"Error creating coefficient heatmaps: {e}")
            
        return results

    def visualize_hyperparameter_sensitivity(self, param_name, metric='acc/test'):
        """
        Visualize the sensitivity of models to a specific hyperparameter.
        
        Args:
            param_name: Name of the hyperparameter to analyze (e.g., 'latent_dim', 'k', 'reg')
            metric: Performance metric to track
        """
        if self.df.empty or param_name not in self.df.columns or metric not in self.df.columns:
            print(f"Missing required parameter '{param_name}' or metric '{metric}'")
            return plt.figure()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate mean and standard deviation for each parameter value
        param_stats = self.df.groupby(param_name)[metric].agg(['mean', 'std']).reset_index()
        
        # Sort by parameter value for better visualization
        param_stats = param_stats.sort_values(param_name)
        
        # Plot the mean with error bars showing standard deviation
        ax.errorbar(
            x=param_stats[param_name], 
            y=param_stats['mean'], 
            yerr=param_stats['std'],
            fmt='o-', 
            capsize=5, 
            linewidth=2,
            markersize=8
        )
        
        # Add value labels
        for x, y, std in zip(param_stats[param_name], param_stats['mean'], param_stats['std']):
            ax.annotate(
                f'{y:.3f}±{std:.3f}',
                (x, y),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                fontsize=8
            )
        
        ax.set_xlabel(f'Hyperparameter: {param_name}')
        ax.set_ylabel(f'Performance: {metric}')
        ax.set_title(f'Sensitivity Analysis for {param_name}')
        ax.grid(True, alpha=0.3)
        
        return self.save_and_return(fig, f"sensitivity_{param_name}")
