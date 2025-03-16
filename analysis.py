from pathlib import Path
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import constants
import numpy as np
import json
from pathlib import Path
from abc import ABC, abstractmethod
import os

class AbstractAnalyzer(ABC):
    def __init__(self, base_path, dpi=300, cmap="viridis", figsize=(6.4, 4.8), remove_outliers=False, outlier_params=None):
        self.base_path = base_path
        self.dpi = dpi
        self.figsize = figsize
        
        # Fix the colormap handling to work with both string names and existing colormap objects
        if isinstance(cmap, str):
            self.cmap = sns.color_palette(cmap, as_cmap=True)
        else:
            # If cmap is already a colormap object, use it directly
            self.cmap = cmap
            
        self.config_paths = list(Path(self.base_path).glob("**/config.json"))
        self.embeddings_paths = [i.parent / "embeddings.pt" for i in self.config_paths]
        self.results_paths = [i.parent / "results.json" for i in self.config_paths]

        self.df = self._create_results_df()
        self.post_process()
        
        # Remove outliers if requested
        if remove_outliers:
            self.df = self.remove_outliers(**(outlier_params or {}))

    def _create_results_df(self):
        """Create a DataFrame from all results files found in the base path."""
        data = []
        for config_path, result_path in zip(self.config_paths, self.results_paths):
            if result_path.exists():  # Load config
                with open(config_path, "r") as f:
                    config = json.load(f)

                # Load results
                with open(result_path, "r") as f:
                    results = json.load(f)

                # Combine config and results
                entry = {**config, **results}
                entry["path"] = str(result_path.parent)
                data.append(entry)
        return pd.DataFrame(data)

    def post_process(self):
        pass


    def load_embeddings(self, path_or_index):
        """Load embeddings from a file path or by index in the DataFrame."""
        if isinstance(path_or_index, int):
            path = self.embeddings_paths[path_or_index]
        else:
            path = path_or_index

        return torch.load(path)

    def load_config(self, path_or_index):
        """Load config from a file path or by index in the DataFrame."""
        if isinstance(path_or_index, int):
            path = self.config_paths[path_or_index]
        else:
            path = path_or_index

        with open(path, "r") as f:
            return json.load(f)

    def load_results(self, path_or_index):
        """Load results from a file path or by index in the DataFrame."""
        if isinstance(path_or_index, int):
            path = self.results_paths[path_or_index]
        else:
            path = path_or_index

        with open(path, "r") as f:
            return json.load(f)

    def get_triplet(self, index):
        """Get the (embeddings, results, config) triplet by index."""
        config = self.load_config(index)
        results = self.load_results(index)
        embeddings = self.load_embeddings(index)
        return embeddings, results, config

    @abstractmethod
    def analyze(self):
        """Perform analysis on the loaded data."""
        pass

    @abstractmethod
    def visualize(self):
        """Create visualizations for the data."""
        pass

    def save_figure(self, fig, filename):
        """Save a matplotlib figure."""
        path = constants.FIGURE_PATH / f"{filename}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def save_and_return(self, fig, base_name):
        """Helper method to save figure and return it."""
        # Create directory structure with analyzer class name as a subdirectory
        analyzer_dir = self.__class__.__name__
        # Create the complete path
        filepath = f"{analyzer_dir}/{base_name}"
        # Save the figure
        save_path = constants.FIGURE_PATH / analyzer_dir
        save_path.mkdir(parents=True, exist_ok=True)
        self.save_figure(fig, filepath)
        return fig

    def run(self):
        """Run all available visualizations for this analyzer.
        
        Returns:
            dict: A dictionary mapping visualization names to the corresponding figure objects
        """
        print(f"Running analysis for {self.__class__.__name__}...")
        results = {}
        
        # Run the main analysis
        analysis_result = self.analyze()
        if analysis_result is not None:
            print("Analysis complete.")
        
        # Run the main visualization
        try:
            results["main"] = self.visualize()
            print("Main visualization created.")
        except Exception as e:
            print(f"Error creating main visualization: {e}")
        
        return results

    def remove_outliers(self, metrics=None, method='iqr', threshold=1.5, keep_filtered=False):
        """
        Remove outliers from the DataFrame based on specified metrics.
        
        Args:
            metrics (list): List of metrics to check for outliers (e.g., ['acc/test', 'lp/auc']).
                           If None, uses ['acc/test'] if available.
            method (str): Method to detect outliers - 'iqr', 'zscore', or 'percentile'
            threshold (float): Threshold for outlier detection:
                              - For IQR: Values outside threshold*IQR from Q1/Q3
                              - For z-score: Values more than threshold standard deviations from mean
                              - For percentile: Values outside the threshold and (100-threshold) percentiles
            keep_filtered (bool): If True, store filtered outliers in self.outliers_df
            
        Returns:
            DataFrame: DataFrame with outliers removed
        """
        if self.df.empty:
            print("No data to filter outliers from.")
            return self.df
            
        # If no metrics specified, try to use acc/test
        if metrics is None:
            if 'acc/test' in self.df.columns:
                metrics = ['acc/test']
            else:
                print("No metrics specified and 'acc/test' not found. Cannot filter outliers.")
                return self.df
        
        # Make a copy of the original DataFrame
        filtered_df = self.df.copy()
        outliers_df = pd.DataFrame()
        outlier_indices = set()
        
        # Process each metric
        for metric in metrics:
            if metric not in filtered_df.columns:
                print(f"Metric '{metric}' not found in data. Skipping.")
                continue
                
            # Select data points without NaN values for this metric
            metric_data = filtered_df[~filtered_df[metric].isna()]
            
            # Apply the selected outlier detection method
            if method.lower() == 'iqr':
                # IQR method
                Q1 = metric_data[metric].quantile(0.25)
                Q3 = metric_data[metric].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = metric_data[(metric_data[metric] < lower_bound) | 
                                      (metric_data[metric] > upper_bound)]
                
                print(f"IQR filtering on {metric}: Q1={Q1:.4f}, Q3={Q3:.4f}, IQR={IQR:.4f}")
                print(f"Bounds: [{lower_bound:.4f}, {upper_bound:.4f}]")
                
            elif method.lower() == 'zscore':
                # Z-score method
                mean = metric_data[metric].mean()
                std = metric_data[metric].std()
                
                # Identify outliers as points with absolute z-score > threshold
                z_scores = abs((metric_data[metric] - mean) / std)
                outliers = metric_data[z_scores > threshold]
                
                print(f"Z-score filtering on {metric}: mean={mean:.4f}, std={std:.4f}, threshold={threshold}")
                
            elif method.lower() == 'percentile':
                # Percentile method
                lower_bound = metric_data[metric].quantile(threshold / 100)
                upper_bound = metric_data[metric].quantile(1 - threshold / 100)
                
                outliers = metric_data[(metric_data[metric] < lower_bound) | 
                                      (metric_data[metric] > upper_bound)]
                
                print(f"Percentile filtering on {metric}: bounds=[{lower_bound:.4f}, {upper_bound:.4f}]")
                
            else:
                print(f"Unknown outlier detection method: {method}. Using IQR method.")
                # Default to IQR method
                Q1 = metric_data[metric].quantile(0.25)
                Q3 = metric_data[metric].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = metric_data[(metric_data[metric] < lower_bound) | 
                                      (metric_data[metric] > upper_bound)]
            
            # Add to the set of outlier indices
            outlier_indices.update(outliers.index)
            
            # Report the number of outliers found
            print(f"Found {len(outliers)} outliers in '{metric}' using {method} method")
        
        # Remove all identified outliers and store them if requested
        if outlier_indices:
            if keep_filtered:
                self.outliers_df = self.df.loc[list(outlier_indices)]
                print(f"Stored {len(self.outliers_df)} outliers in self.outliers_df")
                
            # Remove outliers from the DataFrame
            filtered_df = filtered_df.drop(index=list(outlier_indices))
            print(f"Removed {len(outlier_indices)} total outliers across all metrics")
            
        return filtered_df
        
    def visualize_outliers(self, metrics=None, method='iqr', threshold=1.5):
        """
        Visualize data distribution with outliers highlighted.
        
        Args:
            metrics (list): List of metrics to visualize (e.g., ['acc/test', 'lp/auc']).
                          If None, uses ['acc/test'] if available.
            method (str): Method to detect outliers - 'iqr', 'zscore', or 'percentile'
            threshold (float): Threshold for outlier detection
            
        Returns:
            matplotlib figure with boxplots or histograms showing outliers
        """
        if self.df.empty:
            print("No data to visualize outliers for.")
            return plt.figure()
            
        # If no metrics specified, try to use acc/test
        if metrics is None:
            if 'acc/test' in self.df.columns:
                metrics = ['acc/test']
            else:
                print("No metrics specified and 'acc/test' not found.")
                return plt.figure()
                
        # Create subplots for each metric
        fig, axes = plt.subplots(len(metrics), 2, figsize=(12, 5*len(metrics)), squeeze=False)
        
        # For storing temporary outlier information
        temp_df = self.df.copy()
        
        # Process each metric
        for i, metric in enumerate(metrics):
            if metric not in temp_df.columns:
                print(f"Metric '{metric}' not found in data. Skipping.")
                axes[i, 0].text(0.5, 0.5, f"Metric '{metric}' not found", 
                              ha='center', va='center', transform=axes[i, 0].transAxes)
                axes[i, 1].text(0.5, 0.5, f"Metric '{metric}' not found", 
                              ha='center', va='center', transform=axes[i, 1].transAxes)
                continue
                
            # Select data points without NaN values for this metric
            metric_data = temp_df[~temp_df[metric].isna()]
            
            # Identify outliers based on the selected method
            if method.lower() == 'iqr':
                # IQR method
                Q1 = metric_data[metric].quantile(0.25)
                Q3 = metric_data[metric].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                temp_df[f'{metric}_outlier'] = ((temp_df[metric] < lower_bound) | 
                                             (temp_df[metric] > upper_bound))
                
                title_info = f"IQR method (threshold={threshold})"
                
            elif method.lower() == 'zscore':
                # Z-score method
                mean = metric_data[metric].mean()
                std = metric_data[metric].std()
                
                # Calculate z-scores
                temp_df[f'{metric}_zscore'] = abs((temp_df[metric] - mean) / std)
                temp_df[f'{metric}_outlier'] = temp_df[f'{metric}_zscore'] > threshold
                
                title_info = f"Z-score method (threshold={threshold})"
                
            elif method.lower() == 'percentile':
                # Percentile method
                lower_bound = metric_data[metric].quantile(threshold / 100)
                upper_bound = metric_data[metric].quantile(1 - threshold / 100)
                
                temp_df[f'{metric}_outlier'] = ((temp_df[metric] < lower_bound) | 
                                             (temp_df[metric] > upper_bound))
                                             
                title_info = f"Percentile method (threshold={threshold}%)"
                
            else:
                print(f"Unknown outlier detection method: {method}. Using IQR method.")
                # Default to IQR method
                Q1 = metric_data[metric].quantile(0.25)
                Q3 = metric_data[metric].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                temp_df[f'{metric}_outlier'] = ((temp_df[metric] < lower_bound) | 
                                             (temp_df[metric] > upper_bound))
                                             
                title_info = f"IQR method (threshold={threshold})"
            
            # Create boxplot
            sns.boxplot(y=metric, data=temp_df, ax=axes[i, 0])
            axes[i, 0].set_title(f"Boxplot of {metric}\n{title_info}")
            
            # Create histogram with outliers highlighted
            if f'{metric}_outlier' in temp_df.columns:
                sns.histplot(
                    data=temp_df, 
                    x=metric,
                    hue=f'{metric}_outlier',
                    multiple="stack",
                    palette=["#1f77b4", "#d62728"],  # Blue for normal, red for outliers
                    ax=axes[i, 1]
                )
                axes[i, 1].set_title(f"Histogram of {metric} with outliers\n{title_info}")
                
                # Calculate and display outlier count
                outlier_count = temp_df[f'{metric}_outlier'].sum()
                total_count = len(temp_df[~temp_df[metric].isna()])
                axes[i, 1].text(
                    0.95, 0.95, 
                    f"Outliers: {outlier_count}/{total_count} ({outlier_count/total_count:.1%})",
                    transform=axes[i, 1].transAxes,
                    ha='right', va='top',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
                )
            
        plt.tight_layout()
        return self.save_and_return(fig, f"outlier_detection_{method}")


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
                "lp/auc": ["mean", "std"]
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
                    .agg({"lp/auc": "mean"})
                    .reset_index()
                    .pivot(index="k", columns="reg", values="lp/auc")
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
                .agg({"lp/auc": "mean"})
                .reset_index()
                .pivot(index="k", columns="reg", values="lp/auc")
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
        if "lp/auc" in plot_df.columns:
            metrics.append("lp/auc")
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


class AdditionFusionAnalyzer(FusionAnalyzer):
    def __init__(self, dpi=300, cmap="viridis", figsize=(6.4, 4.8), 
                 remove_outliers=False, outlier_params=None):
        super().__init__("AdditionFusion", dpi, cmap, figsize, 
                         remove_outliers=remove_outliers, outlier_params=outlier_params)

    def run(self):
        """Run all available visualizations for addition fusion analysis."""
        return super().run()


class EarlyFusionAnalyzer(FusionAnalyzer):
    def __init__(self, dpi=300, cmap="viridis", figsize=(6.4, 4.8), 
                 remove_outliers=False, outlier_params=None):
        super().__init__("EarlyFusion", dpi, cmap, figsize, 
                         remove_outliers=remove_outliers, outlier_params=outlier_params)

    def run(self):
        """Run all available visualizations for early fusion analysis."""
        return super().run()


class GatedFusionAnalyzer(FusionAnalyzer):
    def __init__(self, dpi=300, cmap="viridis", figsize=(6.4, 4.8), 
                 remove_outliers=False, outlier_params=None):
        super().__init__("GatedFusion", dpi, cmap, figsize, 
                         remove_outliers=remove_outliers, outlier_params=outlier_params)

    def analyze_gate_scores(self):
        """Analyze the distribution of gate scores."""
        gate_scores = []

        for i, path in enumerate(self.results_paths):
            result_path = path.parent / "gate_scores.json"
            if (result_path.exists()):
                with open(result_path, "r") as f:
                    scores = json.load(f)
                gate_scores.append(
                    {
                        "path": str(path.parent),
                        "mean_score": np.mean(scores),
                        "std_score": np.std(scores),
                        "min_score": np.min(scores),
                        "max_score": np.max(scores),
                        "scores": scores,
                    }
                )

        return pd.DataFrame(gate_scores) if gate_scores else pd.DataFrame()


    def run(self):
        """Run all available visualizations for gated fusion analysis."""
        results = super().run()
        

            
        return results


class LowRankFusionAnalyzer(FusionAnalyzer):
    def __init__(self, dpi=300, cmap="viridis", figsize=(6.4, 4.8), 
                 remove_outliers=False, outlier_params=None):
        super().__init__("LowRankFusion", dpi, cmap, figsize, 
                         remove_outliers=remove_outliers, outlier_params=outlier_params)
    def post_process(self):
        self.df[["textual_name", "relational_name", "textual_dim", "relational_dim", "latent_dim", "rank"]] = self.df.prefix.str.split(
            "[/_]"
        ).tolist()
        self.df = self.df.drop(columns=["prefix"])
    def analyze(self):
        """Analyze with additional focus on rank parameter."""
        if self.df.empty:
            print("No data to analyze.")
            return None


        # Basic analysis from parent
        basic_analysis = super().analyze()

        # Additional analysis by rank
        rank_analysis = (
            self.df.groupby(["latent_dim", "rank"])
            .agg({"acc/test": ["mean", "std", "count"]})
            .reset_index()
        )

        # Format column names
        rank_analysis.columns = [
            "_".join(col).strip() if col[1] else col[0]
            for col in rank_analysis.columns.values
        ]

        return {"basic_analysis": basic_analysis, "rank_analysis": rank_analysis}

    def run(self):
        """Run all available visualizations for low rank fusion analysis."""
        results = super().run()
        

        return results


class TransformerFusionAnalyzer(FusionAnalyzer):
    def __init__(self, dpi=300, cmap="viridis", figsize=(6.4, 4.8), 
                 remove_outliers=False, outlier_params=None):
        super().__init__("TransformerFusion", dpi, cmap, figsize, 
                         remove_outliers=remove_outliers, outlier_params=outlier_params)
    def post_process(self):
        self.df[["textual_name", "relational_name", "textual_dim", "relational_dim", "latent_dim", "num_layers", "nhead", "output_modality"]] = self.df.prefix.str.split(
            "[/_]"
        ).tolist()
        self.df = self.df.drop(columns=["prefix"])
    def analyze(self):
        """Analyze with additional focus on transformer-specific parameters."""
        if self.df.empty:
            print("No data to analyze.")
            return None

        # Extract transformer parameters from path
        path_pattern = r"/\d+_\d+/(\d+)_(\d+)/(\w+)"
        self.df["num_layers"] = (
            self.df["path"].str.extract(path_pattern).iloc[:, 0].astype(int)
        )
        self.df["nhead"] = (
            self.df["path"].str.extract(path_pattern).iloc[:, 1].astype(int)
        )
        self.df["output_modality"] = (
            self.df["path"].str.extract(path_pattern).iloc[:, 2]
        )

        # Basic analysis from parent
        basic_analysis = super().analyze()

        # Additional analysis by transformer parameters
        transformer_analysis = (
            self.df.groupby(["latent_dim", "num_layers", "nhead", "output_modality"])
            .agg({"acc/test": ["mean", "std", "count"]})
            .reset_index()
        )

        # Format column names
        transformer_analysis.columns = [
            "_".join(col).strip() if col[1] else col[0]
            for col in transformer_analysis.columns.values
        ]

        return {
            "basic_analysis": basic_analysis,
            "transformer_analysis": transformer_analysis,
        }

    def visualize_output_modality(self):
        """Visualize the impact of output modality on model performance."""
        if self.df.empty:
            print("No data to visualize.")
            return plt.figure()

        # Extract parameters if not already done
        if "output_modality" not in self.df.columns:
            path_pattern = r"/\d+_\d+/(\d+)_(\d+)/(\w+)"
            self.df["num_layers"] = (
                self.df["path"].str.extract(path_pattern).iloc[:, 0].astype(int)
            )
            self.df["nhead"] = (
                self.df["path"].str.extract(path_pattern).iloc[:, 1].astype(int)
            )
            self.df["output_modality"] = (
                self.df["path"].str.extract(path_pattern).iloc[:, 2]
            )

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(
            x="output_modality", y="acc/test", hue="latent_dim", data=self.df, ax=ax
        )
        ax.set_title("Transformer Fusion: Test Accuracy by Output Modality")
        ax.set_xlabel("Output Modality")
        ax.set_ylabel("Test Accuracy")
        ax.legend(title="Latent Dim")

        return self.save_and_return(fig, "output_modality_impact")

    def visualize_architecture_impact(self):
        """Visualize the impact of transformer architecture parameters."""
        if self.df.empty:
            print("No data to visualize.")
            return plt.figure()

        # Extract parameters if not already done
        if "num_layers" not in self.df.columns:
            path_pattern = r"/\d+_\d+/(\d+)_(\d+)/(\w+)"
            self.df["num_layers"] = (
                self.df["path"].str.extract(path_pattern).iloc[:, 0].astype(int)
            )
            self.df["nhead"] = (
                self.df["path"].str.extract(path_pattern).iloc[:, 1].astype(int)
            )
            self.df["output_modality"] = (
                self.df["path"].str.extract(path_pattern).iloc[:, 2]
            )

        # Create a grid of plots for different combinations
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Impact of number of layers
        sns.boxplot(x="num_layers", y="acc/test", data=self.df, ax=axes[0])
        axes[0].set_title("Impact of Number of Layers")
        axes[0].set_xlabel("Number of Layers")
        axes[0].set_ylabel("Test Accuracy")

        # Plot 2: Impact of number of heads
        sns.boxplot(x="nhead", y="acc/test", data=self.df, ax=axes[1])
        axes[1].set_title("Impact of Number of Attention Heads")
        axes[1].set_xlabel("Number of Heads")
        axes[1].set_ylabel("Test Accuracy")

        plt.tight_layout()
        return self.save_and_return(fig, "architecture_impact")

    def run(self):
        """Run all available visualizations for transformer fusion analysis."""
        results = super().run()
        
        try:
            results["output_modality"] = self.visualize_output_modality()
            print("Output modality visualization created.")
        except Exception as e:
            print(f"Error creating output modality visualization: {e}")
            
        try:
            results["architecture_impact"] = self.visualize_architecture_impact()
            print("Architecture impact visualization created.")
        except Exception as e:
            print(f"Error creating architecture impact visualization: {e}")
            
        return results


class CrossModelAnalyzer(AbstractAnalyzer):
    """Compare results across different fusion models."""

    def __init__(self, dpi=300, cmap="viridis", figsize=(6.4, 4.8), 
                 remove_outliers=False, outlier_params=None):
        # Use a common parent directory that contains all model types
        super().__init__("/home/lcheng/oz318/fusion/logs", dpi, cmap, figsize, 
                        remove_outliers=remove_outliers, outlier_params=outlier_params)

    def analyze(self):
        """Analyze and compare results across different fusion models."""
        if self.df.empty:
            print("No data to analyze.")
            return None

        # Extract model type from path
        self.df["model_type"] = self.df["path"].str.extract(r"/logs/([^/]+)/")

        # Basic comparison across models
        model_comparison = (
            self.df.groupby("model_type")
            .agg({
                "acc/test": ["mean", "std", "max", "min", "count"],
                # Include link prediction metrics if they exist
                **({'lp/auc': ["mean", "std", "max", "min"]} if 'lp/auc' in self.df.columns else {}),
                **({'lp_hard/auc': ["mean", "std", "max", "min"]} if 'lp_hard/auc' in self.df.columns else {})
            })
            .reset_index()
        )

        # Format column names
        model_comparison.columns = [
            "_".join(col).strip() if col[1] else col[0]
            for col in model_comparison.columns.values
        ]

        return model_comparison

    def visualize(self):
        """Visualize comparison of different fusion models across all available metrics."""
        if self.df.empty:
            print("No data to visualize.")
            return plt.figure()

        # Extract model type if not already done
        if "model_type" not in self.df.columns:
            self.df["model_type"] = self.df["path"].str.extract(r"/logs/([^/]+)/")
            
        # Identify which metrics are available
        metrics = ['acc/test']
        if 'lp/auc' in self.df.columns:
            metrics.append('lp/auc')
        if 'lp_hard/auc' in self.df.columns:
            metrics.append('lp_hard/auc')
            
        # Create a subplot for each metric
        num_metrics = len(metrics)
        fig, axes = plt.subplots(1, num_metrics, figsize=(12 * num_metrics / 2, 8), squeeze=False)
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axes[0, i]
            sns.boxplot(x="model_type", y=metric, data=self.df, ax=ax)
            ax.set_title(f"Comparison of Fusion Models: {metric.replace('/', ' ').title()}")
            ax.set_xlabel("Fusion Model")
            ax.set_ylabel(metric.replace('/', ' ').title())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        return self.save_and_return(fig, "model_comparison")

    def visualize_by_embedding_type(self):
        """Compare model performance across different embedding combinations."""
        if self.df.empty:
            print("No data to visualize.")
            return plt.figure()

        # Extract model type and embedding info
        if "model_type" not in self.df.columns:
            self.df["model_type"] = self.df["path"].str.extract(r"/logs/([^/]+)/")

        self.df["textual_name"] = self.df["path"].str.extract(r"/([^/]+)_[^/]+/\d+_\d+")
        self.df["relational_name"] = self.df["path"].str.extract(
            r"/[^/]+_([^/]+)/\d+_\d+"
        )

        # Create a composite embedding type
        self.df["embedding_combo"] = (
            self.df["textual_name"] + "_" + self.df["relational_name"]
        )
        
        # Identify which metrics are available
        metrics = ['acc/test']
        if 'lp/auc' in self.df.columns:
            metrics.append('lp/auc')
        if 'lp_hard/auc' in self.df.columns:
            metrics.append('lp_hard/auc')
            
        # Create a subplot for each metric
        num_metrics = len(metrics)
        fig, axes = plt.subplots(num_metrics, 1, figsize=(14, 8 * num_metrics), squeeze=False)
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axes[i, 0]
            sns.boxplot(
                x="embedding_combo", y=metric, hue="model_type", data=self.df, ax=ax
            )
            ax.set_title(f"Model {metric.replace('/', ' ').title()} by Embedding Combination")
            ax.set_xlabel("Embedding Combination (Textual_Relational)")
            ax.set_ylabel(metric.replace('/', ' ').title())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            ax.legend(title="Fusion Model")

        plt.tight_layout()
        return self.save_and_return(fig, "embedding_type_comparison")
    
    def visualize_metrics_comparison(self):
        """Compare different metrics across models in a grouped bar chart."""
        if self.df.empty:
            print("No data to visualize correlations.")
            return plt.figure()
            
        # Extract model type if not already done
        if "model_type" not in self.df.columns:
            self.df["model_type"] = self.df["path"].str.extract(r"/logs/([^/]+)/")
            
        # Identify available metrics
        metrics = ['acc/test']
        if 'lp/auc' in self.df.columns:
            metrics.append('lp/auc')
        if 'lp_hard/auc' in self.df.columns:
            metrics.append('lp_hard/auc')
            
        if len(metrics) <= 1:
            print("Not enough metrics for comparison.")
            return plt.figure()
            
        # Calculate average metrics by model type
        avg_metrics = self.df.groupby('model_type')[metrics].mean().reset_index()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Set up positions for grouped bars
        models = avg_metrics['model_type'].unique()
        x = np.arange(len(models))
        width = 0.8 / len(metrics)  # Adjust bar width based on number of metrics
        
        # Colors for different metrics
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, orange, green
        
        # Create grouped bars
        for i, metric in enumerate(metrics):
            pos = x + width * (i - (len(metrics) - 1) / 2)
            bars = ax.bar(pos, avg_metrics[metric], width, label=metric.replace('/', ' ').title(), color=colors[i % len(colors)])
            
            # Add value labels on bars
            for bar_idx, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8, color=colors[i % len(colors)])
        
        ax.set_xlabel('Fusion Model')
        ax.set_ylabel('Performance')
        ax.set_title('Cross-Model Performance Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(title='Metric')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        return self.save_and_return(fig, "metrics_comparison")

    def visualize_fusion_heatmaps(self):
        """
        Create heatmap visualizations for each fusion method, comparing different textual and relational combinations.
        
        Returns:
            dict: Dictionary containing the three figures for test accuracy, lp/auc and lp_hard/auc
        """
        if self.df.empty:
            print("No data to visualize fusion heatmaps.")
            return {}
            
        # Extract model type if not already done
        if "model_type" not in self.df.columns:
            self.df["model_type"] = self.df["path"].str.extract(r"/logs/([^/]+)/")
            
        # Extract textual and relational model names if not already done
        if "textual_name" not in self.df.columns:
            self.df["textual_name"] = self.df["path"].str.extract(r"/([^/]+)_[^/]+/\d+_\d+")
            
        if "relational_name" not in self.df.columns:
            self.df["relational_name"] = self.df["path"].str.extract(r"/[^/]+_([^/]+)/\d+_\d+")
            
        # Check if required columns are available
        required_cols = ["model_type", "textual_name", "relational_name"]
        for col in required_cols:
            if col not in self.df.columns or self.df[col].isnull().all():
                print(f"Missing required column: {col}")
                return {}
                
        # Filter out non-fusion models or rows with missing values
        fusion_models = self.df.dropna(subset=required_cols)
        if fusion_models.empty:
            print("No fusion models found with complete data.")
            return {}
        
        # Define metrics to visualize
        metrics = ["acc/test"]
        if "lp/auc" in fusion_models.columns:
            metrics.append("lp/auc")
        if "lp_hard/auc" in fusion_models.columns:
            metrics.append("lp_hard/auc")
            
        # Create a figure for each metric
        results = {}
        
        for metric in metrics:
            # Skip if metric not available
            if metric not in fusion_models.columns:
                continue
                
            # Get unique fusion models, textual models, and relational models
            unique_fusion_models = sorted(fusion_models["model_type"].unique())
            unique_textual_models = sorted(fusion_models["textual_name"].unique())
            unique_relational_models = sorted(fusion_models["relational_name"].unique())
            
            # Use a 2x2 grid for the 4 fusion methods
            n_fusion_models = len(unique_fusion_models)
            n_cols = 2  # Fixed 2 columns for 2x2 grid
            n_rows = (n_fusion_models + 1) // 2  # This will give 2 rows for 3-4 methods
            
            # Create figure and subplots with appropriate size for 2x2 grid
            fig_width = 10  # 5 per column x 2 columns
            fig_height = 12  # 6 per row x 2 rows
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
            
            # Set title for entire figure
            metric_name = metric.replace("/", " ").replace("_", " ").title()
            fig.suptitle(f"Comparison of Fusion Methods: {metric_name}", fontsize=16, y=0.98)
            
            # Create a heatmap for each fusion model
            for i, fusion_model in enumerate(unique_fusion_models):
                # Calculate subplot position in 2x2 grid
                row_idx = i // n_cols
                col_idx = i % n_cols
                ax = axes[row_idx, col_idx]
                
                # Filter data for this fusion model
                model_data = fusion_models[fusion_models["model_type"] == fusion_model]
                
                # Create pivot table for the heatmap
                try:
                    # Group by textual and relational model and calculate mean of the metric
                    pivot_data = (
                        model_data.groupby(["textual_name", "relational_name"])
                        .agg({metric: "mean"})
                        .reset_index()
                        .pivot(index="textual_name", columns="relational_name", values=metric)
                    )
                    
                    # Ensure all combinations have a value (even if not present in the data)
                    for tx in unique_textual_models:
                        if tx not in pivot_data.index:
                            pivot_data.loc[tx] = np.nan
                    for rel in unique_relational_models:
                        if rel not in pivot_data.columns:
                            pivot_data[rel] = np.nan
                            
                    # Sort rows and columns for consistent ordering
                    pivot_data = pivot_data.loc[sorted(pivot_data.index), sorted(pivot_data.columns)]
                    
                    # Create heatmap
                    sns.heatmap(pivot_data, cmap=self.cmap, annot=True, fmt=".3f", ax=ax,
                                linewidths=.5, cbar=False)
                    
                    # Set title and labels
                    ax.set_title(f"{fusion_model}")
                    ax.set_xlabel("Relational Model")
                    ax.set_ylabel("Textual Model")
                    
                    # Rotate x-axis labels for better readability
                    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
                    
                except Exception as e:
                    # Handle case where pivot fails (e.g., no data for this combination)
                    ax.text(0.5, 0.5, f"No data available for\n{fusion_model}\n{e}",
                           ha="center", va="center", transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
            
            # Hide unused subplots
            for i in range(n_fusion_models, n_rows * n_cols):
                row_idx = i // n_cols
                col_idx = i % n_cols
                axes[row_idx, col_idx].axis('off')
            
            # Add a colorbar at the bottom spanning all subplots
            cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])  # [left, bottom, width, height]
            vmin = fusion_models[metric].min()
            vmax = fusion_models[metric].max()
            sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
            cbar.set_label(metric_name)
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0.07, 1, 0.96])  # [left, bottom, right, top]
            
            # Save figure
            results[metric] = self.save_and_return(fig, f"fusion_heatmap_{metric.replace('/', '_')}")
            
        return results
        
    def run(self):
        """Run all available visualizations for cross-model analysis."""
        results = super().run()
        
        # Get available metrics
        metrics = ['acc/test']
        if 'lp/auc' in self.df.columns:
            metrics.append('lp/auc')
        if 'lp_hard/auc' in self.df.columns:
            metrics.append('lp_hard/auc')
        
        try:
            results["embedding_type"] = self.visualize_by_embedding_type()
            print("Embedding type comparison visualization created.")
        except Exception as e:
            print(f"Error creating embedding type comparison: {e}")
            
        try:
            results["metrics_comparison"] = self.visualize_metrics_comparison()
            print("Metrics comparison visualization created.")
        except Exception as e:
            print(f"Error creating metrics comparison: {e}")

        # Add the new fusion heatmap visualizations
        try:
            fusion_heatmaps = self.visualize_fusion_heatmaps()
            for metric, fig in fusion_heatmaps.items():
                results[f"fusion_heatmap_{metric}"] = fig
            if fusion_heatmaps:
                print("Fusion heatmap visualizations created.")
        except Exception as e:
            print(f"Error creating fusion heatmap visualizations: {e}")
            
        # Add performance tradeoffs visualization
        try:
            results["performance_tradeoffs"] = self.visualize_performance_tradeoffs()
            print("Performance tradeoffs visualization created.")
        except Exception as e:
            print(f"Error creating performance tradeoffs visualization: {e}")
            
        # Add metric correlations visualization
        try:
            results["metric_correlations"] = self.visualize_metric_correlations()
            print("Metric correlations visualization created.")
        except Exception as e:
            print(f"Error creating metric correlations visualization: {e}")
            
        # Create performance distributions for all available metrics
        for metric in metrics:
            try:
                metric_key = f"performance_distributions_{metric.replace('/', '_')}"
                results[metric_key] = self.visualize_performance_distributions(metric)
                print(f"Performance distributions visualization created for {metric}.")
            except Exception as e:
                print(f"Error creating performance distributions visualization for {metric}: {e}")
            
        # Create embedding quality impact for all available metrics
        for metric in metrics:
            try:
                metric_key = f"embedding_quality_impact_{metric.replace('/', '_')}"
                results[metric_key] = self.visualize_embedding_quality_impact(metric)
                print(f"Embedding quality impact visualization created for {metric}.")
            except Exception as e:
                print(f"Error creating embedding quality impact visualization for {metric}: {e}")
            
        # Create performance gain for all available metrics
        for metric in metrics:
            try:
                metric_key = f"performance_gain_{metric.replace('/', '_')}"
                results[metric_key] = self.visualize_performance_gain(metric)
                print(f"Performance gain visualization created for {metric}.")
            except Exception as e:
                print(f"Error creating performance gain visualization for {metric}: {e}")

        return results

    def visualize_performance_tradeoffs(self):
        """
        Create a scatter plot showing trade-offs between node classification and link prediction performance.
        Each point represents a model/experiment, positioned by its accuracy vs LP performance.
        """
        if self.df.empty or 'acc/test' not in self.df.columns or 'lp/auc' not in self.df.columns:
            print("Missing required metrics for trade-off visualization.")
            return plt.figure()
            
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Add model type column if not present
        if 'model_type' not in self.df.columns:
            self.df['model_type'] = self.df['path'].str.extract(r'/logs/([^/]+)/')
        
        # Create scatter plot with different colors for different model types
        models = sorted(self.df['model_type'].unique())
        palette = sns.color_palette(n_colors=len(models))
        
        for i, model in enumerate(models):
            model_data = self.df[self.df['model_type'] == model]
            ax.scatter(
                model_data['acc/test'], 
                model_data['lp/auc'], 
                s=80, 
                color=palette[i], 
                alpha=0.7,
                label=model
            )
        
        # Add diagonal line representing equal performance on both tasks
        min_val = min(self.df['acc/test'].min(), self.df['lp/auc'].min())
        max_val = max(self.df['acc/test'].max(), self.df['lp/auc'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)
        
        # Add labels for interesting points
        top_models = self.df.nlargest(5, 'acc/test')
        for _, row in top_models.iterrows():
            ax.annotate(
                f"{row['model_type']}",
                (row['acc/test'], row['lp/auc']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )
        
        ax.set_xlabel('Node Classification Accuracy')
        ax.set_ylabel('Link Prediction AUC')
        ax.set_title('Performance Trade-offs: Node Classification vs. Link Prediction')
        ax.grid(True, alpha=0.3)
        ax.legend(title='Model Type')
        
        return self.save_and_return(fig, "performance_tradeoffs")

    def visualize_metric_correlations(self):
        """
        Create a heatmap showing correlations between different performance metrics.
        """
        if self.df.empty:
            print("No data to visualize correlations.")
            return plt.figure()
        
        # Identify metrics columns (those containing '/' in their name)
        metric_cols = [col for col in self.df.columns if '/' in col]
        
        if len(metric_cols) < 2:
            print("Not enough metrics for correlation analysis.")
            return plt.figure()
        
        # Calculate correlation matrix
        corr_matrix = self.df[metric_cols].corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            fmt=".2f", 
            cmap="coolwarm", 
            mask=mask,
            vmin=-1, 
            vmax=1, 
            ax=ax
        )
        
        ax.set_title('Correlation Matrix of Performance Metrics')
        
        return self.save_and_return(fig, "metric_correlations")

    def visualize_performance_distributions(self, metric='acc/test'):
        """
        Visualize the performance distribution across different models using violin plots.
        
        Args:
            metric: The performance metric to visualize
        """
        if self.df.empty or metric not in self.df.columns:
            print(f"Missing required metric '{metric}'")
            return plt.figure()
        
        # Add model type column if not present
        if 'model_type' not in self.df.columns:
            self.df['model_type'] = self.df['path'].str.extract(r'/logs/([^/]+)/')
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Create violin plot
        sns.violinplot(
            x='model_type', 
            y=metric, 
            data=self.df,
            palette="Set3",
            inner="box",  # Show box plot inside violin
            ax=ax
        )
        
        # Add individual data points
        sns.stripplot(
            x='model_type', 
            y=metric, 
            data=self.df,
            size=4, 
            alpha=0.4,
            jitter=True,
            ax=ax
        )
        
        # Add mean values as text
        for i, model in enumerate(sorted(self.df['model_type'].unique())):
            model_data = self.df[self.df['model_type'] == model]
            mean_val = model_data[metric].mean()
            ax.text(
                i, mean_val + 0.01,
                f'μ={mean_val:.3f}',
                ha='center', fontsize=9
            )
        
        ax.set_xlabel('Model Type')
        ax.set_ylabel(f'{metric} Performance')
        ax.set_title(f'Distribution of {metric} Performance by Model Type')
        plt.xticks(rotation=45, ha='right')
        
        return self.save_and_return(fig, f"performance_distribution_{metric}".replace('/', '_'))
        

    def visualize_embedding_quality_impact(self, metric='acc/test'):
        """
        Visualize how the quality of input embeddings (textual and relational)
        affects the fusion model performance.
        
        Args:
            metric: Performance metric to visualize (e.g., 'acc/test', 'lp/auc')
        """
        if self.df.empty:
            print("No data to visualize.")
            return plt.figure()
            
        # Extract needed information from the fusion models DataFrame
        if "model_type" not in self.df.columns:
            self.df["model_type"] = self.df["path"].str.extract(r"/logs/([^/]+)/")
            
        if "textual_name" not in self.df.columns:
            self.df["textual_name"] = self.df["path"].str.extract(r"/([^/]+)_[^/]+/\d+_\d+")
            
        if "relational_name" not in self.df.columns:
            self.df["relational_name"] = self.df["path"].str.extract(r"/[^/]+_([^/]+)/\d+_\d+")
            
        # Check required columns are available
        for col in ["model_type", "textual_name", "relational_name", metric]:
            if col not in self.df.columns or self.df[col].isnull().all():
                print(f"Missing required column: {col}")
                return plt.figure()
                
        # Load baseline metrics
        baselines = self.load_baseline_metrics()
        if not baselines["relational"] or not baselines["textual"]:
            print("No baseline metrics available for comparison.")
            return plt.figure()
            
        # Create a temporary DataFrame for the visualization with matched baselines
        plot_data_rows = []
        
        for idx, row in self.df.iterrows():
            # Skip rows without required info
            if pd.isna(row["textual_name"]) or pd.isna(row["relational_name"]):
                continue
                
            # Try to find matching baseline metrics
            try:
                # Match with textual baseline
                textual_key = f"{row['textual_name']}/{row['textual_dim']}"
                if textual_key in baselines['textual'] and metric in baselines['textual'][textual_key]:
                    textual_baseline = baselines['textual'][textual_key][metric]
                    
                    # Match with relational baseline
                    rel_dim = str(row['relational_dim'])
                    if rel_dim in baselines['relational'] and metric in baselines['relational'][rel_dim]:
                        relational_baseline = baselines['relational'][rel_dim][metric]
                        
                        # Add row with matched baselines
                        plot_data_rows.append({
                            'model_type': row['model_type'],
                            metric: row[metric],
                            'textual_baseline': textual_baseline,
                            'relational_baseline': relational_baseline,
                            'textual_name': row['textual_name'],
                            'relational_name': row['relational_name']
                        })
            except Exception as e:
                # Skip rows with missing information or errors
                pass
                
        # Create plot DataFrame from collected rows
        if not plot_data_rows:
            print(f"No rows with matching baseline metrics found for {metric}.")
            return plt.figure()
            
        plot_df = pd.DataFrame(plot_data_rows)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create scatter plot with model type markers
        models = sorted(plot_df['model_type'].unique())
        markers = ['o', 's', '^', 'd', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '|', '_'][:len(models)]
        
        # Calculate improvement over baseline for color intensity
        plot_df['avg_baseline'] = (plot_df['textual_baseline'] + plot_df['relational_baseline']) / 2
        plot_df['improvement'] = plot_df[metric] - plot_df['avg_baseline']
        
        # Setup color normalization based on improvement
        norm = plt.Normalize(plot_df['improvement'].min(), plot_df['improvement'].max())
        
        # Create a proxy artist for the color bar
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        
        # Plot points for each model type with custom markers
        for i, model in enumerate(models):
            model_data = plot_df[plot_df['model_type'] == model]
            scatter = ax.scatter(
                model_data['textual_baseline'],
                model_data['relational_baseline'],
                c=model_data['improvement'], 
                cmap='viridis',
                s=100, 
                marker=markers[i % len(markers)],
                alpha=0.8,
                edgecolors='white',
                linewidth=1,
                label=model
            )
            
            # Add annotations for significant points
            if len(model_data) > 0:
                best_idx = model_data[metric].idxmax()
                best_row = model_data.loc[best_idx]
                ax.annotate(
                    f"{model}: {best_row[metric]:.3f}",
                    (best_row['textual_baseline'], best_row['relational_baseline']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
                )
        
        # Diagonal line representing equal performance of both embeddings
        min_val = min(plot_df['textual_baseline'].min(), plot_df['relational_baseline'].min())
        max_val = max(plot_df['textual_baseline'].max(), plot_df['relational_baseline'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        # Add colorbar and labels
        cbar = plt.colorbar(sm)
        metric_name = metric.replace('/', ' ').replace('_', ' ').title()
        cbar.set_label(f'Improvement over Average Baseline ({metric_name})')
        
        ax.set_xlabel(f'Textual Embedding {metric_name}')
        ax.set_ylabel(f'Relational Embedding {metric_name}')
        ax.set_title(f'Impact of Input Embedding Quality on Fusion {metric_name}')
        
        ax.set_xlim(plot_df['textual_baseline'].min() - 0.02, plot_df['textual_baseline'].max() + 0.02)
        ax.set_ylim(plot_df['relational_baseline'].min() - 0.02, plot_df['relational_baseline'].max() + 0.02)
        
        ax.grid(True, alpha=0.3)
        ax.legend(title='Fusion Method')
        
        metric_suffix = metric.replace('/', '_')
        return self.save_and_return(fig, f"embedding_quality_impact_{metric_suffix}")

    def visualize_performance_gain(self, metric='acc/test'):
        """
        Visualize how much performance is gained by fusion compared to individual baselines.
        
        Args:
            metric: Performance metric to visualize (e.g., 'acc/test', 'lp/auc')
        """
        if self.df.empty:
            print("No data to visualize.")
            return plt.figure()
            
        # Extract needed information from the DataFrame
        if "model_type" not in self.df.columns:
            self.df["model_type"] = self.df["path"].str.extract(r"/logs/([^/]+)/")
            
        if "textual_name" not in self.df.columns:
            self.df["textual_name"] = self.df["path"].str.extract(r"/([^/]+)_[^/]+/\d+_\d+")
            
        if "relational_name" not in self.df.columns:
            self.df["relational_name"] = self.df["path"].str.extract(r"/[^/]+_([^/]+)/\d+_\d+")
            
        # Load baseline metrics
        baselines = self.load_baseline_metrics()
        if not baselines["relational"] or not baselines["textual"]:
            print("No baseline metrics available for comparison.")
            return plt.figure()
            
        # Create a temporary DataFrame for the visualization with computed gains
        plot_data_rows = []
        
        for idx, row in self.df.iterrows():
            # Skip rows without required info
            if pd.isna(row["textual_name"]) or pd.isna(row["relational_name"]) or pd.isna(row[metric]):
                continue
                
            # Try to find matching baseline metrics
            try:
                # Match with textual baseline
                textual_key = f"{row['textual_name']}/{row['textual_dim']}"
                if textual_key in baselines['textual'] and metric in baselines['textual'][textual_key]:
                    textual_baseline = baselines['textual'][textual_key][metric]
                    
                    # Match with relational baseline
                    rel_dim = str(row['relational_dim'])
                    if rel_dim in baselines['relational'] and metric in baselines['relational'][rel_dim]:
                        relational_baseline = baselines['relational'][rel_dim][metric]
                        
                        # Calculate performance gains
                        gain_over_textual = row[metric] - textual_baseline
                        gain_over_relational = row[metric] - relational_baseline
                        gain_over_best = row[metric] - max(textual_baseline, relational_baseline)
                        gain_over_avg = row[metric] - ((textual_baseline + relational_baseline) / 2)
                        
                        # Add row with calculated gains
                        plot_data_rows.append({
                            'model_type': row['model_type'],
                            metric: row[metric],
                            'gain_over_textual': gain_over_textual,
                            'gain_over_relational': gain_over_relational,
                            'gain_over_best': gain_over_best,
                            'gain_over_avg': gain_over_avg
                        })
            except Exception as e:
                # Skip rows with missing information or errors
                pass
                
        # Create DataFrame from collected rows
        if not plot_data_rows:
            print(f"No rows with matching baseline metrics found for {metric}.")
            return plt.figure()
            
        plot_df = pd.DataFrame(plot_data_rows)
        
        # Prepare data for grouped bar chart
        gain_metrics = ['gain_over_textual', 'gain_over_relational', 'gain_over_best', 'gain_over_avg']
        
        # Group by model type and calculate mean gains
        model_gains = plot_df.groupby('model_type')[gain_metrics].mean().reset_index()
        
        # Reshape for grouped bar chart
        melted_gains = pd.melt(
            model_gains,
            id_vars=['model_type'],
            value_vars=gain_metrics,
            var_name='Comparison',
            value_name='Gain'
        )
        
        # Map gain metric names to more readable labels
        comparison_labels = {
            'gain_over_textual': 'vs Textual Only',
            'gain_over_relational': 'vs Relational Only',
            'gain_over_best': 'vs Best Single',
            'gain_over_avg': 'vs Average Single'
        }
        melted_gains['Comparison'] = melted_gains['Comparison'].map(comparison_labels)
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(14, 8))
        
        sns.barplot(
            x='model_type',
            y='Gain',
            hue='Comparison',
            data=melted_gains,
            palette='viridis',
            ax=ax
        )
        
        # Add value labels on bars
        for p in ax.patches:
            height = p.get_height()
            if abs(height) > 0.001:  # Only label non-zero values
                ax.annotate(
                    f'{height:.3f}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', 
                    va='bottom' if height > 0 else 'top',
                    fontsize=8,
                    color='black' if height > 0 else 'red',
                    xytext=(0, 5 if height > 0 else -5),
                    textcoords='offset points'
                )
        
        # Add reference line at y=0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Color negative bars differently
        for patch in ax.patches:
            if patch.get_height() < 0:
                patch.set_facecolor('#d62728')  # Red
        
        metric_name = metric.replace('/', ' ').replace('_', ' ').title()
        ax.set_title(f'Performance Gains from Fusion Models vs Single-Modality Baselines ({metric_name})')
        ax.set_xlabel('Fusion Model Type')
        ax.set_ylabel(f'{metric_name} Gain (higher is better)')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        metric_suffix = metric.replace('/', '_')
        return self.save_and_return(fig, f"performance_gains_{metric_suffix}")

    def load_baseline_metrics(self):
        """
        Load baseline metrics for textual and relational embeddings.
        
        Returns:
            dict: Dictionary containing baseline metrics for textual and relational embeddings
                 with the structure:
                 {
                     'textual': {
                         'model_name/dim': {'acc/test': value, 'lp/auc': value, ...},
                         ...
                     },
                     'relational': {
                         'dim': {'acc/test': value, 'lp/auc': value, ...},
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
            textual_analyzer = TextualEmbeddingsAnalyzer(dpi=self.dpi, cmap=self.cmap, figsize=self.figsize)
            for _, row in textual_analyzer.df.iterrows():
                if 'model' in row and 'embedding_dim' in row:
                    # Create a unique key combining model name and dimension
                    key = f"{row['model']}/{row['embedding_dim']}"
                    baselines['textual'][key] = {}
                    
                    # Add all available metrics
                    for metric in ['acc/test', 'acc/valid', 'lp/auc', 'lp_hard/auc']:
                        if metric in row and not pd.isna(row[metric]):
                            baselines['textual'][key][metric] = row[metric]
        
        # Load Node2Vec relational baselines
        node2vec_path = "/home/lcheng/oz318/fusion/logs/Node2VecLightning"
        if Path(node2vec_path).exists():
            node2vec_analyzer = Node2VecAnalyzer(dpi=self.dpi, cmap=self.cmap, figsize=self.figsize)
            # Group by dimension and average the metrics
            if not node2vec_analyzer.df.empty:
                metrics = ['acc/test', 'acc/val', 'lp/auc', 'lp_hard/auc']
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
            asgc_analyzer = ASGCAnalyzer(dpi=self.dpi, cmap=self.cmap, figsize=self.figsize)
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
                    for metric in ['acc/test', 'acc/valid', 'lp/auc', 'lp_hard/auc']:
                        if metric in row and not pd.isna(row[metric]):
                            baselines['relational'][f"asgc_{dim}"] = {}
                            baselines['relational'][f"asgc_{dim}"][metric] = row[metric]
        
        return baselines

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
            results["seed_variance"] = self.visualize_seed_variance()
            print("Seed variance visualization created.")
        except Exception as e:
            print(f"Error creating seed variance visualization: {e}")
            
        try:
            results["val_test_comparison"] = self.compare_val_test()
            print("Validation vs test comparison created.")
        except Exception as e:
            print(f"Error creating validation vs test comparison: {e}")
        
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
                "lp/auc": ["mean", "std"]
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
        lp_aucs = [plot_data[plot_data['model'] == model]['lp/auc'].mean() for model in models]
        
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
        metrics = ['acc/valid', 'acc/test', 'lp/auc']
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



%matplotlib inline
# analyzer = ASGCAnalyzer()
# analyzer.run()
# analyzer = TextualEmbeddingsAnalyzer()
# analyzer.run()
# analyzer = Node2VecAnalyzer()
# analyzer.run()
# analyzer = EarlyFusionAnalyzer()
# analyzer.run()
# analyzer = GatedFusionAnalyzer()
# analyzer.run()
# analyzer = LowRankFusionAnalyzer()
# analyzer.run()
# analyzer = CrossModelAnalyzer()
# analyzer.run()
# baseline = analyzer.load_baseline_metrics()
# baseline
# analyzer.visualize_performance_distributions("lp_hard/auc")
# analyzer.run()