from pathlib import Path
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import constants

import json
from pathlib import Path
from abc import ABC, abstractmethod

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
