from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from .textual import TextualEmbeddingsAnalyzer
from .node2vec import Node2VecAnalyzer
from .asgc import ASGCAnalyzer
from pathlib import Path

from .base import AbstractAnalyzer
from .utils import load_baseline_metrics as load_utils_baseline_metrics


class CrossModelAnalyzer(AbstractAnalyzer):
    """Compare results across different fusion models."""

    def __init__(self, dpi=300, cmap="viridis", figsize=(6.4, 4.8)):
        # Use a common parent directory that contains all model types
        super().__init__("/home/lcheng/oz318/fusion/logs", dpi, cmap, figsize)

    def post_process(self):
        """Analyze and compare results across different fusion models."""
        self.df["model_type"] = self.df["path"].str.extract(r"/logs/([^/]+)/")
        return self.df

    def visualize(self):
        """Visualize comparison of different fusion models across all available metrics."""

        # Create a subplot for each metric
        num_metrics = len(self.metrics)
        fig, axes = plt.subplots(
            1, num_metrics, figsize=(12 * num_metrics / 2, 8), squeeze=False
        )

        # Plot each metric
        for i, metric in enumerate(self.metrics):
            ax = axes[0, i]
            sns.boxplot(x="model_type", y=metric, data=self.df, ax=ax)
            ax.set_title(
                f"Comparison of Fusion Models: {metric.replace('/', ' ').title()}"
            )
            ax.set_xlabel("Fusion Model")
            ax.set_ylabel(metric.replace("/", " ").title())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        return self.save_and_return(fig, "model_comparison")

    def visualize_by_embedding_type(self):
        """Compare model performance across different embedding combinations."""

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

        # Create a subplot for each metric
        num_metrics = len(self.metrics)
        fig, axes = plt.subplots(
            num_metrics, 1, figsize=(14, 8 * num_metrics), squeeze=False
        )

        # Plot each metric
        for i, metric in enumerate(self.metrics):
            ax = axes[i, 0]
            sns.boxplot(
                x="embedding_combo", y=metric, hue="model_type", data=self.df, ax=ax
            )
            ax.set_title(
                f"Model {metric.replace('/', ' ').title()} by Embedding Combination"
            )
            ax.set_xlabel("Embedding Combination (Textual_Relational)")
            ax.set_ylabel(metric.replace("/", " ").title())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            ax.legend(title="Fusion Model")

        plt.tight_layout()
        return self.save_and_return(fig, "embedding_type_comparison")

    def visualize_metrics_comparison(self):
        """Compare different metrics across models in a grouped bar chart."""

        # Calculate average metrics by model type
        avg_metrics = self.df.groupby("model_type")[self.metrics].mean().reset_index()

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))

        # Set up positions for grouped bars
        models = avg_metrics["model_type"].unique()
        x = np.arange(len(models))
        width = 0.8 / len(self.metrics)  # Adjust bar width based on number of metrics

        # Colors for different metrics
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Blue, orange, green

        # Create grouped bars
        for i, metric in enumerate(self.metrics):
            pos = x + width * (i - (len(self.metrics) - 1) / 2)
            bars = ax.bar(
                pos,
                avg_metrics[metric],
                width,
                label=metric.replace("/", " ").title(),
                color=colors[i % len(colors)],
            )

            # Add value labels on bars
            for bar_idx, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=colors[i % len(colors)],
                )

        ax.set_xlabel("Fusion Model")
        ax.set_ylabel("Performance")
        ax.set_title("Cross-Model Performance Metrics Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.legend(title="Metric")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        plt.tight_layout()
        return self.save_and_return(fig, "metrics_comparison")

    def visualize_fusion_heatmaps(self):
        """
        Create heatmap visualizations for each fusion method, comparing different textual and relational combinations.

        Returns:
            dict: Dictionary containing the three figures for test accuracy, lp_uniform/auc and lp_hard/auc
        """

        # Extract model type if not already done
        if "model_type" not in self.df.columns:
            self.df["model_type"] = self.df["path"].str.extract(r"/logs/([^/]+)/")

        # Extract textual and relational model names if not already done
        if "textual_name" not in self.df.columns:
            self.df["textual_name"] = self.df["path"].str.extract(
                r"/([^/]+)_[^/]+/\d+_\d+"
            )

        if "relational_name" not in self.df.columns:
            self.df["relational_name"] = self.df["path"].str.extract(
                r"/[^/]+_([^/]+)/\d+_\d+"
            )

        # Check if required columns are available
        required_cols = ["model_type", "textual_name", "relational_name"]
        for col in required_cols:
            if col not in self.df.columns or self.df[col].isnull().all():
                print(f"Missing required column: {col}")
                return {}

        # Filter out non-fusion models or rows with missing values
        fusion_models = self.df.dropna(subset=required_cols)

        # Create a figure for each metric
        results = {}

        for metric in self.metrics:
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
            fig, axes = plt.subplots(
                n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False
            )

            # Set title for entire figure
            metric_name = metric.replace("/", " ").replace("_", " ").title()
            fig.suptitle(
                f"Comparison of Fusion Methods: {metric_name}", fontsize=16, y=0.98
            )

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
                        .pivot(
                            index="textual_name",
                            columns="relational_name",
                            values=metric,
                        )
                    )

                    # Ensure all combinations have a value (even if not present in the data)
                    for tx in unique_textual_models:
                        if tx not in pivot_data.index:
                            pivot_data.loc[tx] = np.nan
                    for rel in unique_relational_models:
                        if rel not in pivot_data.columns:
                            pivot_data[rel] = np.nan

                    # Sort rows and columns for consistent ordering
                    pivot_data = pivot_data.loc[
                        sorted(pivot_data.index), sorted(pivot_data.columns)
                    ]

                    # Create heatmap
                    sns.heatmap(
                        pivot_data,
                        cmap=self.cmap,
                        annot=True,
                        fmt=".3f",
                        ax=ax,
                        linewidths=0.5,
                        cbar=False,
                    )

                    # Set title and labels
                    ax.set_title(f"{fusion_model}")
                    ax.set_xlabel("Relational Model")
                    ax.set_ylabel("Textual Model")

                    # Rotate x-axis labels for better readability
                    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

                except Exception as e:
                    # Handle case where pivot fails (e.g., no data for this combination)
                    ax.text(
                        0.5,
                        0.5,
                        f"No data available for\n{fusion_model}\n{e}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])

            # Hide unused subplots
            for i in range(n_fusion_models, n_rows * n_cols):
                row_idx = i // n_cols
                col_idx = i % n_cols
                axes[row_idx, col_idx].axis("off")

            # Add a colorbar at the bottom spanning all subplots
            cbar_ax = fig.add_axes(
                [0.15, 0.05, 0.7, 0.02]
            )  # [left, bottom, width, height]
            vmin = fusion_models[metric].min()
            vmax = fusion_models[metric].max()
            sm = plt.cm.ScalarMappable(
                cmap=self.cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)
            )
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
            cbar.set_label(metric_name)

            # Adjust layout
            plt.tight_layout(rect=[0, 0.07, 1, 0.96])  # [left, bottom, right, top]

            # Save figure
            results[metric] = self.save_and_return(
                fig, f"fusion_heatmap_{metric.replace('/', '_')}"
            )

        return results

    def run(self):
        """Run all available visualizations for cross-model analysis."""

        results = {"main": self.visualize()}
        results["embedding_type"] = self.visualize_by_embedding_type()

        results["metrics_comparison"] = self.visualize_metrics_comparison()

        # Add the new fusion heatmap visualizations
        fusion_heatmaps = self.visualize_fusion_heatmaps()
        for metric, fig in fusion_heatmaps.items():
            results[f"fusion_heatmap_{metric}"] = fig
        if fusion_heatmaps:
            print("Fusion heatmap visualizations created.")

        # Add performance tradeoffs visualization

        results["performance_tradeoffs"] = self.visualize_performance_tradeoffs()

        # Add metric correlations visualization
        results["metric_correlations"] = self.visualize_metric_correlations()

        # Create performance distributions for all available metrics
        for metric in self.metrics:
            metric_key = f"performance_distributions_{metric.replace('/', '_')}"
            results[metric_key] = self.visualize_performance_distributions(metric)

        return results

    def visualize_performance_tradeoffs(self):
        """
        Create a scatter plot showing trade-offs between node classification and link prediction performance.
        Each point represents a model/experiment, positioned by its accuracy vs LP performance.
        """

        fig, ax = plt.subplots(figsize=(10, 8))

        # Add model type column if not present
        if "model_type" not in self.df.columns:
            self.df["model_type"] = self.df["path"].str.extract(r"/logs/([^/]+)/")

        # Create scatter plot with different colors for different model types
        models = sorted(self.df["model_type"].unique())
        palette = sns.color_palette(n_colors=len(models))

        for i, model in enumerate(models):
            model_data = self.df[self.df["model_type"] == model]
            ax.scatter(
                model_data["acc/test"],
                model_data["lp_uniform/auc"],
                s=80,
                color=palette[i],
                alpha=0.7,
                label=model,
            )

        # Add diagonal line representing equal performance on both tasks
        min_val = min(self.df["acc/test"].min(), self.df["lp_uniform/auc"].min())
        max_val = max(self.df["acc/test"].max(), self.df["lp_uniform/auc"].max())
        ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.3)

        # Add labels for interesting points
        top_models = self.df.nlargest(5, "acc/test")
        for _, row in top_models.iterrows():
            ax.annotate(
                f"{row['model_type']}",
                (row["acc/test"], row["lp_uniform/auc"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        ax.set_xlabel("Node Classification Accuracy")
        ax.set_ylabel("Link Prediction AUC")
        ax.set_title("Performance Trade-offs: Node Classification vs. Link Prediction")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Model Type")

        return self.save_and_return(fig, "performance_tradeoffs")

    def visualize_metric_correlations(self):
        """
        Create a heatmap showing correlations between different performance metrics.
        """

        # Identify metrics columns (those containing '/' in their name)
        metric_cols = [col for col in self.df.columns if "/" in col]

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
            ax=ax,
        )

        ax.set_title("Correlation Matrix of Performance Metrics")

        return self.save_and_return(fig, "metric_correlations")

    def visualize_performance_distributions(self, metric="acc/test"):
        """
        Visualize the performance distribution across different models using violin plots.
        Highlights the trial with maximum performance for each model type.

        Args:
            metric: The performance metric to visualize
        """

        # Add model type column if not present
        if "model_type" not in self.df.columns:
            self.df["model_type"] = self.df["path"].str.extract(r"/logs/([^/]+)/")

        fig, ax = plt.subplots(figsize=(12, 7))

        # Create violin plot
        sns.violinplot(
            x="model_type",
            y=metric,
            data=self.df,
            palette="Set3",
            inner="box",  # Show box plot inside violin
            ax=ax,
        )

        # Add individual data points
        sns.stripplot(
            x="model_type",
            y=metric,
            data=self.df,
            size=4,
            alpha=0.4,
            jitter=True,
            ax=ax,
        )

        # Add mean values as text
        for i, model in enumerate(sorted(self.df["model_type"].unique())):
            model_data = self.df[self.df["model_type"] == model]
            mean_val = model_data[metric].mean()
            ax.text(i, mean_val + 0.01, f"μ={mean_val:.3f}", ha="center", fontsize=9)

        # Highlight max value for each model type
        for i, model in enumerate(sorted(self.df["model_type"].unique())):
            model_data = self.df[self.df["model_type"] == model]
            if not model_data.empty:
                # Get the maximum value and its index
                max_val = model_data[metric].max()
                max_idx = model_data[metric].idxmax()

                # Calculate position
                x_pos = i
                y_pos = max_val

                # Highlight the max value point with a red circle
                ax.plot(x_pos, y_pos, "o", ms=10, mfc="none", mec="red", mew=2)

                # Add annotation for max value
                ax.text(
                    x_pos,
                    y_pos + 0.01,
                    f"max={max_val:.3f}",
                    ha="center",
                    fontsize=9,
                    color="red",
                    weight="bold",
                )

        ax.set_xlabel("Model Type")
        ax.set_ylabel(f"{metric} Performance")
        ax.set_title(f"Distribution of {metric} Performance by Model Type")
        plt.xticks(rotation=45, ha="right")

        return self.save_and_return(
            fig, f"performance_distribution_{metric}".replace("/", "_")
        )

    def load_baseline_metrics(self):
        """
        Load baseline metrics for textual and relational embeddings.

        Returns:
            dict: Dictionary containing baseline metrics for textual and relational embeddings
        """
        # Load baseline metrics from utils
        baseline_df = load_utils_baseline_metrics()

        # Convert DataFrame to the expected dictionary format
        baselines = {"textual": {}, "relational": {}}

        # Process textual models
        textual_models = baseline_df[baseline_df["type"] == "textual"]
        for _, row in textual_models.iterrows():
            model_key = f"{row['name']}/{row['dim']}"
            baselines["textual"][model_key] = {
                metric: row[metric]
                for metric in self.metrics
                if metric in row and not pd.isna(row[metric])
            }

        # Process relational models
        relational_models = baseline_df[baseline_df["type"] == "relational"]
        for _, row in relational_models.iterrows():
            # Use dimension as the key for relational models
            dim_key = str(row["dim"])
            if dim_key not in baselines["relational"]:
                baselines["relational"][dim_key] = {}

            # Add metrics for this model
            metrics_dict = {
                metric: row[metric]
                for metric in self.metrics
                if metric in row and not pd.isna(row[metric])
            }

            # Update with better metrics if we have multiple models with same dim
            for metric, value in metrics_dict.items():
                if (
                    metric not in baselines["relational"][dim_key]
                    or value > baselines["relational"][dim_key][metric]
                ):
                    baselines["relational"][dim_key][metric] = value

        return baselines
