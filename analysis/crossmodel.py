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

from typing import Callable


class CrossModelAnalyzer(AbstractAnalyzer):
    """Compare results across different fusion models."""

    def __init__(
        self,
        dpi=300,
        cmap="viridis",
        figsize=(6.4, 4.8),
        remove_transformer_fusion=True,
        mask=None
    ):
        # Use a common parent directory that contains all model types
        # self.mask = mask
        self.mask = mask
        self.remove_transformer_fusion = remove_transformer_fusion
        super().__init__("/home/lcheng/oz318/fusion/logs", dpi, cmap, figsize)

    def post_process(self):
        """Analyze and compare results across different fusion models."""
        df = self._df.copy()
        df["model_type"] = df["path"].str.extract(r"/logs/([^/]+)/")

        transformer_fusion = df[df.model_type == "TransformerFusion"]
        transformer_fusion["output_modality"] = transformer_fusion.kwargs.map(
            lambda x: x["output_modality"]
        )

        transformer_fusion["model_type"] = transformer_fusion.apply(
            lambda row: f"TransformerFusion/{row['output_modality']}", axis=1
        )

        df.update(transformer_fusion)

        df["model_type"] = df["path"].str.extract(r"/logs/([^/]+)/")

        df["textual_name"] = df["path"].str.extract(r"/([^/]+)_[^/]+/\d+_\d+")
        df["relational_name"] = df["path"].str.extract(
            r"/[^/]+_([^/]+)/\d+_\d+"
        )

        # Create a composite embedding type
        df["embedding_combo"] = (
            df["textual_name"] + "_" + df["relational_name"]
        )
        # if self.mask is s

        if self.remove_transformer_fusion:
            df = df[~df.model_type.str.startswith("TransformerFusion")]

        return df
        
    @property
    def df(self):
        if self.mask is None:
            return self._df
        else:
            return self._df[self.mask]

        
    def per_embedding_combo(self):
        unique_embedding_combos = self._df.embedding_combo.dropna().unique()
        results = {}
        for embedding_combo in unique_embedding_combos:
            self.mask = (
                (self._df.embedding_combo == embedding_combo)
                | self.baseline_mask
            )
            results[f"per_embedding_combo/{embedding_combo}"] = self.save_and_return(self.visualize_performance_distributions_subplots(), f"per_embedding_combo/{embedding_combo}")
        self.mask = None
        return results
    
    def per_textual_model(self):
        unique_textual_names = [
            i.name
            for i in Path("/home/lcheng/oz318/fusion/logs/TextualEmbeddings").iterdir()
            if i.is_dir()
        ]
        results = {}
        for textual_name in unique_textual_names:
            self.mask = (
                (self._df.textual_name == textual_name)
                | (self._df.prefix == textual_name)
                | (self._df.model_type.isin(["Node2VecLightning", "ASGC"])))
            # analyzer = CrossModelAnalyzer(mask=f)
            results[f"per_textual_model/{textual_name}"] = self.save_and_return(self.visualize_performance_distributions_subplots(), f"per_textual_model/{textual_name}")
        self.mask = None
        return results

    
    def per_latent_dim(self):
        unique_latent_dims = [32, 64, 128, 256]
        results = {}
        for latent_dim in unique_latent_dims:
            self.mask = (self._df.latent_dim == latent_dim) | self.baseline_mask
            results[f"per_latent_dim/{latent_dim}"] = self.save_and_return(self.visualize_performance_distributions_subplots(), f"per_latent_dim/{latent_dim}")
        self.mask = None
        return results

    @property
    def baseline_mask(self):
        return (self._df.model_type.isin(["Node2VecLightning", "ASGC", "TextualEmbeddings"]))


    def visualize_by_embedding_type(self):
        """Compare model performance across different embedding combinations."""

        # Extract model type and embedding info

        # Create a subplot for each metric
        df = self.df
        num_metrics = len(self.metrics)
        fig, axes = plt.subplots(
            num_metrics, 1, figsize=(14, 8 * num_metrics), squeeze=False
        )

        # Plot each metric
        for i, metric in enumerate(self.metrics):
            ax = axes[i, 0]
            sns.boxplot(
                x="embedding_combo", y=metric, hue="model_type", data=df, ax=ax
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
        df = self.df
        avg_metrics = df.groupby("model_type")[self.metrics].mean().reset_index()

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

        # Check if required columns are available
        required_cols = ["model_type", "textual_name", "relational_name"]
        df = self.df
        for col in required_cols:
            if col not in df.columns or df[col].isnull().all():
                print(f"Missing required column: {col}")
                return {}

        # Filter out non-fusion models or rows with missing values
        fusion_models = df.dropna(subset=required_cols)

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



    def visualize_performance_distributions_subplots(self):
        """
        Visualize the performance distribution across different models using violin plots for all metrics.
        Each metric is displayed in its own subplot within a single figure.
        Models are categorized into three groups: textual baselines, relational baselines, and fusion methods.
        """
        # Add model type column if not present

        # Fixed layout: 1 column, 3 rows
        df = self.df
        n_cols = 1
        n_rows = 3

        # Create figure with appropriate size - wider layout for single column
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(14, 6 * n_rows), squeeze=False
        )

        # Flatten axes for easy iteration
        axes_flat = axes.flatten()

        # Define model categories
        textual_models = ["TextualEmbeddings"]
        relational_models = ["ASGC", "Node2VecLightning"]

        # Determine which models are fusion models (not in textual or relational)
        all_models = df["model_type"].unique()
        fusion_models = [
            m
            for m in all_models
            if m not in textual_models and m not in relational_models
        ]

        # Create a new column for model category
        df["model_category"] = "Fusion"
        df.loc[df["model_type"].isin(textual_models), "model_category"] = (
            "Textual"
        )
        df.loc[df["model_type"].isin(relational_models), "model_category"] = (
            "Relational"
        )

        # Sort the dataframe to ensure consistent ordering of categories and models
        category_order = ["Textual", "Relational", "Fusion"]
        df["category_order"] = df["model_category"].map(
            {cat: i for i, cat in enumerate(category_order)}
        )
        df = df.sort_values(["category_order", "model_type"])

        for i, metric in enumerate(self.metrics):
            ax = axes_flat[i]

            # Create violin plot with categorical x-axis
            sns.violinplot(
                x="model_type",
                y=metric,
                data=df,
                palette="Set3",
                inner="box",  # Show box plot inside violin
                ax=ax,
                order=sorted(textual_models)
                + sorted(relational_models)
                + sorted(fusion_models),
            )

            # Add individual data points
            sns.stripplot(
                x="model_type",
                y=metric,
                data=df,
                size=4,
                alpha=0.4,
                jitter=True,
                ax=ax,
                order=sorted(textual_models)
                + sorted(relational_models)
                + sorted(fusion_models),
            )

            # Add visual separation between categories (vertical lines)
            if textual_models:
                ax.axvline(
                    len(textual_models) - 0.5, color="gray", linestyle="--", alpha=0.5
                )
            if relational_models:
                ax.axvline(
                    len(textual_models) + len(relational_models) - 0.5,
                    color="gray",
                    linestyle="--",
                    alpha=0.5,
                )

            # Add category labels above the plot
            if textual_models:
                mid_textual = (len(textual_models) - 1) / 2
                ax.text(
                    mid_textual,
                    ax.get_ylim()[1] * 1.05,
                    "Textual Baselines",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            if relational_models:
                mid_relational = len(textual_models) + (len(relational_models) - 1) / 2
                ax.text(
                    mid_relational,
                    ax.get_ylim()[1] * 1.05,
                    "Relational Baselines",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            if fusion_models:
                mid_fusion = (
                    len(textual_models)
                    + len(relational_models)
                    + (len(fusion_models) - 1) / 2
                )
                ax.text(
                    mid_fusion,
                    ax.get_ylim()[1] * 1.05,
                    "Fusion Methods",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            # Highlight max value for each model type
            for j, model in enumerate(
                sorted(textual_models)
                + sorted(relational_models)
                + sorted(fusion_models)
            ):
                model_data = df[df["model_type"] == model]
                if not model_data.empty:
                    # Get the maximum value
                    max_val = model_data[metric].max()
                    mean_val = model_data[metric].mean()

                    # Calculate position
                    x_pos = j
                    y_pos = max_val

                    # Add mean values as text
                    ax.text(
                        x_pos,
                        mean_val + 0.01,
                        f"μ={mean_val:.3f}",
                        ha="center",
                        fontsize=8,
                    )

                    # Highlight the max value point with a red circle
                    ax.plot(x_pos, y_pos, "o", ms=8, mfc="none", mec="red", mew=1.5)

                    # Add annotation for max value
                    ax.text(
                        x_pos,
                        y_pos + 0.01,
                        f"max={max_val:.3f}",
                        ha="center",
                        fontsize=8,
                        color="red",
                        weight="bold",
                    )

            ax.set_title(f"{metric.replace('/', ' ').title()}")
            ax.set_xlabel("")  # Remove individual x-labels for cleaner look
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

        # Hide unused subplots
        for i in range(len(self.metrics), n_rows * n_cols):
            axes_flat[i].set_visible(False)

        # Add common labels
        fig.text(0.5, 0.02, "Model Type", ha="center", va="center", fontsize=12)
        fig.text(
            0.02,
            0.5,
            "Performance",
            ha="center",
            va="center",
            rotation="vertical",
            fontsize=12,
        )
        fig.suptitle(
            "Performance Distribution by Model Type Across All Metrics", fontsize=14
        )

        plt.tight_layout(rect=[0.03, 0.03, 1, 0.97])

        return self.save_and_return(fig, "performance_distributions_all_metrics")

    def visualize_performance_tradeoffs(self):
        """
        Create three scatter plots showing trade-offs between different performance metrics.
        Each subplot represents a different combination of metrics:
        1. Node classification vs Uniform link prediction
        2. Node classification vs Hard link prediction
        3. Uniform link prediction vs Hard link prediction
        """
        # Create a 1x3 subplot figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Define metric pairs for each subplot
        metric_pairs = [
            (
                "acc/test",
                "lp_uniform/auc",
                "Node Classification vs Uniform Link Prediction",
            ),
            ("acc/test", "lp_hard/auc", "Node Classification vs Hard Link Prediction"),
            (
                "lp_uniform/auc",
                "lp_hard/auc",
                "Uniform Link Prediction vs Hard Link Prediction",
            ),
        ]
        df = self.df

        # Create models and color palette once
        models = sorted(df["model_type"].unique())
        palette = sns.color_palette(n_colors=len(models))

        # Create scatter plots for each metric pair
        for i, (x_metric, y_metric, title) in enumerate(metric_pairs):
            ax = axes[i]

            for j, model in enumerate(models):
                model_data = df[df["model_type"] == model]
                ax.scatter(
                    model_data[x_metric],
                    model_data[y_metric],
                    s=80,
                    color=palette[j],
                    alpha=0.7,
                    label=model if i == 0 else None,  # Only add label in first subplot
                )

            # Add diagonal line
            min_val = min(df[x_metric].min(), df[y_metric].min())
            max_val = max(df[x_metric].max(), df[y_metric].max())
            ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.3)

            # Add labels for top performers
            top_models = df.nlargest(3, y_metric)
            for _, row in top_models.iterrows():
                ax.annotate(
                    f"{row['model_type']}",
                    (row[x_metric], row[y_metric]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

            # Set labels and title
            ax.set_xlabel(x_metric.replace("/", " ").title())
            ax.set_ylabel(y_metric.replace("/", " ").title())
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        # Add a single legend for the entire figure
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            title="Model Type",
            loc="lower center",
            ncol=len(models),
            bbox_to_anchor=(0.5, -0.05),
        )

        fig.suptitle("Performance Trade-offs Between Different Metrics", fontsize=16)
        plt.tight_layout(
            rect=[0, 0.05, 1, 0.95]
        )  # Adjust to make room for legend and title

        return self.save_and_return(fig, "performance_tradeoffs")

    def visualize_metric_correlations(self):
        """
        Create a heatmap showing correlations between different performance metrics.
        """
        df = self.df
        # Identify metrics columns (those containing '/' in their name)
        metric_cols = [col for col in df.columns if "/" in col]

        # Calculate correlation matrix
        corr_matrix = df[metric_cols].corr()

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
    def run(self):
        """Run all available visualizations for cross-model analysis."""

        results = {}
        results["embedding_type"] = self.visualize_by_embedding_type()

        results["metrics_comparison"] = self.visualize_metrics_comparison()

        # Add the new fusion heatmap visualizations
        results.update(self.visualize_fusion_heatmaps())

        # Add performance tradeoffs visualization
        results["performance_tradeoffs"] = self.visualize_performance_tradeoffs()

        # Add metric correlations visualization
        results["metric_correlations"] = self.visualize_metric_correlations()

        # Create subplots for all performance distributions in a single figure
        results["performance_distributions"] = self.visualize_performance_distributions_subplots()

        results.update(self.per_textual_model())
        
        results.update(self.per_latent_dim())
        
        return results