import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from .base import FusionAnalyzer


class TransformerFusionAnalyzer(FusionAnalyzer):
    def __init__(
        self,
        dpi=300,
        cmap="viridis",
        figsize=(6.4, 4.8),
    ):
        super().__init__(
            "TransformerFusion",
            dpi,
            cmap,
            figsize,
        )

    def post_process(self):
        self.df[
            [
                "textual_name",
                "relational_name",
                "textual_dim",
                "relational_dim",
                "latent_dim",
                "num_layers",
                "nhead",
                "output_modality",
            ]
        ] = self.df.prefix.str.split("[/_]").tolist()
        self.df = self.df.drop(columns=["prefix"])

    def analyze(self):
        """Analyze with additional focus on transformer-specific parameters."""
        if self.df.empty:
            print("No data to analyze.")
            return None
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
        """Visualize the impact of output modality on model performance across multiple metrics."""
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

        # Define metrics to visualize
        # metrics = ["acc/test", "lp_uniform/auc", "lp_hard/auc"]

        # Get unique output modalities
        output_modalities = self.df["output_modality"].unique()
        n_modalities = len(output_modalities)

        # Create figure with subplot grid: metrics as rows, modalities as columns
        fig, axes = plt.subplots(
            len(self.metrics),
            n_modalities,
            figsize=(6 * n_modalities, 5 * len(self.metrics)),
            squeeze=False,
        )

        # Create a separate subplot for each combination of metric and modality
        for i, metric in enumerate(self.metrics):
            # Create a row of plots that share the y-axis
            for j, modality in enumerate(output_modalities):
                modality_data = self.df[self.df["output_modality"] == modality]

                # Skip if metric doesn't exist in dataframe
                if metric not in modality_data.columns:
                    axes[i, j].text(
                        0.5,
                        0.5,
                        f"No data for {metric}",
                        ha="center",
                        va="center",
                        transform=axes[i, j].transAxes,
                    )
                    continue

                # Filter out rows with NaN values for this metric
                valid_data = modality_data[~modality_data[metric].isna()]

                if valid_data.empty:
                    axes[i, j].text(
                        0.5,
                        0.5,
                        f"No valid data for {metric}",
                        ha="center",
                        va="center",
                        transform=axes[i, j].transAxes,
                    )
                    continue

                # Create boxplot for this modality and metric
                sns.boxplot(x="latent_dim", y=metric, data=valid_data, ax=axes[i, j])

                # Set titles and labels
                if i == 0:  # Only set modality title on the top row
                    axes[i, j].set_title(f"Output Modality: {modality}")

                if j == 0:  # Only set y-label for leftmost plots
                    axes[i, j].set_ylabel(metric)
                else:
                    axes[i, j].set_ylabel("")

                if i == len(self.metrics) - 1:  # Only set x-label for bottom row
                    axes[i, j].set_xlabel("Latent Dimension")
                else:
                    axes[i, j].set_xlabel("")

                # Share y-axis limits within each row (same metric)
                if j > 0:
                    axes[i, j].sharey(axes[i, 0])

        plt.suptitle(
            "Transformer Fusion: Performance Metrics by Output Modality", fontsize=16
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle

        return self.save_and_return(fig, "output_modality_impact")

    def visualize_architecture_impact(self):
        """Visualize the impact of different output modalities across tasks."""

        # Use textual_name as a proxy for different tasks
        # Get unique tasks to plot
        tasks = self.df["textual_name"].unique()

        # Create a figure with subplots based on number of tasks
        fig, axes = plt.subplots(len(tasks), 1, figsize=(12, 4 * len(tasks)))
        if len(tasks) == 1:
            axes = [axes]  # Convert to list if only one task

        for i, task in enumerate(tasks):
            task_data = self.df[self.df["textual_name"] == task]

            # Create boxplot for this task
            sns.boxplot(
                x="output_modality",
                y="acc/test",
                hue="relational_name",  # Use relational_name to differentiate data types
                data=task_data,
                ax=axes[i],
            )
            axes[i].set_title(f"Impact of Output Modality for Task: {task}")
            axes[i].set_xlabel("Output Modality")
            axes[i].set_ylabel("Test Accuracy")
            axes[i].legend(title="Relational Data Type")

        plt.tight_layout()
        return self.save_and_return(fig, "output_modality_by_task")

    def analyze_parameter_importance(self, metrics=None):
        """
        Quantifies and ranks which parameters have the strongest effect on performance.

        Args:
            metrics (list, optional): Performance metrics to analyze. Defaults to ['acc/test'].

        Returns:
            dict: Dictionary mapping metrics to DataFrames of ranked parameters by importance.
        """

        # Default metrics if none provided
        metrics = ["acc/test", "lp_uniform/auc", "lp_hard/auc"]

        parameters = [
            "latent_dim",
            "num_layers",
            "nhead",
            "output_modality",
            "textual_dim",
            "relational_dim",
            "textual_name",
            "relational_name",
        ]

        results = {}
        figures = {}

        for metric in metrics:

            # Get data with non-null values for this metric
            metric_data = self.df[~self.df[metric].isna()].copy()

            # Filter out rows with infinite values in the metric column
            metric_data = metric_data[np.isfinite(metric_data[metric])]

            # Filter out rows with NaN or infinite values in numeric columns
            numeric_cols = metric_data.select_dtypes(include=np.number).columns
            finite_mask = np.all(np.isfinite(metric_data[numeric_cols]), axis=1)
            metric_data = metric_data[finite_mask]

            # Calculate effect sizes and p-values for each parameter
            param_effects = []

            for param in parameters:

                if param not in metric_data.columns:
                    continue

                # Convert parameter to categorical if it isn't already
                if metric_data[param].dtype == "object" or isinstance(
                    metric_data[param].dtype, pd.CategoricalDtype
                ):
                    groups = metric_data.groupby(param)[metric]
                else:
                    # For numerical parameters, bin them into quartiles
                    metric_data[f"{param}_binned"] = pd.qcut(
                        metric_data[param], 4, duplicates="drop"
                    )
                    groups = metric_data.groupby(f"{param}_binned")[metric]

                # Calculate metrics for parameter importance
                group_values = [group[1].values for group in groups]

                # Skip if we have fewer than 2 groups or any group is empty
                if len(group_values) < 2 or any(len(g) == 0 for g in group_values):
                    continue

                # Effect size: max group mean - min group mean (normalized by overall std)
                group_means = [g.mean() for g in group_values if len(g) > 0]
                overall_std = metric_data[metric].std()
                effect_size = (
                    (max(group_means) - min(group_means)) / overall_std
                    if overall_std > 0
                    else 0
                )

                # Range of means
                mean_range = max(group_means) - min(group_means) if group_means else 0

                # Variance explained: ratio of between-group variance to total variance
                grand_mean = metric_data[metric].mean()
                total_variance = sum(
                    (val - grand_mean) ** 2 for g in group_values for val in g
                )
                between_variance = sum(
                    len(g) * ((g.mean() - grand_mean) ** 2)
                    for g in group_values
                    if len(g) > 0
                )
                variance_explained = (
                    between_variance / total_variance if total_variance > 0 else 0
                )

                param_effects.append(
                    {
                        "parameter": param,
                        "effect_size": effect_size,
                        "mean_range": mean_range,
                        "variance_explained": variance_explained,
                        "num_groups": len(group_values),
                    }
                )

            # Create DataFrame and sort by importance

            effects_df = pd.DataFrame(param_effects)
            effects_df = effects_df.sort_values("variance_explained", ascending=False)
            results[metric] = effects_df

            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x="parameter", y="variance_explained", data=effects_df, ax=ax)
            ax.set_title(f"Parameter Importance for {metric}")
            ax.set_xlabel("Parameter")
            ax.set_ylabel("Variance Explained")
            plt.xticks(rotation=45)
            plt.tight_layout()

            figures[metric] = self.save_and_return(
                fig, f"parameter_importance_{metric.replace('/', '_')}"
            )

        return {"importance_data": results, "importance_figures": figures}

    def run(self):
        """Run all available visualizations for transformer fusion analysis."""

        results = super().run()

        results["output_modality"] = self.visualize_output_modality()
        results["architecture_impact"] = self.visualize_architecture_impact()
        results["parameter_importance"] = self.analyze_parameter_importance(
            self.metrics
        )

        return results
