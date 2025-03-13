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


class AbstractAnalyzer(ABC):
    def __init__(self, base_path, dpi=300, cmap="viridis", figsize=(6.4, 4.8)):
        self.base_path = base_path
        self.dpi = dpi
        self.figsize = figsize
        self.cmap = sns.color_palette(cmap, as_cmap=True)
        self.config_paths = list(Path(self.base_path).glob("**/config.json"))
        self.embeddings_paths = [i.parent / "embeddings.pt" for i in self.config_paths]
        self.results_paths = [i.parent / "results.json" for i in self.config_paths]

        self.df = self._create_results_df()
        self.post_process()

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


class ASGC(AbstractAnalyzer):
    def __init__(self, dpi=300, cmap="viridis"):
        super().__init__("/home/lcheng/oz318/fusion/logs/ASGC", dpi, cmap)

    def post_process(self):
        self.df[["dim", "reg"]] = self.pd.DataFrame(
            self.df.prefix.map(lambda x: x.split("/")).tolist(), index=self.df.index
        )
        self.df = self.df.drop(columns=["prefix"])

    def analyze(self):
        """Analyze ASGC results by grouping by dim, k and reg parameters."""
        if self.df.empty:
            print("No data to analyze.")
            return None

        result = (
            self.df.groupby(["dim", "k", "reg"])
            .agg({"acc/test": ["mean", "std"]})
            .reset_index()
        )

        # Format for better readability
        result.columns = ["_".join(col).strip() for col in result.columns.values]
        return result

    def visualize(self):
        """Visualize ASGC results as a heatmap of test accuracy vs k and reg for each dimension."""
        if self.df.empty:
            print("No data to visualize.")
            return plt.figure()

        # Get unique dimensions
        dimensions = self.df["dim"].unique()

        if len(dimensions) > 1:
            # Create a figure with a 2x2 grid for different dimensions
            n_dims = len(dimensions)
            n_cols = min(2, n_dims)
            n_rows = (
                n_dims + n_cols - 1
            ) // n_cols  # Ceiling division for number of rows

            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(self.figsize[0] * n_cols, self.figsize[1] * n_rows),
                squeeze=False,
            )

            for i, dim in enumerate(sorted(dimensions)):
                # Calculate row and column for this dimension
                row = i // n_cols
                col = i % n_cols

                # Filter data for this dimension
                dim_df = self.df[self.df["dim"] == dim]

                # Create pivot table for this dimension
                pivot_data = (
                    dim_df.groupby(["k", "reg"])
                    .agg({"acc/test": "mean"})
                    .reset_index()
                    .pivot(index="k", columns="reg", values="acc/test")
                )

                # Plot heatmap
                ax = axes[row, col]
                sns.heatmap(pivot_data, cmap=self.cmap, annot=True, fmt=".3f", ax=ax)
                ax.set_title(f"ASGC Test Accuracy (dim={dim})")
                ax.set_xlabel("Regularization")
                ax.set_ylabel("k")

            # Hide empty subplots if any
            for i in range(len(dimensions), n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                axes[row, col].axis("off")

            plt.tight_layout()
            return fig

        else:
            # Single dimension case - use original code
            fig, ax = plt.subplots(figsize=self.figsize)
            pivot_data = (
                self.df.groupby(["k", "reg"])
                .agg({"acc/test": "mean"})
                .reset_index()
                .pivot(index="k", columns="reg", values="acc/test")
            )

            sns.heatmap(pivot_data, cmap=self.cmap, annot=True, fmt=".3f", ax=ax)
            ax.set_title(f"ASGC Test Accuracy (dim={dimensions[0]})")
            ax.set_xlabel("Regularization")
            ax.set_ylabel("k")

            return fig


class FusionAnalyzer(AbstractAnalyzer):
    """Base class for analyzing fusion experiments."""

    def __init__(self, experiment_type, dpi=300, cmap="viridis", figsize=(6.4, 4.8)):
        self.experiment_type = experiment_type
        base_path = f"/home/lcheng/oz318/fusion/logs/{experiment_type}"
        super().__init__(base_path, dpi, cmap, figsize)
    def post_process(self):
        self.df[["textual_name", "textual_dim", "relational_name", "relational_dim"]] = self.df.prefix.str.split(
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

    def visualize(self, focus_param="latent_dim", fixed_params=None):
        """
        Visualize fusion results focusing on a specific parameter.

        Args:
            focus_param: Parameter to analyze variation (e.g., "latent_dim")
            fixed_params: Dictionary of parameters to fix (e.g., {"textual_name": "bert"})
        """
        if self.df.empty:
            print("No data to visualize.")
            return plt.figure()

        # Filter data if fixed parameters are provided
        plot_df = self.df.copy()
        if fixed_params:
            for param, value in fixed_params.items():
                plot_df = plot_df[plot_df[param] == value]

        if plot_df.empty:
            print("No data matches the fixed parameters.")
            return plt.figure()

        # Create visualization based on focus parameter
        fig, ax = plt.subplots(figsize=(12, 8))

        if focus_param == "latent_dim":
            sns.boxplot(x="latent_dim", y="acc/test", data=plot_df, ax=ax)
            ax.set_title(f"{self.experiment_type}: Test Accuracy vs Latent Dimension")
            ax.set_xlabel("Latent Dimension")
            ax.set_ylabel("Test Accuracy")

        elif focus_param == "textual_relational":
            # Create a composite plot showing comparison between textual and relational embeddings
            pivot_data = (
                plot_df.groupby(["textual_name", "relational_name"])
                .agg({"acc/test": "mean"})
                .reset_index()
                .pivot(
                    index="textual_name", columns="relational_name", values="acc/test"
                )
            )
            sns.heatmap(pivot_data, cmap=self.cmap, annot=True, fmt=".3f", ax=ax)
            ax.set_title(f"{self.experiment_type}: Test Accuracy by Embedding Type")
            ax.set_xlabel("Relational Embedding")
            ax.set_ylabel("Textual Embedding")

        return fig


class AdditionFusionAnalyzer(FusionAnalyzer):
    def __init__(self, dpi=300, cmap="viridis", figsize=(6.4, 4.8)):
        super().__init__("AdditionFusion", dpi, cmap, figsize)


class EarlyFusionAnalyzer(FusionAnalyzer):
    def __init__(self, dpi=300, cmap="viridis", figsize=(6.4, 4.8)):
        super().__init__("EarlyFusion", dpi, cmap, figsize)


class GatedFusionAnalyzer(FusionAnalyzer):
    def __init__(self, dpi=300, cmap="viridis", figsize=(6.4, 4.8)):
        super().__init__("GatedFusion", dpi, cmap, figsize)

    def analyze_gate_scores(self):
        """Analyze the distribution of gate scores."""
        gate_scores = []

        for i, path in enumerate(self.results_paths):
            result_path = path.parent / "gate_scores.json"
            if result_path.exists():
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

    def visualize_gate_distribution(self, num_samples=5):
        """Visualize the distribution of gate scores for a few samples."""
        gate_data = self.analyze_gate_scores()
        if gate_data.empty:
            print("No gate score data available.")
            return plt.figure()

        # Sample a few experiments
        if len(gate_data) > num_samples:
            gate_data = gate_data.sample(num_samples, random_state=42)

        fig, axes = plt.subplots(
            len(gate_data), 1, figsize=(10, 4 * len(gate_data)), squeeze=False
        )

        for i, (_, row) in enumerate(gate_data.iterrows()):
            ax = axes[i, 0]
            sns.histplot(row["scores"], ax=ax, bins=50, kde=True)
            ax.set_title(f"Gate Score Distribution: {Path(row['path']).name}")
            ax.set_xlabel("Gate Score")
            ax.set_ylabel("Count")
            ax.axvline(x=0.5, color="r", linestyle="--", label="Equal weight")
            ax.legend()

        plt.tight_layout()
        return fig


class LowRankFusionAnalyzer(FusionAnalyzer):
    def __init__(self, dpi=300, cmap="viridis", figsize=(6.4, 4.8)):
        super().__init__("LowRankFusion", dpi, cmap, figsize)

    def analyze(self):
        """Analyze with additional focus on rank parameter."""
        if self.df.empty:
            print("No data to analyze.")
            return None

        # Extract rank from path
        self.df["rank"] = self.df["path"].str.extract(r"/\d+_\d+/(\d+)").astype(int)

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

    def visualize_rank_impact(self):
        """Visualize the impact of rank on model performance."""
        if self.df.empty:
            print("No data to visualize.")
            return plt.figure()

        # Extract rank if not already done
        if "rank" not in self.df.columns:
            self.df["rank"] = self.df["path"].str.extract(r"/\d+_\d+/(\d+)").astype(int)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x="rank", y="acc/test", hue="latent_dim", data=self.df, ax=ax)
        ax.set_title("Low Rank Fusion: Test Accuracy by Rank")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Test Accuracy")
        ax.legend(title="Latent Dim")

        return fig


class TransformerFusionAnalyzer(FusionAnalyzer):
    def __init__(self, dpi=300, cmap="viridis", figsize=(6.4, 4.8)):
        super().__init__("TransformerFusion", dpi, cmap, figsize)

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

        return fig

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
        return fig


class CrossModelAnalyzer(AbstractAnalyzer):
    """Compare results across different fusion models."""

    def __init__(self, dpi=300, cmap="viridis", figsize=(6.4, 4.8)):
        # Use a common parent directory that contains all model types
        super().__init__("/home/lcheng/oz318/fusion/logs", dpi, cmap, figsize)

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
            .agg({"acc/test": ["mean", "std", "max", "min", "count"]})
            .reset_index()
        )

        # Format column names
        model_comparison.columns = [
            "_".join(col).strip() if col[1] else col[0]
            for col in model_comparison.columns.values
        ]

        return model_comparison

    def visualize(self):
        """Visualize comparison of different fusion models."""
        if self.df.empty:
            print("No data to visualize.")
            return plt.figure()

        # Extract model type if not already done
        if "model_type" not in self.df.columns:
            self.df["model_type"] = self.df["path"].str.extract(r"/logs/([^/]+)/")

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(x="model_type", y="acc/test", data=self.df, ax=ax)
        ax.set_title("Comparison of Fusion Models")
        ax.set_xlabel("Fusion Model")
        ax.set_ylabel("Test Accuracy")
        plt.xticks(rotation=45)

        return fig

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

        fig, ax = plt.subplots(figsize=(14, 10))
        sns.boxplot(
            x="embedding_combo", y="acc/test", hue="model_type", data=self.df, ax=ax
        )
        ax.set_title("Model Performance by Embedding Combination")
        ax.set_xlabel("Embedding Combination (Textual_Relational)")
        ax.set_ylabel("Test Accuracy")
        plt.xticks(rotation=45)
        ax.legend(title="Fusion Model")

        return fig

%matplotlib inline
analyzer = AdditionFusionAnalyzer()

analyzer.visualize()
# %matplotlib inline
# asgc = ASGC()
# fig = asgc.visualize()

# Split the series into two columns


# Alternative approach using str accessor:
# asgc.df[['first_part', 'second_part']] = asgc.df.prefix.str.split("/", expand=True)
