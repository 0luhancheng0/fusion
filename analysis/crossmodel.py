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
        mask=None,
        fontsize_range=(8, 12),
    ):
        self.mask = mask
        self.remove_transformer_fusion = remove_transformer_fusion
        self.fontsize_range = fontsize_range  # Store the font size range
        super().__init__("/home/lcheng/oz318/fusion/logs", dpi, cmap, figsize)
        self._df = self.post_process()
        self._backup_df = self._df.copy()

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

        if self.remove_transformer_fusion:
            df = df[~df.model_type.str.startswith("TransformerFusion")]

        return df
        
    def per_textual_model_32(self, textual_names=None):
        results = {}
        unique_textual_names = [
            i.name
            for i in Path("/home/lcheng/oz318/fusion/logs/TextualEmbeddings").iterdir()
            if i.is_dir()
        ]

        latent_dim = 32
        for textual_name in unique_textual_names:
            self.mask = (
                (self._df.textual_name == textual_name)
                | (self._df.prefix == textual_name)
                | (self._df.model_type.isin(["Node2VecLightning", "ASGC"])) 
                | (self._df.latent_dim == latent_dim))
            
            # Extract TextualEmbeddings baseline values for this textual model
            textual_baselines = self._extract_textual_baselines(textual_name)
            
            results[f"per_textual_model_32/{textual_name}"] = self.save_and_return(
                self.visualize_performance_distributions_subplots(baselines={'textual': textual_baselines}), 
                f"per_textual_model_32/{textual_name}"
            )
        self.mask = None
        return results
                
    def per_embedding_combo(self):
        unique_embedding_combos = self._df.embedding_combo.dropna().unique()
        results = {}
        for embedding_combo in unique_embedding_combos:
            self.mask = (
                (self._df.embedding_combo == embedding_combo)
                | self.baseline_mask
            )
        self.mask = None
        return results
            
    @property
    def baseline_mask(self):
        return (self._df.model_type.isin(["Node2VecLightning", "ASGC", "TextualEmbeddings"]))


    def visualize_by_embedding_type(self):
        """Compare model performance across different embedding combinations."""

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
                f"Model {metric.replace('/', ' ').title()} by Embedding Combination",
                fontsize=self.fontsize_range[1]  # Use max font size for title
            )
            ax.set_xlabel("Embedding Combination (Textual_Relational)", fontsize=self.fontsize_range[0]) # Use min font size
            ax.set_ylabel(metric.replace("/", " ").title(), fontsize=self.fontsize_range[0]) # Use min font size
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=self.fontsize_range[0]) # Use min font size
            ax.legend(title="Fusion Model", fontsize=self.fontsize_range[0], title_fontsize=self.fontsize_range[0]) # Use min font size

        plt.tight_layout()
        return self.save_and_return(fig, "embedding_type_comparison")



    def visualize_performance_distributions_subplots(self, title=None, baselines=None):
        """Visualize performance distribution using violin plots with baseline lines."""
        metric_names = {
            "acc/test": "Node Classification Accuracy",
            "lp_hard/auc": "Link Prediction AUROC (Hard)",
        }
        
        # Setup data and model categorization
        df = self.df.copy()
        textual_models = ["TextualEmbeddings"]
        relational_models = ["ASGC", "Node2VecLightning"]
        fusion_models = [m for m in df["model_type"].unique() 
                        if m not in textual_models + relational_models]
        model_order = sorted(textual_models + relational_models + fusion_models)

        # Determine if textual baselines should be shown as lines
        show_textual_as_lines = (baselines and 'textual' in baselines and 
                                not df[df["model_type"].isin(textual_models)].empty and
                                all(df[df["model_type"].isin(textual_models)][metric].notna().sum() == 1 
                                    for metric in metric_names.keys()) and len(metric_names) >= 2)

        # Filter out textual models if baselines provided
        if baselines and 'textual' in baselines:
            df = df[~df["model_type"].isin(textual_models)]
            model_order = [m for m in model_order if m not in textual_models]

        # Prepare data for plotting
        melted_df = pd.melt(df, 
                           id_vars=[col for col in df.columns if col not in metric_names.keys()],
                           value_vars=list(metric_names.keys()), 
                           var_name='metric', value_name='value')
        melted_df['metric'] = melted_df['metric'].map(metric_names)

        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(18, 8))

        if not melted_df.empty:
            # Main violin plot (without individual data points)
            sns.violinplot(x="model_type", y="value", hue="metric", data=melted_df,
                          palette="Set2", inner="box", ax=ax, order=model_order, 
                          dodge=True, legend=True)

        # Add baseline lines
        if baselines:
            colors = {'textual': ['red', 'darkred'], 'relational': 'blue'}
            linestyles = {'textual': '--', 'relational': '-.'}
            
            for baseline_type, baseline_values in baselines.items():
                for i, (metric, value) in enumerate(baseline_values.items()):
                    if not pd.isna(value):
                        if baseline_type == 'textual' and isinstance(colors[baseline_type], list):
                            color = colors[baseline_type][i % len(colors[baseline_type])]
                        else:
                            color = colors.get(baseline_type, 'gray')
                        
                        # Map metric key to metric name for label
                        metric_label = metric_names.get(metric, metric)
                        
                        ax.axhline(y=value, color=color, 
                                  linestyle=linestyles.get(baseline_type, '--'),
                                  alpha=0.8, linewidth=2,
                                  label=f'{baseline_type.title()} Baseline ({metric_label})')

        # Styling and annotations
        ax.set_ylabel("Performance Score", fontsize=self.fontsize_range[0])
        ax.set_xlabel("Model Type", fontsize=self.fontsize_range[0])
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=self.fontsize_range[0])

        # Add category separators and labels (updated for when TextualEmbeddings are baseline lines)
        if baselines and 'textual' in baselines:
            # When textual baselines are lines, only show relational and fusion categories
            relational_count = len([m for m in relational_models if m in model_order])
            fusion_count = len([m for m in fusion_models if m in model_order])
            
            positions = []
            if relational_count > 0:
                positions.append(relational_count - 0.5)
            
            labels = ["Relational Baselines", "Fusion Methods"]
            label_positions = [
                relational_count / 2 - 0.5,
                relational_count + fusion_count / 2 - 0.5
            ]
        else:
            # Original category layout
            counts = [len([m for m in models if m in model_order]) 
                     for models in [textual_models, relational_models]]
            positions = [counts[0] - 0.5, sum(counts[:2]) - 0.5]
            labels = ["Textual Baselines", "Relational Baselines", "Fusion Methods"]
            label_positions = [counts[0]/2 - 0.5, counts[0] + counts[1]/2 - 0.5, 
                              sum(counts) + len(fusion_models)/2 - 0.5]
        
        for pos in positions:
            if pos >= 0:
                ax.axvline(pos, color="gray", linestyle="--", alpha=0.7)
        
        y_top = ax.get_ylim()[1]
        label_y = y_top + (y_top - ax.get_ylim()[0]) * 0.02
        
        for i, (pos, label) in enumerate(zip(label_positions, labels)):
            if baselines and 'textual' in baselines:
                # For baseline mode: relational (i==0), fusion (i==1)
                if (i == 0 and relational_count > 0) or (i == 1 and fusion_count > 0):
                    ax.text(pos, label_y, label, ha="center", va="bottom", 
                           fontweight="bold", fontsize=self.fontsize_range[0])
            else:
                # Original mode: textual (i==0), relational (i==1), fusion (i==2)
                if (i == 0 and counts[0] > 0) or (i == 1 and counts[1] > 0) or (i == 2 and fusion_models):
                    ax.text(pos, label_y, label, ha="center", va="bottom", 
                           fontweight="bold", fontsize=self.fontsize_range[0])

        # Setup legend
        handles, labels = ax.get_legend_handles_labels()
        legend_handles = handles[:len(melted_df['metric'].unique())] if not melted_df.empty else []
        legend_labels = labels[:len(legend_handles)]
        
        if baselines:
            baseline_handles = [h for h in handles if hasattr(h, 'get_linestyle')]
            baseline_labels = [l for h, l in zip(handles, labels) if hasattr(h, 'get_linestyle')]
            legend_handles.extend(baseline_handles)
            legend_labels.extend(baseline_labels)
        
        if legend_handles:
            ax.legend(legend_handles, legend_labels, title="Metric", 
                     fontsize=self.fontsize_range[0], title_fontsize=self.fontsize_range[0])

        fig.suptitle("Performance Distribution by Model Type" if title is None else title,
                    fontsize=self.fontsize_range[1])
        plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
        
        return fig

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
    
    def _extract_textual_baselines(self, textual_name):
        """Extract TextualEmbeddings baseline values for a specific textual model."""
        metric_names = {
            "acc/test": "Node Classification Accuracy",
            "lp_hard/auc": "Link Prediction AUROC (Hard)",
        }
        
        # Get TextualEmbeddings data for this specific textual model
        textual_data = self._df[
            (self._df.model_type == "TextualEmbeddings") & 
            ((self._df.textual_name == textual_name) | (self._df.prefix == textual_name))
        ]
        
        baselines = {}
        if not textual_data.empty:
            for metric_key, metric_name in metric_names.items():
                if metric_key in textual_data.columns:
                    baseline_value = textual_data[metric_key].iloc[0]
                    baselines[metric_key] = baseline_value
        
        return baselines

    def run(self):
        """Run all available visualizations for cross-model analysis."""

        results = {}
        results["embedding_type"] = self.visualize_by_embedding_type()

        # Create subplots for all performance distributions in a single figure
        results["performance_distributions"] = self.visualize_performance_distributions_subplots()

        results.update(self.per_textual_model_32())
        
        return results