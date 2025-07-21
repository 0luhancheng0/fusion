from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import constants
import json
from abc import ABC, abstractmethod
from scipy.stats import wilcoxon, mannwhitneyu

class AbstractAnalyzer(ABC):
    def __init__(self, base_path, dpi=300, cmap="viridis", figsize=(6.4, 4.8)):
        self.base_path = base_path
        self.dpi = dpi
        self.figsize = figsize
        self.mask = None
        
        if isinstance(cmap, str):
            self.cmap = sns.color_palette(cmap, as_cmap=True)
        else:
            self.cmap = cmap
            
        self.config_paths = []
        self.results_paths = []
        self._df = self._create_results_df()

    @property
    def df(self):
        if self.mask is None:
            return self._df
        else:
            return self._df[self.mask]

    @property
    def metrics(self):
        return ['acc/test', "lp_hard/auc"]

    def _create_results_df(self):
        """
        Create a DataFrame from all results files, using a cached results.json if available.
        To force a rebuild, delete the 'results.json' file in the logs directory.
        """
        cache_path = Path(self.base_path) / "results.json"

        if cache_path.exists():
            print(f"Loading cached results from {cache_path}")
            try:
                return pd.read_json(cache_path)
            except Exception as e:
                print(f"Could not load cache file: {e}. Rebuilding...")

        print("No valid cache found. Building results from source files.")
        
        self.config_paths = list(Path(self.base_path).glob("**/config.json"))
        self.results_paths = [i.parent / "results.json" for i in self.config_paths]
        
        data = []
        for config_path, result_path in zip(self.config_paths, self.results_paths):
            if result_path.exists():
                try:
                    with open(config_path, "r") as f:
                        config = json.load(f)
                    with open(result_path, "r") as f:
                        results = json.load(f)
                    
                    entry = {**config, **results}
                    entry["path"] = str(result_path.parent)
                    data.append(entry)
                except Exception as e:
                    print(f"Warning: Could not process {result_path}: {e}. Skipping.")

        df = pd.DataFrame(data)
        
        try:
            df.to_json(cache_path, orient='records', indent=4)
            print(f"Results cached to {cache_path}")
        except Exception as e:
            print(f"Could not save cache file: {e}")
            
        return df

    def post_process(self):
        pass
    
    def save_figure(self, fig, filename):
        """Save a matplotlib figure."""
        path = constants.FIGURE_PATH / f"{filename}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def save_and_return(self, fig, base_name):
        analyzer_dir = self.__class__.__name__
        filepath = f"{analyzer_dir}/{base_name}"
        save_path = constants.FIGURE_PATH / analyzer_dir
        save_path.mkdir(parents=True, exist_ok=True)
        self.save_figure(fig, filepath)
        return fig

class CrossModelAnalyzer(AbstractAnalyzer):
    """Compare results across different fusion models."""
    METRIC_NAMES = {
        "acc/test": "Node Classification Accuracy",
        "lp_hard/auc": "Link Prediction AUROC (Hard)",
    }

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

        # Unify dimension column from 'dim' and 'latent_dim'
        if 'dim' in df.columns:
            if 'latent_dim' in df.columns:
                df['latent_dim'] = df['latent_dim'].fillna(df['dim'])
            else:
                df['latent_dim'] = df['dim']
        
        df["model_type"] = df["path"].str.extract(r"/logs/([^/]+)/")

        transformer_fusion = df[df.model_type == "TransformerFusion"]
        if not transformer_fusion.empty:
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
        
    def per_textual_model(self, textual_names=["nvembedv2", "word2vec"], latent_dim=32):
        if textual_names is None:
            textual_names = self._df['textual_name'].dropna().unique()

        fig, axes = plt.subplots(len(textual_names), 1, figsize=(18, 8 * len(textual_names)))
        if len(textual_names) == 1:
            axes = [axes]

        for ax, textual_name in zip(axes, textual_names):
            base_mask = (
                (self._df.textual_name == textual_name)
                | (self._df.prefix == textual_name)
                | (self._df.model_type.isin(["Node2VecLightning", "ASGC"]))
            )
            
            if latent_dim is not None:
                self.mask = base_mask | (self._df.latent_dim == latent_dim)
                title = f"Performance for {textual_name} (dim={latent_dim})"
            else:
                self.mask = base_mask
                title = f"Performance for {textual_name} (all dims)"
            
            textual_baselines = self._extract_textual_baselines(textual_name)
            
            self.visualize_performance_distributions_subplots(
                ax=ax,
                title=title,
                baselines={'textual': textual_baselines},
            )
        
        plt.tight_layout(rect=[0.03, 0.03, 1, 0.97])
        
        if latent_dim is not None:
            save_path = f"per_textual_model_{latent_dim}"
        else:
            save_path = "per_textual_model_all_dims"
        
        self.save_and_return(fig, f"{save_path}/{'_'.join(textual_names)}_stacked")
        self.mask = None
        return fig
                
    def visualize_performance_distributions_subplots(self, ax, title=None, baselines=None):
        """Visualize performance distribution using violin plots with baseline lines on a given axis."""
        df = self.df.copy()
        textual_models = ["TextualEmbeddings"]
        relational_models = ["ASGC", "Node2VecLightning"]
        
        # Correctly order models by category
        t_models = sorted([m for m in df["model_type"].unique() if m in textual_models])
        r_models = sorted([m for m in df["model_type"].unique() if m in relational_models])
        f_models = sorted([m for m in df["model_type"].unique() if m not in textual_models + relational_models])
        model_order = t_models + r_models + f_models

        if baselines and 'textual' in baselines:
            df = df[~df["model_type"].isin(textual_models)]
            model_order = [m for m in model_order if m not in textual_models]

        melted_df = pd.melt(df, 
                           id_vars=[col for col in df.columns if col not in self.METRIC_NAMES.keys()],
                           value_vars=list(self.METRIC_NAMES.keys()), 
                           var_name='metric', value_name='value')
        melted_df['metric'] = melted_df['metric'].map(self.METRIC_NAMES)

        if not melted_df.empty:
            sns.violinplot(x="model_type", y="value", hue="metric", data=melted_df,
                          palette="Set2", inner="box", ax=ax, order=model_order, 
                          dodge=True, legend=True)

        if baselines:
            colors = {'textual': ['red', 'darkred'], 'relational': 'blue'}
            linestyles = {'textual': '--', 'relational': '-.'}
            
            for baseline_type, baseline_values in baselines.items():
                for i, (metric, value) in enumerate(baseline_values.items()):
                    if not pd.isna(value):
                        color = colors.get(baseline_type, 'gray')
                        if baseline_type == 'textual' and isinstance(color, list):
                            color = color[i % len(color)]
                        
                        metric_label = self.METRIC_NAMES.get(metric, metric)
                        ax.axhline(y=value, color=color, linestyle=linestyles.get(baseline_type, '--'),
                                  alpha=0.8, linewidth=2, label=f'{baseline_type.title()} Baseline ({metric_label})')

        ax.set_ylabel("Performance Score", fontsize=self.fontsize_range[0])
        ax.set_xlabel("Model Type", fontsize=self.fontsize_range[0])
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=self.fontsize_range[0])

        relational_count = len([m for m in r_models if m in model_order])
        fusion_count = len([m for m in f_models if m in model_order])
        
        positions = [relational_count - 0.5] if relational_count > 0 else []
        labels = ["Relational Baselines", "Fusion Methods"]
        label_positions = [relational_count / 2 - 0.5, relational_count + fusion_count / 2 - 0.5]
        
        for pos in positions:
            if pos >= 0:
                ax.axvline(pos, color="gray", linestyle="--", alpha=0.7)
        
        y_top = ax.get_ylim()[1]
        label_y = y_top + (y_top - ax.get_ylim()[0]) * 0.02
        
        for i, (pos, label) in enumerate(zip(label_positions, labels)):
            if (i == 0 and relational_count > 0) or (i == 1 and fusion_count > 0):
                ax.text(pos, label_y, label, ha="center", va="bottom", 
                       fontweight="bold", fontsize=self.fontsize_range[0])

        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys(), title="Metric", 
                 fontsize=self.fontsize_range[0], title_fontsize=self.fontsize_range[0])

        ax.set_title(title, fontsize=self.fontsize_range[1])
        return ax

    def _extract_textual_baselines(self, textual_name):
        """Extract TextualEmbeddings baseline values for a specific textual model."""
        # Get TextualEmbeddings data for this specific textual model
        textual_data = self._df[
            (self._df.model_type == "TextualEmbeddings") & 
            ((self._df.textual_name == textual_name) | (self._df.prefix == textual_name))
        ]
        
        baselines = {}
        if not textual_data.empty:
            for metric_key, metric_name in self.METRIC_NAMES.items():
                if metric_key in textual_data.columns:
                    baseline_value = textual_data[metric_key].iloc[0]
                    baselines[metric_key] = baseline_value
        
        return baselines

    def generate_p_value_df(self, latent_dim=None, metrics=None, textual_names=None, baseline_model=None):
        """
        Generates a pandas DataFrame of p-values for each fusion model compared
        to a specified baseline model.
        """
        all_results = []

        relational_models = ["ASGC", "Node2VecLightning"]
        relational_model_names_lower = [m.replace("Lightning", "").lower() for m in relational_models]
        all_model_types = self._df['model_type'].dropna().unique()
        fusion_models = [m for m in all_model_types if m not in relational_models + ["TextualEmbeddings"]]

        if metrics:
            if isinstance(metrics, str):
                metrics = [metrics]
            metrics_to_process = {
                k: v for k, v in self.METRIC_NAMES.items()
                if v in metrics or k in metrics
            }
        else:
            metrics_to_process = self.METRIC_NAMES

        df_to_process = self._df
        if latent_dim is not None:
            df_to_process = df_to_process[df_to_process.latent_dim == latent_dim]

        if textual_names:
            df_to_process = df_to_process[df_to_process['textual_name'].isin(textual_names)]

        embedding_combos = df_to_process['embedding_combo'].dropna().unique()

        for combo in embedding_combos:
            textual_name, relational_name = combo.split('_')

            # Determine which baseline to use
            is_textual_baseline = baseline_model and baseline_model.lower() not in relational_model_names_lower
            is_relational_baseline = baseline_model and baseline_model.lower() in relational_model_names_lower

            if baseline_model and is_textual_baseline and textual_name.lower() != baseline_model.lower():
                continue
            
            if baseline_model and is_relational_baseline and relational_name.lower() != baseline_model.lower():
                continue

            # Get the single-value textual baseline scores
            textual_baseline_scores = self._extract_textual_baselines(textual_name)

            for fusion_model in fusion_models:
                fusion_model_df = df_to_process[
                    (df_to_process['model_type'] == fusion_model) &
                    (df_to_process['embedding_combo'] == combo)
                ]

                if fusion_model_df.empty:
                    continue

                # Compare against Textual Baseline
                if not baseline_model or is_textual_baseline:
                    for metric_key, metric_name in metrics_to_process.items():
                        result_row = {
                            'Model': f'{fusion_model} ({combo})',
                            'Baseline_Model': textual_name,
                            'Metric': metric_name,
                            'P Value': 'N/A',
                            'Improvement (%)': 'N/A'
                        }
                        
                        fusion_data = fusion_model_df[metric_key].dropna()
                        baseline_value = textual_baseline_scores.get(metric_key)

                        if baseline_value is not None and not pd.isna(baseline_value) and len(fusion_data) > 0:
                            if baseline_value != 0:
                                mean_fusion_score = fusion_data.mean()
                                improvement = (mean_fusion_score - baseline_value) / baseline_value * 100
                                result_row['Improvement (%)'] = f"{improvement:.2f}%"

                            if len(fusion_data) > 1:
                                diffs = fusion_data - baseline_value
                                if len(diffs[diffs != 0]) > 0:
                                    try:
                                        _, p_value = wilcoxon(diffs[diffs != 0], alternative='greater')
                                        result_row['P Value'] = f"{p_value:.4f}"
                                    except ValueError:
                                        pass
                        all_results.append(result_row)

                # Compare against Relational Baselines
                if not baseline_model or is_relational_baseline:
                    for relational_model in relational_models:
                        if not baseline_model or (is_relational_baseline and relational_model.lower().startswith(baseline_model.lower())):
                            for metric_key, metric_name in metrics_to_process.items():
                                result_row = {
                                    'Model': f'{fusion_model} ({combo})',
                                    'Baseline_Model': relational_model.replace("Lightning", "").lower(),
                                    'Metric': metric_name,
                                    'P Value': 'N/A',
                                    'Improvement (%)': 'N/A'
                                }

                                fusion_data = fusion_model_df[metric_key].dropna()
                                relational_data = df_to_process[df_to_process['model_type'] == relational_model][metric_key].dropna()

                                if len(fusion_data) > 0 and len(relational_data) > 0:
                                    mean_relational_score = relational_data.mean()
                                    if mean_relational_score != 0:
                                        mean_fusion_score = fusion_data.mean()
                                        improvement = (mean_fusion_score - mean_relational_score) / mean_relational_score * 100
                                        result_row['Improvement (%)'] = f"{improvement:.2f}%"
                                    
                                    if len(fusion_data) > 1 and len(relational_data) > 1:
                                        try:
                                            _, p_value = mannwhitneyu(fusion_data, relational_data, alternative='greater')
                                            result_row['P Value'] = f"{p_value:.4f}"
                                        except ValueError:
                                            pass
                                all_results.append(result_row)

        results_df = pd.DataFrame(all_results)
        if not results_df.empty:
            results_df['Improvement (p-value)'] = results_df.apply(
                lambda row: f"{row['Improvement (%)']} ({row['P Value']})", axis=1
            )
            
            index_cols = ['Model']
            if not baseline_model:
                index_cols.append('Baseline_Model')

            pivot_df = results_df.pivot_table(
                index=index_cols,
                columns='Metric',
                values='Improvement (p-value)',
                aggfunc='first'
            )
            return pivot_df

        return results_df


if __name__ == "__main__":
    analyzer = CrossModelAnalyzer()
    
    # Generate and print p-value tables
    p_value_df_32 = analyzer.generate_p_value_df(latent_dim=32)
    print("P-values for latent_dim=32")
    print(p_value_df_32)
    
    p_value_df_all = analyzer.generate_p_value_df(latent_dim=None)
    print("\nP-values for all latent_dims")
    print(p_value_df_all)

    # Example for a single metric
    p_value_df_32_acc = analyzer.generate_p_value_df(latent_dim=32, metrics=["Node Classification Accuracy"])
    print("\nP-values for latent_dim=32 (Accuracy only)")
    print(p_value_df_32_acc)

    # Example for a single baseline
    p_value_df_32_asgc = analyzer.generate_p_value_df(latent_dim=32, baseline_model='asgc')
    print("\nP-values for latent_dim=32 (against ASGC baseline)")
    print(p_value_df_32_asgc)