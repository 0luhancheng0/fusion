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
    def __init__(self, base_path, dpi=300, cmap="viridis", figsize=(6.4, 4.8)):
        self.base_path = base_path
        self.dpi = dpi
        self.figsize = figsize
        self.mask = None
        
        if isinstance(cmap, str):
            self.cmap = sns.color_palette(cmap, as_cmap=True)
        else:
            self.cmap = cmap
            
        self.config_paths = list(Path(self.base_path).glob("**/config.json"))
        self.embeddings_paths = [i.parent / "embeddings.pt" for i in self.config_paths]
        self.results_paths = [i.parent / "results.json" for i in self.config_paths]

        self._df = self._create_results_df()
        # self._df = self.post_process()
        # self._backup_df = self._df.copy()

    # @setattr
    # def df(self):
        # """Return the DataFrame with the mask applied."""
        # if self.mask is None:
        #     return self._df
        # else:
        #     return self._df[self.mask]
    
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
    def run(self):
        """Run the analysis. This method should be overridden by subclasses."""
        raise NotImplementedError("Subclasses should implement this method.")


        
