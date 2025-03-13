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
    def __init__(self, base_path, dpi=300, cmap="viridis"):
        self.base_path = base_path
        self.dpi = dpi
        self.cmap = sns.color_palette(cmap, as_cmap=True)
        self.config_paths = list(Path(self.base_path).glob("**/config.json"))
        self.embeddings_paths = [i.parent / "embeddings.pt" for i in self.config_paths]
        self.results_paths = [i.parent / "results.json" for i in self.config_paths]
        
        self.df = self._create_results_df()
    
    def _create_results_df(self):
        """Create a DataFrame from all results files found in the base path."""
        data = []
        for config_path, result_path in zip(self.config_paths, self.results_paths):
            if result_path.exists():                # Load config
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Load results
                with open(result_path, 'r') as f:
                    results = json.load(f)
                
                # Combine config and results
                entry = {**config, **results}
                entry['path'] = str(result_path.parent)
                data.append(entry)
        
        return pd.DataFrame(data) if data else pd.DataFrame()
    
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
            
        with open(path, 'r') as f:
            return json.load(f)
    
    def load_results(self, path_or_index):
        """Load results from a file path or by index in the DataFrame."""
        if isinstance(path_or_index, int):
            path = self.results_paths[path_or_index]
        else:
            path = path_or_index
            
        with open(path, 'r') as f:
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
    def analyze(self):
        pass
    def visualize(self):
        ax = sns.heatmap(self.df.groupby(
            ["k", "reg"]
        ).agg(
            {"acc/test": "mean"}
        ).reset_index().pivot(
            index="k", columns="reg", values="acc/test"
        ), cmap=self.cmap, annot=True, fmt=".2f")
        return ax.get_figure()

asgc_analyzer = ASGC()

fig = asgc_analyzer.visualize()

fig.show()