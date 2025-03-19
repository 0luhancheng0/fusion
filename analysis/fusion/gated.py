from .base import FusionAnalyzer
import numpy as np
import json
import pandas as pd
class GatedFusionAnalyzer(FusionAnalyzer):
    def __init__(self, dpi=300, cmap="viridis", figsize=(6.4, 4.8), 
                 remove_outliers=False, outlier_params=None):
        super().__init__("GatedFusion", dpi, cmap, figsize, 
                         remove_outliers=remove_outliers, outlier_params=outlier_params)
    def run(self):
        """Run all available visualizations for gated fusion analysis."""
        results = super().run()
        return results
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

