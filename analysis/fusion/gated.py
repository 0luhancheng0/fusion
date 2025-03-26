from .base import FusionAnalyzer
import numpy as np
import json
import pandas as pd
class GatedFusionAnalyzer(FusionAnalyzer):
    def __init__(self, dpi=300, cmap="viridis", figsize=(6.4, 4.8)):
        super().__init__("GatedFusion", dpi, cmap, figsize)
    def run(self):
        """Run all available visualizations for gated fusion analysis."""
        results = super().run()
        return results