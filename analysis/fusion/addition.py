from .base import FusionAnalyzer
class AdditionFusionAnalyzer(FusionAnalyzer):
    def __init__(self, dpi=300, cmap="viridis", figsize=(6.4, 4.8), 
                 remove_outliers=False, outlier_params=None):
        super().__init__("AdditionFusion", dpi, cmap, figsize, 
                         remove_outliers=remove_outliers, outlier_params=outlier_params)

    def run(self):
        """Run all available visualizations for addition fusion analysis."""
        return super().run()