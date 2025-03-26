from .base import FusionAnalyzer

class EarlyFusionAnalyzer(FusionAnalyzer):
    def __init__(self, dpi=300, cmap="viridis", figsize=(6.4, 4.8)):
        super().__init__("EarlyFusion", dpi, cmap, figsize)

    def run(self):
        """Run all available visualizations for early fusion analysis."""
        return super().run()

