

from .base import FusionAnalyzer
class LowRankFusionAnalyzer(FusionAnalyzer):
    def __init__(self, dpi=300, cmap="viridis", figsize=(6.4, 4.8)):
        super().__init__("LowRankFusion", dpi, cmap, figsize)
    def post_process(self):
        self._df[["textual_name", "relational_name", "textual_dim", "relational_dim", "latent_dim", "rank"]] = self._df.prefix.str.split(
            "[/_]"
        ).tolist()
        self._df = self._df.drop(columns=["prefix"])
    def analyze(self):
        """Analyze with additional focus on rank parameter."""
        if self._df.empty:
            print("No data to analyze.")
            return None


        # Basic analysis from parent
        basic_analysis = super().analyze()

        # Additional analysis by rank
        rank_analysis = (
            self._df.groupby(["latent_dim", "rank"])
            .agg({"acc/test": ["mean", "std", "count"]})
            .reset_index()
        )

        # Format column names
        rank_analysis.columns = [
            "_".join(col).strip() if col[1] else col[0]
            for col in rank_analysis.columns.values
        ]

        return {"basic_analysis": basic_analysis, "rank_analysis": rank_analysis}

    def run(self):
        """Run all available visualizations for low rank fusion analysis."""
        results = super().run()
        

        return results