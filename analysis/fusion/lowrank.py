

from .base import FusionAnalyzer
class LowRankFusionAnalyzer(FusionAnalyzer):
    def __init__(self, dpi=300, cmap="viridis", figsize=(6.4, 4.8), 
                 remove_outliers=False, outlier_params=None):
        super().__init__("LowRankFusion", dpi, cmap, figsize, 
                         remove_outliers=remove_outliers, outlier_params=outlier_params)
    def post_process(self):
        self.df[["textual_name", "relational_name", "textual_dim", "relational_dim", "latent_dim", "rank"]] = self.df.prefix.str.split(
            "[/_]"
        ).tolist()
        self.df = self.df.drop(columns=["prefix"])
    def analyze(self):
        """Analyze with additional focus on rank parameter."""
        if self.df.empty:
            print("No data to analyze.")
            return None


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

    def run(self):
        """Run all available visualizations for low rank fusion analysis."""
        results = super().run()
        

        return results