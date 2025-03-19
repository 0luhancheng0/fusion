from analysis.fusion.transformer import TransformerFusionAnalyzer
import numpy as np
import torch
from pathlib import Path
from dataloading import OGBNArxivDataset
from evaluation import NodeEmbeddingEvaluator
analyzer = TransformerFusionAnalyzer()
df = analyzer.df.sort_values(by=["lp_hard/auc"], ascending=False)
embeddings = torch.load(Path(df.iloc[0].path) / "embeddings.pt", map_location="cpu")

# embeddings.v
graph = OGBNArxivDataset.load_graph()
evaluator = OGBNArxivDataset.evaluator()

# evaluator.evaluate_link_prediction(graph, embeddings, use_hard_negatives=True)
# evaluator.evaluate_arxiv_embeddings(embeddings)

