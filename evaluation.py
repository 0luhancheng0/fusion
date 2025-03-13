import torch
import numpy as np
from pathlib import Path
from typing import Union
from torch import Tensor
LOGISTIC_REGRESSION_PROVIDER = "cuml"
if torch.cuda.is_available():
    from cuml import LogisticRegression
else:
    from sklearn.linear_model import LogisticRegression
    LOGISTIC_REGRESSION_PROVIDER = "sklearn"
from torchmetrics.functional.classification import auroc
from dgl.nn.pytorch.link import EdgePredictor
import dgl
class NodeEmbeddingEvaluator:
    
    def __init__(self, labels, masks):
        self.labels = labels
        self.masks = masks

    def evaluate_arxiv_embeddings(self, node_embeddings: Tensor, split: str = "valid") -> float:
        return self.evaluate_node_classification(
            node_embeddings,
            self.labels,
            self.masks["train"], 
            self.masks[split]
        )
    @staticmethod
    def evaluate_link_prediction(
        graph,
        embeddings,  
        predictor=EdgePredictor("cos"),
        neg_ratio=1.,
        eval_fn=auroc  
    ):
        src, dst = graph.edges()
        pos_scores = predictor(embeddings[src], embeddings[dst]).squeeze()
        neg_src, neg_dst = dgl.sampling.global_uniform_negative_sampling(graph, int(graph.num_nodes() * neg_ratio))
        neg_scores = predictor(embeddings[neg_src], embeddings[neg_dst]).squeeze()
        labels = torch.hstack([torch.ones_like(pos_scores).squeeze(), torch.zeros_like(neg_scores).squeeze()]).type(torch.bool)
        scores = torch.hstack([pos_scores, neg_scores])
        return eval_fn(scores, labels, task="binary").item()
    @staticmethod
    def evaluate_node_classification(
        node_embeddings: Union[Tensor, np.ndarray],
        node_labels: Tensor,
        training_mask: Tensor,
        evaluation_mask: Tensor,
        *args,
        **kwargs
    ) -> float:
        """Evaluate node embeddings quality using logistic regression."""
        if isinstance(node_embeddings, torch.Tensor):
            node_embeddings = node_embeddings.detach().cpu().numpy()
        else:
            assert isinstance(node_embeddings, np.ndarray)
            
        node_labels = node_labels.detach().cpu().numpy()
        training_mask = training_mask.detach().cpu().numpy().astype(bool)
        evaluation_mask = evaluation_mask.detach().cpu().numpy().astype(bool)

        train_embeddings, train_labels = node_embeddings[training_mask], node_labels[training_mask]
        test_embeddings, test_labels = node_embeddings[evaluation_mask], node_labels[evaluation_mask]
        if LOGISTIC_REGRESSION_PROVIDER == "cuml":
            classifier = LogisticRegression(solver='qn', *args, **kwargs).fit(train_embeddings, train_labels)
        else:
            classifier = LogisticRegression(solver='lbfgs', *args, **kwargs).fit(train_embeddings, train_labels)
        return classifier.score(test_embeddings, test_labels)
