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

    def evaluate_arxiv_embeddings(
        self, node_embeddings: Tensor, split: str = "valid"
    ) -> float:
        return self.evaluate_node_classification(
            node_embeddings, self.labels, self.masks["train"], self.masks[split]
        )

    @staticmethod
    def evaluate_link_prediction(
        graph, embeddings, predictor=EdgePredictor("cos"), neg_ratio=1.0, eval_fn=auroc, batch_size=100000
    ):
        """
        Evaluate link prediction with batched processing to avoid OOM errors.
        
        Args:
            graph: DGL graph
            embeddings: Node embeddings
            predictor: Edge predictor function
            neg_ratio: Ratio of negative to positive samples
            eval_fn: Evaluation metric function
            batch_size: Batch size for processing edges to avoid OOM
            
        Returns:
            Link prediction score
        """
        src, dst = graph.edges()
        total_edges = src.shape[0]
        
        # Sample negative edges all at once (this is memory-efficient as we're just getting indices)
        neg_src, neg_dst = dgl.sampling.global_uniform_negative_sampling(
            graph, int(graph.num_nodes() * neg_ratio)
        )

        # Process positive edges in batches
        all_pos_scores = []
        for i in range(0, total_edges, batch_size):
            batch_src = src[i:i+batch_size]
            batch_dst = dst[i:i+batch_size]
            
            # Get scores for this batch
            with torch.no_grad():  # No need for gradients during evaluation
                batch_pos_scores = predictor(embeddings[batch_src], embeddings[batch_dst]).squeeze()
                all_pos_scores.append(batch_pos_scores)
        
        # Concatenate all positive scores
        pos_scores = torch.cat(all_pos_scores)
        
        # Process negative edges in batches
        all_neg_scores = []
        total_neg_edges = neg_src.shape[0]
        for i in range(0, total_neg_edges, batch_size):
            batch_neg_src = neg_src[i:i+batch_size]
            batch_neg_dst = neg_dst[i:i+batch_size]
            
            # Get scores for this batch
            with torch.no_grad():
                batch_neg_scores = predictor(embeddings[batch_neg_src], embeddings[batch_neg_dst]).squeeze()
                all_neg_scores.append(batch_neg_scores)
        
        # Concatenate all negative scores
        neg_scores = torch.cat(all_neg_scores)
        
        # Create labels and combine scores
        labels = torch.cat([
            torch.ones(pos_scores.shape[0], device=pos_scores.device),
            torch.zeros(neg_scores.shape[0], device=neg_scores.device)
        ]).type(torch.bool)
        
        scores = torch.cat([pos_scores, neg_scores])
        
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

        train_embeddings, train_labels = (
            node_embeddings[training_mask],
            node_labels[training_mask],
        )
        test_embeddings, test_labels = (
            node_embeddings[evaluation_mask],
            node_labels[evaluation_mask],
        )
        if LOGISTIC_REGRESSION_PROVIDER == "cuml":
            classifier = LogisticRegression(solver="qn", *args, **kwargs).fit(
                train_embeddings, train_labels
            )
        else:
            classifier = LogisticRegression(solver="lbfgs", *args, **kwargs).fit(
                train_embeddings, train_labels
            )
        return classifier.score(test_embeddings, test_labels)
