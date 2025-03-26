import torch
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional, List, Dict, Set
from torch import Tensor
import multiprocessing
from functools import partial
import math
import random
from tqdm import tqdm

LOGISTIC_REGRESSION_PROVIDER = "cuml"
if torch.cuda.is_available():
    from cuml import LogisticRegression
else:
    from sklearn.linear_model import LogisticRegression

    LOGISTIC_REGRESSION_PROVIDER = "sklearn"
from torchmetrics.functional.classification import auroc
from dgl.nn.pytorch.link import EdgePredictor
import dgl


# Move static methods out as module-level functions

def evaluate_node_classification(
    node_embeddings: Union[Tensor, np.ndarray],
    node_labels: Tensor,
    training_mask: Tensor,
    evaluation_mask: Tensor,
    *args,
    **kwargs,
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


def evaluate_link_prediction(
    graph, 
    embeddings, 
    negative_samples, 
    predictor=EdgePredictor("cos"), 
    eval_fn=auroc, 
    batch_size=100000, 
    check_invalid_values=True
):
    """Evaluate link prediction with batched processing."""
    # Create a temporary evaluator for backward compatibility
    evaluator = NodeEmbeddingEvaluator(graph=graph)
    
    # Call the instance method
    return evaluator._evaluate_link_prediction(
        graph=graph,
        embeddings=embeddings,
        negative_samples=negative_samples,
        predictor=predictor,
        eval_fn=eval_fn,
        batch_size=batch_size,
        check_invalid_values=check_invalid_values,
    )



class NodeEmbeddingEvaluator:
    """
    Evaluation utilities for node embeddings, including link prediction and node classification.
    This class handles generation, storage, and loading of negative samples for link prediction.
    """

    def __init__(
        self,
        labels=None,
        masks=None,
        graph=None,
        hard_negative_path=None,
        uniform_negative_path=None,
    ):
        # For node classification
        self.labels = labels
        self.masks = masks

        # For link prediction
        self.graph = graph

        self.hard_negatives = torch.load(hard_negative_path, map_location="cpu") if hard_negative_path else None
        self.uniform_negatives = torch.load(uniform_negative_path, map_location="cpu") if uniform_negative_path else None

    def evaluate_arxiv_embeddings(
        self, node_embeddings: Tensor, split: str = "valid"
    ) -> float:
        """Evaluate embeddings on node classification for ogbn-arxiv dataset."""
        if self.labels is None or self.masks is None:
            raise ValueError(
                "Labels and masks must be provided for node classification"
            )
            
        return evaluate_node_classification(
            node_embeddings, self.labels, self.masks["train"], self.masks[split]
        )
    

    
    def evaluate_link_prediction(
        self,
        embeddings,
        predictor=EdgePredictor("cos"),
        eval_fn=auroc,
        batch_size=100000,
        neg_ratio=1.0,
    ):
        # print(self.hard_negatives)
        # print(self.uniform_negatives)
        func = partial(NodeEmbeddingEvaluator._evaluate_link_prediction,
            graph=self.graph,
            embeddings=embeddings,
            predictor=predictor,
            eval_fn=eval_fn,
            batch_size=batch_size,
            neg_ratio=neg_ratio,
        )
        return {
            "lp_hard/auc": func(negative_samples=self.hard_negatives),
            "lp_uniform/auc": func(negative_samples=self.uniform_negatives),
        }
    @staticmethod
    def _evaluate_link_prediction(
        graph,
        embeddings,
        negative_samples,
        predictor=EdgePredictor("cos"),
        eval_fn=auroc,
        batch_size=100000,
        neg_ratio=1.0,
    ):
        """Internal method for link prediction evaluation."""
        if not torch.isfinite(embeddings).all():
            print(
                "Warning: Embeddings contain NaN, Inf, or -Inf values. Returning 0 for evaluation."
            )
            return 0.0

        src, dst = graph.edges()
        total_edges = src.shape[0]
        device = embeddings.device

        # Use provided negative samples
        neg_src, neg_dst = negative_samples
        # Ensure they're on the correct device
        neg_src = neg_src.to(device)
        neg_dst = neg_dst.to(device)

        # Apply neg_ratio if needed
        if neg_ratio < 1.0 and len(neg_src) > int(total_edges * neg_ratio):
            needed = int(total_edges * neg_ratio)
            perm = torch.randperm(len(neg_src), device=device)
            idx = perm[:needed]
            neg_src = neg_src[idx]
            neg_dst = neg_dst[idx]

        # Process positive edges in batches
        all_pos_scores = []
        for i in range(0, total_edges, batch_size):
            batch_src = src[i : i + batch_size]
            batch_dst = dst[i : i + batch_size]

            # Get scores for this batch
            with torch.no_grad():  # No need for gradients during evaluation
                batch_pos_scores = predictor(
                    embeddings[batch_src], embeddings[batch_dst]
                ).squeeze()
                all_pos_scores.append(batch_pos_scores)

        # Concatenate all positive scores
        pos_scores = torch.cat(all_pos_scores)

        # Process negative edges in batches
        all_neg_scores = []
        total_neg_edges = neg_src.shape[0]
        for i in range(0, total_neg_edges, batch_size):
            batch_neg_src = neg_src[i : i + batch_size]
            batch_neg_dst = neg_dst[i : i + batch_size]

            # Get scores for this batch
            with torch.no_grad():
                batch_neg_scores = predictor(
                    embeddings[batch_neg_src], embeddings[batch_neg_dst]
                ).squeeze()
                all_neg_scores.append(batch_neg_scores)

        # Concatenate all negative scores
        neg_scores = torch.cat(all_neg_scores)

        # Create labels and combine scores
        labels = torch.cat(
            [
                torch.ones(pos_scores.shape[0], device=pos_scores.device),
                torch.zeros(neg_scores.shape[0], device=neg_scores.device),
            ]
        ).type(torch.bool)

        scores = torch.cat([pos_scores, neg_scores])

        return eval_fn(scores, labels, task="binary").item()
