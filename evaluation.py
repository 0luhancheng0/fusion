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
    def _process_node_batch(node_batch, src_tensor, dst_tensor, max_samples_per_node, device="cpu"):
        """Process a batch of source nodes to limit samples per node.
        
        Args:
            node_batch: List of source nodes to process
            src_tensor: Full tensor of source nodes
            dst_tensor: Full tensor of destination nodes
            max_samples_per_node: Maximum samples per source node
            device: Device to process on
            
        Returns:
            Tuple of (source_nodes, destination_nodes)
        """
        batch_src_list = []
        batch_dst_list = []
        
        for node in node_batch:
            mask = src_tensor == node
            node_dsts = dst_tensor[mask]
            
            # If this source has too many destinations, sample them
            if len(node_dsts) > max_samples_per_node:
                idx = torch.randperm(len(node_dsts))[:max_samples_per_node]
                node_dsts = node_dsts[idx]
            
            # Add to final lists
            batch_src_list.append(torch.full((len(node_dsts),), node, device=device))
            batch_dst_list.append(node_dsts)
        
        if batch_src_list:
            return torch.cat(batch_src_list), torch.cat(batch_dst_list)
        else:
            return torch.tensor([], device=device), torch.tensor([], device=device)

    @staticmethod
    def _find_hard_negatives_for_nodes(
        nodes: torch.Tensor, 
        graph: dgl.DGLGraph, 
        max_samples_per_node: int = 10,
        device: str = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find hard negatives for a batch of source nodes.
        
        Args:
            nodes: Source nodes to find hard negatives for
            graph: Input graph
            max_samples_per_node: Maximum samples per source node
            device: Device to process on
            
        Returns:
            Tuple of (source_nodes, destination_nodes)
        """
        # Move to CPU for processing
        cpu_graph = graph.cpu() if device != "cpu" else graph
        
        all_src = []
        all_dst = []
        
        # For each node, find its 2-hop neighbors
        for node in nodes:
            # Get direct neighbors
            _, neighbors = cpu_graph.out_edges(node)
            if len(neighbors) == 0:
                continue
                
            # Get 2-hop neighbors
            _, two_hop_neighbors = cpu_graph.out_edges(neighbors)
            if len(two_hop_neighbors) == 0:
                continue
                
            # Remove duplicates
            two_hop_neighbors = torch.unique(two_hop_neighbors)
            
            # Remove direct connections and self-loops
            direct_connections, _ = cpu_graph.out_edges(node)
            mask = ~torch.isin(two_hop_neighbors, direct_connections)
            mask &= (two_hop_neighbors != node)
            valid_targets = two_hop_neighbors[mask]
            
            if len(valid_targets) == 0:
                continue
                
            # Limit samples per node
            if len(valid_targets) > max_samples_per_node:
                idx = torch.randperm(len(valid_targets))[:max_samples_per_node]
                valid_targets = valid_targets[idx]
            
            # Add to results
            all_src.append(torch.full((len(valid_targets),), node.item(), dtype=torch.int64))
            all_dst.append(valid_targets)
        
        # Combine results
        if all_src:
            src_tensor = torch.cat(all_src)
            dst_tensor = torch.cat(all_dst)
            return src_tensor, dst_tensor
        else:
            return torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)

    @staticmethod
    def _process_node_chunk(
        nodes: torch.Tensor,
        graph: dgl.DGLGraph,
        max_samples_per_node: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a chunk of nodes to find hard negatives."""
        return NodeEmbeddingEvaluator._find_hard_negatives_for_nodes(
            nodes=nodes,
            graph=graph,
            max_samples_per_node=max_samples_per_node
        )

    @staticmethod
    def optimized_hard_negative_sampling(
        graph: dgl.DGLGraph, 
        num_samples: int = None, 
        fallback_random: bool = True,
        max_samples_per_node: int = 10,
        save_path: Optional[Union[str, Path]] = None,
        load_path: Optional[Union[str, Path]] = None,
        force_regenerate: bool = False,
        num_workers: int = None,
        node_sampling_fraction: float = 0.2  # Sample a fraction of nodes to search for hard negatives
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate hard negative samples from a graph by sampling source nodes and finding 
        their 2-hop neighbors that aren't directly connected.
        
        Args:
            graph: Input graph
            num_samples: Number of samples to generate
            fallback_random: Whether to fall back to random sampling if not enough hard negatives
            max_samples_per_node: Maximum samples per source node
            save_path: Path to save samples
            load_path: Path to load samples
            force_regenerate: Force regeneration even if samples exist
            num_workers: Number of worker processes for multiprocessing (default: CPU count)
            node_sampling_fraction: Fraction of nodes to sample for hard negative generation
        """
        device = graph.device
        if num_samples is None:
            num_samples = graph.num_edges()
        
        # Set number of workers if not specified
        if num_workers is None:
            num_workers = max(1, multiprocessing.cpu_count() - 1)
            
        # Try to load from file if load_path is provided
        if load_path is not None and not force_regenerate:
            try:
                saved_data = torch.load(load_path)
                hard_neg_src = saved_data['src'].to(device)
                hard_neg_dst = saved_data['dst'].to(device)
                
                if len(hard_neg_src) >= num_samples:
                    if len(hard_neg_src) > num_samples:
                        perm = torch.randperm(len(hard_neg_src), device=device)
                        idx = perm[:num_samples]
                        hard_neg_src = hard_neg_src[idx]
                        hard_neg_dst = hard_neg_dst[idx]
                    return hard_neg_src, hard_neg_dst
                print(f"Loaded samples insufficient ({len(hard_neg_src)}), need {num_samples}. Regenerating...")
            except (FileNotFoundError, KeyError, RuntimeError) as e:
                print(f"Could not load negative samples from {load_path}: {e}. Generating new ones.")
        
        # Ensure graph is homogeneous
        g_homo = dgl.to_homogeneous(graph) if hasattr(graph, 'ntypes') else graph
        
        # Sample source nodes
        num_nodes = g_homo.num_nodes()
        # Adjust sampling fraction to ensure we get enough samples
        # We want to sample more nodes than needed to ensure we can generate enough hard negatives
        sample_size = min(num_nodes, max(int(num_nodes * node_sampling_fraction), 
                                     int(num_samples / max_samples_per_node * 3)))
        print(f"Sampling {sample_size} source nodes out of {num_nodes} total nodes")
        
        # Generate random node indices without replacement
        sampled_nodes = torch.randperm(num_nodes)[:sample_size]
        
        # Split nodes into chunks for parallel processing
        chunk_size = math.ceil(len(sampled_nodes) / num_workers)
        node_chunks = [
            sampled_nodes[i:i+chunk_size] 
            for i in range(0, len(sampled_nodes), chunk_size)
        ]
        
        print(f"Processing {len(node_chunks)} chunks with {num_workers} workers")
        
        # Move graph to CPU for multiprocessing
        cpu_graph = g_homo.cpu()
        
        # Process chunks in parallel
        hard_neg_src_list = []
        hard_neg_dst_list = []
        collected_samples = 0
        
        with multiprocessing.Pool(processes=num_workers) as pool:
            # Create a partial function with fixed parameters
            process_func = partial(
                NodeEmbeddingEvaluator._process_node_chunk,
                graph=cpu_graph,
                max_samples_per_node=max_samples_per_node
            )
            
            # Process chunks and collect results incrementally
            for src_chunk, dst_chunk in tqdm(pool.imap(process_func, node_chunks), 
                                            total=len(node_chunks), 
                                            desc="Finding hard negatives"):
                if len(src_chunk) > 0:
                    hard_neg_src_list.append(src_chunk)
                    hard_neg_dst_list.append(dst_chunk)
                    
                    collected_samples += len(src_chunk)
                    if collected_samples >= num_samples:
                        break
        
        # Combine results
        if hard_neg_src_list:
            hard_neg_src = torch.cat(hard_neg_src_list).to(device)
            hard_neg_dst = torch.cat(hard_neg_dst_list).to(device)
            print(f"Collected {len(hard_neg_src)} hard negative samples")
        else:
            hard_neg_src = torch.tensor([], device=device, dtype=torch.int64)
            hard_neg_dst = torch.tensor([], device=device, dtype=torch.int64)
            print("No hard negative samples found")
        
        # If we need more samples and fallback is enabled, add random samples
        if (len(hard_neg_src) < num_samples) and fallback_random:
            needed = num_samples - len(hard_neg_src)
            print(f"Need {needed} more samples, using random sampling")
            rand_src, rand_dst = dgl.sampling.global_uniform_negative_sampling(graph, needed)
            
            # Only concatenate if we have some hard negatives
            if len(hard_neg_src) > 0:
                hard_neg_src = torch.cat([hard_neg_src, rand_src])
                hard_neg_dst = torch.cat([hard_neg_dst, rand_dst])
            else:
                hard_neg_src, hard_neg_dst = rand_src, rand_dst
        
        # Trim if we have too many
        if len(hard_neg_src) > num_samples:
            perm = torch.randperm(len(hard_neg_src), device=device)
            idx = perm[:num_samples]
            hard_neg_src = hard_neg_src[idx]
            hard_neg_dst = hard_neg_dst[idx]
        
        # Save the generated samples if save_path is provided
        if save_path is not None:
            # Create directory if it doesn't exist
            save_dir = Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as dictionary with CPU tensors to ensure portability
            save_dict = {
                'src': hard_neg_src.cpu(),
                'dst': hard_neg_dst.cpu(),
                'num_nodes': graph.num_nodes(),
                'num_edges': graph.num_edges()
            }
            torch.save(save_dict, save_path)
            print(f"Saved {len(hard_neg_src)} hard negative samples to {save_path}")
            
        return hard_neg_src, hard_neg_dst

    @staticmethod
    def evaluate_link_prediction(
        graph, embeddings, predictor=EdgePredictor("cos"), neg_ratio=1.0, eval_fn=auroc, batch_size=100000,
        use_hard_negatives=False
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
            use_hard_negatives: Whether to use hard negative sampling instead of random
            
        Returns:
            Link prediction score
        """
        src, dst = graph.edges()
        total_edges = src.shape[0]
        
        # Sample negative edges
        if use_hard_negatives:
            neg_src, neg_dst = NodeEmbeddingEvaluator.optimized_hard_negative_sampling(
                graph, int(total_edges * neg_ratio)
            )
        else:
            neg_src, neg_dst = dgl.sampling.global_uniform_negative_sampling(
                graph, int(total_edges * neg_ratio)
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

