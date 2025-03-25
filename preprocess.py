from constants import HARD_NEGATIVE_SAMPLES
from constants import UNIFORM_NEGATIVE_SAMPLES
from dataloading import OGBNArxivDataset
import torch
import typer
import multiprocessing
import dgl
from typing import Tuple, Optional, Union
from functools import partial
import math
from tqdm import tqdm
from pathlib import Path

app = typer.Typer(help="Generate hard negative samples for ogbn-arxiv dataset")

def _find_hard_negatives_for_node(node, graph):
    _, neighbors = graph.out_edges(node)
    if len(neighbors) == 0:
        return []

    # Get 2-hop neighbors
    _, two_hop_neighbors = graph.out_edges(neighbors)
    if len(two_hop_neighbors) == 0:
        return []

    # Remove duplicates
    two_hop_neighbors = torch.unique(two_hop_neighbors)

    # Remove direct connections and self-loops
    # direct_connections, _ = graph.out_edges(node)
    mask = ~torch.isin(two_hop_neighbors, neighbors)
    mask &= two_hop_neighbors != node
    valid_targets = two_hop_neighbors[mask]

    return valid_targets
    

def _find_hard_negatives_for_nodes(
    nodes: torch.Tensor,
    graph: dgl.DGLGraph,
    nsample_per_node: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find hard negatives for a batch of source nodes."""
    # Move to CPU for processing

    all_src = []
    all_dst = []

    # For each node, find its 2-hop neighbors
    for node in nodes:
        # Get direct neighbors

        hard_negatives_neighbors = _find_hard_negatives_for_node(node, graph)
        idx = torch.randperm(len(hard_negatives_neighbors))[:nsample_per_node]
        hard_negatives_neighbors = hard_negatives_neighbors[idx]


        # Add to results
        all_src.append(
            torch.full((len(hard_negatives_neighbors),), node.item(), dtype=torch.int64)
        )
        all_dst.append(hard_negatives_neighbors)

    src_tensor = torch.cat(all_src).to(graph.device)
    dst_tensor = torch.cat(all_dst).to(graph.device)
    return src_tensor, dst_tensor


    
def hard_negative_sampling(
    graph: dgl.DGLGraph,
    save_path: Optional[Union[str, Path]],
    neg_ratio: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_nodes = graph.num_nodes()
    # Adjust sampling fraction to ensure we get enough samples

    neg_edge_nums = int(float(neg_ratio) * graph.num_edges())

    # Generate random node indices without replacement
    src_nodes = torch.randperm(num_nodes).to(graph.device)

    hard_neg_src, hard_neg_dst = _find_hard_negatives_for_nodes(src_nodes, graph, neg_edge_nums // num_nodes + 1)
    

    idx = torch.randperm(len(hard_neg_src))[:neg_edge_nums]
    hard_neg_src = hard_neg_src[idx]
    hard_neg_dst = hard_neg_dst[idx]  # Fixed: use idx to sample both src and dst

    print(f"Generated {hard_neg_src.shape[0]} hard negative samples saved to {save_path}")
    torch.save((hard_neg_src, hard_neg_dst), save_path)
    return hard_neg_src, hard_neg_dst


def uniform_negative_sampling(
    graph: dgl.DGLGraph,
    save_path: Optional[Union[str, Path]],
    neg_ratio: float = 1.0,
    
) -> Tuple[torch.Tensor, torch.Tensor]:


    neg_src, neg_dst = dgl.sampling.global_uniform_negative_sampling(
        graph, int(float(neg_ratio) * graph.num_edges())
    )
    print(f"Generated {neg_src.shape[0]} uniform negative samples saved to {save_path}")
    torch.save((neg_src, neg_dst), save_path)



class Sampling:
    @app.command()
    def sample_hard_negatives(neg_ratio: float = 1.0):
        """
        Sample hard negatives from the ogbn-arxiv dataset.
        """
        # Load graph
        graph = OGBNArxivDataset().graph
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        graph = graph.to(device)
        HARD_NEGATIVE_SAMPLES.parent.mkdir(exist_ok=True, parents=True)

        return hard_negative_sampling(
            graph=graph,
            save_path=HARD_NEGATIVE_SAMPLES,
            neg_ratio=neg_ratio
        )
        
    @app.command()
    def sample_uniform_negatives(neg_ratio: float = 1.0):
        """
        Sample uniform negatives from the ogbn-arxiv dataset.
        """
        # Load graph
        graph = OGBNArxivDataset().graph
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        graph = graph.to(device)
        UNIFORM_NEGATIVE_SAMPLES.parent.mkdir(exist_ok=True, parents=True)

        return uniform_negative_sampling(
            graph=graph,
            save_path=UNIFORM_NEGATIVE_SAMPLES,
            neg_ratio=neg_ratio
        )


if __name__ == "__main__":
    app()