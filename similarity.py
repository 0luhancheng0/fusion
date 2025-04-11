from cuvs.distance import pairwise_distance
import torch
import cupy as cp
from dataloading import OGBNArxivDataset
import seaborn as sns
from itertools import product
import matplotlib.pyplot as plt
from constants import FIGURE_PATH
import typer
from typing import List, Tuple, Optional
from pathlib import Path
from rich import print

app = typer.Typer(help="Compute similarity between embeddings and visualize results")


def load_embeddings(paths, device="cuda:0"):
    """
    Load embeddings from specified paths.
    
    Args:
        paths (list): List of paths to embedding files
        device (str): Device to load tensors to
    
    Returns:
        list: List of loaded embeddings as CuPy arrays
    """
    embeddings = []
    for path in paths:
        embedding = cp.asarray(torch.load(path, map_location=device))
        embeddings.append(embedding)
    return embeddings


def compute_similarity_matrix(embeddings, labels, num_classes):
    """
    Compute similarity matrix between classes across different embeddings.
    
    Args:
        embeddings (list): List of embedding arrays
        labels (array): Class labels
        num_classes (int): Number of unique classes
    
    Returns:
        numpy.ndarray: Similarity matrix
    """
    results = cp.empty((num_classes, num_classes))
    labels_cp = cp.asarray(labels)
    
    for i, j in product(range(num_classes), range(num_classes)):
        mask1 = labels_cp == i
        mask2 = labels_cp == j
        
        dist1 = pairwise_distance(embeddings[0][mask1], embeddings[0][mask2], metric="cosine")
        dist2 = pairwise_distance(embeddings[1][mask1], embeddings[1][mask2], metric="cosine")
        corr = pairwise_distance(dist1, dist2, metric="correlation")
        results[i, j] = 1 - corr.copy_to_host().mean()
    
    return results.get()


def plot_and_save_heatmap(matrix, save_path, figsize=(12, 10)):
    """
    Plot and save a heatmap visualization of the similarity matrix.
    
    Args:
        matrix (numpy.ndarray): The similarity matrix to visualize
        save_path (str): Path to save the figure
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="coolwarm", cbar=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


@app.command()
def compute_similarity(
    embedding_path1: str = typer.Argument(
        "/fred/oz318/luhanc/fusion/saved_embeddings/ogbn-arxiv/textual/mpnet/768.pt",
        help="Path to first embedding file"
    ),
    embedding_path2: str = typer.Argument(
        "/fred/oz318/luhanc/fusion/saved_embeddings/ogbn-arxiv/textual/roberta/768.pt",
        help="Path to second embedding file"
    ),
    output_name: Path = typer.Option(
        "rank_correlation", help="Path to save the figure"
    ),
    figsize: Tuple[int, int] = typer.Option(
        (12, 10), help="Figure size (width, height)"
    ),
    device: str = typer.Option(
        "cuda:0", help="Device to load tensors to"
    )
):
    """
    Compute and visualize the similarity matrix between embeddings.
    
    This script loads embeddings from specified paths, computes the similarity between 
    class clusters across different embeddings, and visualizes the result as a heatmap.
    """
    # Combine paths into a list
    
    embedding_paths = [embedding_path1, embedding_path2]
    
    print(f"Loading embeddings from {embedding_paths}")
    embeddings = load_embeddings(embedding_paths, device)
    
    print("Loading dataset")
    arxiv = OGBNArxivDataset()
    num_classes = arxiv.labels.unique().shape[0]
    
    print("Computing similarity matrix")
    similarity_matrix = compute_similarity_matrix(embeddings, arxiv.labels, num_classes)
    
    output_path = (FIGURE_PATH / output_name).with_suffix(".png")
    print(f"Plotting and saving heatmap to {output_path}")
    plot_and_save_heatmap(similarity_matrix, output_path, figsize)
    print("Done!")


if __name__ == "__main__":
    try:
        app()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


