%reload_ext autoreload
%autoreload 2
%matplotlib inline
from analysis.crossmodel import CrossModelAnalyzer
import torch
from dataloading import OGBNArxivDataset
import seaborn as sns
import torch.nn.functional as F
import matplotlib.pyplot as plt
import constants
from pathlib import Path
from analysis.textual import TextualEmbeddingsAnalyzer
import numpy as np
import pandas as pd
from IPython.display import display
from scipy.stats import spearmanr
import faiss
import time
from faiss_utils import build_faiss_index, faiss_search_topk, build_and_save_faiss_index, load_faiss_index, get_index_path_from_embedding_path, compute_similarity_with_saved_faiss, compute_similarity_with_faiss

def compute_similarity(embeddings, mask, normalize=True, batch_size=1000, use_faiss=False, k=None, 
                     embedding_path=None, index_root="/home/lcheng/oz318/fusion/faiss_indices", 
                     force_rebuild=False):
    """
    Compute cosine similarity between embeddings.
    
    Args:
        embeddings: Node embeddings
        mask: Boolean mask indicating which nodes to compute similarity for
        normalize: Whether to normalize embeddings before computing similarity
        batch_size: Number of masked nodes to process in each batch to save memory
        use_faiss: Whether to use FAISS for faster similarity search
        k: Number of top similar items to return (only used with FAISS)
        embedding_path: Path to the embedding file (used to determine index location)
        index_root: Root directory for storing FAISS indices
        force_rebuild: Whether to rebuild the index even if it already exists
        
    Returns:
        If use_faiss=True: Tuple of (similarities, indices) tensors for top-k similar items
        If use_faiss=False: Tensor of shape [num_masked_nodes, num_total_nodes] containing cosine similarities
    """
    if use_faiss:
        if embedding_path:
            return compute_similarity_with_saved_faiss(
                embeddings, mask, embedding_path=embedding_path, k=k, 
                normalize=normalize, index_root=index_root, force_rebuild=force_rebuild
            )
        else:
            return compute_similarity_with_faiss(embeddings, mask, k=k, normalize=normalize)
    
    # Original batch computation without FAISS
    if normalize:
        embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Get the indices of nodes where mask is True
    masked_indices = torch.where(mask)[0]
    num_masked = masked_indices.size(0)

    # Create tensor to store similarities
    device = embeddings.device
    cosine_similarity = torch.zeros((num_masked, embeddings.size(0)), device=device)
    
    # Process in batches to avoid memory issues
    for i in range(0, num_masked, batch_size):
        end_idx = min(i + batch_size, num_masked)
        batch_indices = masked_indices[i:end_idx]
        
        # Compute similarity between this batch of masked embeddings and all embeddings
        batch_similarity = embeddings[batch_indices] @ embeddings.T
        
        # Store in the result tensor
        cosine_similarity[i:end_idx] = batch_similarity
    
    # Zero out self-similarities
    for i, idx in enumerate(masked_indices):
        cosine_similarity[i, idx] = 0
    
    return cosine_similarity

def compare_embeddings(dataset, embeddings1, embeddings2=None, mask=None, k=5, model_names=None, 
                      num_display=5, use_faiss=True, embedding_paths=None, force_rebuild=False):
    """
    Compare top-k similar papers using one or two different embedding models.
    
    Args:
        dataset: Dataset containing paper information
        embeddings1: First embedding model
        embeddings2: Second embedding model (optional)
        mask: Mask indicating which papers to analyze
        k: Number of similar papers to find
        model_names: Names of the embedding models
        num_display: Number of source papers to display
        use_faiss: Whether to use FAISS for faster similarity search
        embedding_paths: List of paths to embedding files (used for index management)
        force_rebuild: Whether to rebuild FAISS indices even if they exist
        
    Returns:
        Matplotlib figure object
    """
    # Check if comparing one or two models
    is_single_model = embeddings2 is None
    
    # Set default model names if not provided
    if model_names is None:
        model_names = ["Model 1"] if is_single_model else ["Model 1", "Model 2"]
    elif isinstance(model_names, str):
        model_names = [model_names]
    
    # Set up embedding paths
    if embedding_paths is None:
        embedding_paths = [None, None]
    elif isinstance(embedding_paths, str) or isinstance(embedding_paths, Path):
        embedding_paths = [embedding_paths, None] if is_single_model else [embedding_paths, embedding_paths]
    elif len(embedding_paths) < 2 and not is_single_model:
        embedding_paths.append(None)
    
    # Require a mask to avoid memory issues
    if mask is None:
        raise ValueError("A mask must be provided to avoid memory issues. Use a targeted subset of nodes.")
    
    # Compute similarity and get top-k similar papers for first model
    if use_faiss:
        print(f"Using FAISS for {model_names[0]} similarity search")
        sim_values1, sim_indices1 = compute_similarity(
            embeddings1, mask, use_faiss=True, k=k, 
            embedding_path=embedding_paths[0], force_rebuild=force_rebuild
        )
        top_results_model1 = get_topk_similar(sim_values1, k=k, indices=sim_indices1)
    else:
        similarity_model1 = compute_similarity(embeddings1, mask)
        top_results_model1 = get_topk_similar(similarity_model1, k=k)
    
    # If comparing two models, compute similarity for second model
    if not is_single_model:
        if use_faiss:
            print(f"Using FAISS for {model_names[1]} similarity search")
            sim_values2, sim_indices2 = compute_similarity(
                embeddings2, mask, use_faiss=True, k=k, 
                embedding_path=embedding_paths[1], force_rebuild=force_rebuild
            )
            top_results_model2 = get_topk_similar(sim_values2, k=k, indices=sim_indices2)
        else:
            similarity_model2 = compute_similarity(embeddings2, mask)
            top_results_model2 = get_topk_similar(similarity_model2, k=k)
    
    # Get indices where mask is True
    source_indices = torch.where(mask)[0]
    num_sources = len(source_indices)
    
    # Get paper mappings for top-k results from both models
    papers1 = get_paper_mappings(dataset, top_results_model1.indices)
    titles1 = get_paper_titles(dataset, papers1)
    
    if not is_single_model:
        papers2 = get_paper_mappings(dataset, top_results_model2.indices)
        titles2 = get_paper_titles(dataset, papers2)
    
    # Get source paper IDs and titles
    source_paper_ids = dataset.node_to_paper_mapping.loc[source_indices.cpu()].values.flatten()
    source_titles = dataset.paper_metadata.loc[source_paper_ids].title.values
    
    # Visualize results
    if is_single_model:
        return visualize_single_embedding(titles1, source_titles, top_results_model1.values, k, model_names[0])
    else:
        return visualize_embedding_comparison(
            titles1, titles2, source_titles, top_results_model1.values, top_results_model2.values, 
            k, model_names
        )

# Refactored visualization functions using matplotlib and seaborn

def visualize_largest_rank_shifts(dataset, embeddings1, embeddings2, mask, k=20, num_display=10, 
                                  model_names=None, use_faiss=True, embedding_paths=None, force_rebuild=False):
    """
    Visualize the papers with the largest rank shifts between two embedding models using matplotlib.
    """
    # Compute similarities, topk results, paper mappings, and top_shifts
    if model_names is None:
        model_names = ["Model 1", "Model 2"]
    
    if embedding_paths is None:
        embedding_paths = [None, None]
    elif isinstance(embedding_paths, str) or isinstance(embedding_paths, Path):
        embedding_paths = [embedding_paths, embedding_paths]
    elif len(embedding_paths) < 2:
        embedding_paths.append(None)
    
    if use_faiss:
        sim_values1, sim_indices1 = compute_similarity(
            embeddings1, mask, use_faiss=True, k=k, 
            embedding_path=embedding_paths[0], force_rebuild=force_rebuild
        )
        topk1 = get_topk_similar(sim_values1, k=k, indices=sim_indices1)
        
        sim_values2, sim_indices2 = compute_similarity(
            embeddings2, mask, use_faiss=True, k=k, 
            embedding_path=embedding_paths[1], force_rebuild=force_rebuild
        )
        topk2 = get_topk_similar(sim_values2, k=k, indices=sim_indices2)
    else:
        sim1 = compute_similarity(embeddings1, mask)
        topk1 = get_topk_similar(sim1, k=k)
        
        sim2 = compute_similarity(embeddings2, mask)
        topk2 = get_topk_similar(sim2, k=k)
    
    source_indices = torch.where(mask)[0]
    num_sources = len(source_indices)
    
    papers1 = get_paper_mappings(dataset, topk1.indices)
    papers2 = get_paper_mappings(dataset, topk2.indices)
    
    titles1 = get_paper_titles(dataset, papers1)
    titles2 = get_paper_titles(dataset, papers2)
    
    source_paper_ids = dataset.node_to_paper_mapping.loc[source_indices.cpu()].values.flatten()
    source_titles = dataset.paper_metadata.loc[source_paper_ids].title.values
    
    all_shifts = []
    
    for i in range(num_sources):
        source_title = source_titles[i]
        source_paper_id = source_paper_ids[i]
        
        paper1_pos = {titles1[i, j]: j for j in range(k)}
        
        for j, paper_title in enumerate(titles2[i]):
            if paper_title in paper1_pos:
                pos1 = paper1_pos[paper_title]
                pos2 = j
                shift = pos2 - pos1
                
                if abs(shift) > 0:
                    all_shifts.append({
                        'source_title': source_title,
                        'source_paper_id': source_paper_id,
                        'paper_title': paper_title,
                        'pos1': pos1,
                        'pos2': pos2,
                        'shift': shift,
                        'abs_shift': abs(shift),
                        'source_idx': i
                    })
    
    all_shifts.sort(key=lambda x: x['abs_shift'], reverse=True)
    top_shifts = all_shifts[:num_display]
    
    if not top_shifts:
        fig = plt.figure()
        plt.title("No rank shifts to display")
        return fig
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for shift in top_shifts:
        pos1 = shift['pos1'] + 1
        pos2 = shift['pos2'] + 1
        y  = shift['source_idx']
        ax.plot([pos1, pos2], [y, y], 'o-', color='gray')
        color = 'red' if pos2 > pos1 else 'green'
        ax.annotate("", xy=(pos2, y), xytext=(pos1, y),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2))
        ax.text(pos1, y+0.1, f"{model_names[0]}: {pos1}", color='blue', ha='center', fontsize=8)
        ax.text(pos2, y-0.1, f"{model_names[1]}: {pos2}", color='red', ha='center', fontsize=8)
    
    ax.set_xlabel("Rank Position")
    ax.set_ylabel("Source Paper Index")
    ax.set_title(f"Largest Rank Shifts Between {model_names[0]} and {model_names[1]}")
    ax.set_xticks(range(0, k+2))
    plt.tight_layout()
    return fig

def visualize_single_embedding(titles, source_titles, similarities, k, model_name):
    """
    Visualize similar papers for a single embedding model using seaborn's heatmap.
    """
    num_sources = len(source_titles)
    fig, ax = plt.subplots(figsize=(8, max(3, num_sources * 0.5)))
    
    sim_data = similarities.detach().cpu().numpy() if hasattr(similarities, 'detach') else similarities
    
    sns.heatmap(sim_data[:, :k], cmap="viridis", ax=ax, cbar=True)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Source Paper")
    ax.set_title(f"Top {k} Similar Papers using {model_name}")
    
    ax.set_xticks(np.arange(k) + 0.5)
    ax.set_xticklabels([str(i+1) for i in range(k)])
    ax.set_yticks(np.arange(num_sources) + 0.5)
    ax.set_yticklabels([str(i+1) for i in range(num_sources)])
    plt.tight_layout()
    return fig

def visualize_embedding_comparison(titles1, titles2, source_titles, similarities1, similarities2, k, model_names):
    """
    Create a side-by-side comparison visualization using two seaborn heatmaps.
    """
    num_sources = len(source_titles)
    max_sources_to_show = min(num_sources, 5)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, max(3, max_sources_to_show * 0.8)))
    
    sim1 = similarities1.detach().cpu().numpy() if hasattr(similarities1, 'detach') else similarities1
    sim2 = similarities2.detach().cpu().numpy() if hasattr(similarities2, 'detach') else similarities2
    
    sns.heatmap(sim1[:max_sources_to_show, :k], cmap="viridis", ax=axes[0], cbar=True)
    axes[0].set_title(f"{model_names[0]} Similarity")
    axes[0].set_xlabel("Rank")
    axes[0].set_ylabel("Source")
    axes[0].set_xticks(np.arange(k) + 0.5)
    axes[0].set_xticklabels([str(i+1) for i in range(k)])
    axes[0].set_yticks(np.arange(max_sources_to_show) + 0.5)
    axes[0].set_yticklabels([source_titles[i] for i in range(max_sources_to_show)])
    
    sns.heatmap(sim2[:max_sources_to_show, :k], cmap="viridis", ax=axes[1], cbar=True)
    axes[1].set_title(f"{model_names[1]} Similarity")
    axes[1].set_xlabel("Rank")
    axes[1].set_ylabel("")
    axes[1].set_xticks(np.arange(k) + 0.5)
    axes[1].set_xticklabels([str(i+1) for i in range(k)])
    axes[1].set_yticks(np.arange(max_sources_to_show) + 0.5)
    axes[1].set_yticklabels([source_titles[i] for i in range(max_sources_to_show)])
    
    plt.suptitle(f"Comparing Top {k} Similar Papers: {model_names[0]} vs {model_names[1]}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def get_topk_similar(similarity_values, k=5, indices=None):
    """
    Get top-k most similar items from similarity values.
    
    Args:
        similarity_values: Tensor of similarity values
                          If tuple (similarities, indices), use pre-computed indices
        k: Number of top similar items to return
        indices: Optional pre-computed indices from FAISS
        
    Returns:
        namedtuple with fields 'values' and 'indices'
    """
    from collections import namedtuple
    
    TopkResults = namedtuple('TopkResults', ['values', 'indices'])
    
    if isinstance(similarity_values, tuple) and len(similarity_values) == 2:
        return TopkResults(values=similarity_values[0], indices=similarity_values[1])
    
    if indices is not None:
        return TopkResults(values=similarity_values, indices=indices)
    
    topk_values, topk_indices = torch.topk(similarity_values, k=min(k, similarity_values.shape[1]), dim=1)
    return TopkResults(values=topk_values, indices=topk_indices)

def get_paper_mappings(dataset, node_indices):
    """
    Map node indices to paper IDs.
    
    Args:
        dataset: Dataset containing node to paper mappings
        node_indices: Tensor of node indices
        
    Returns:
        2D array of paper IDs
    """
    if isinstance(node_indices, torch.Tensor):
        indices_np = node_indices.cpu().numpy()
    else:
        indices_np = node_indices
    
    paper_ids = np.zeros_like(indices_np, dtype=object)
    
    for i in range(indices_np.shape[0]):
        for j in range(indices_np.shape[1]):
            node_idx = indices_np[i, j]
            if node_idx >= 0 and node_idx < len(dataset.node_to_paper_mapping):
                try:
                    paper_ids[i, j] = dataset.node_to_paper_mapping.iloc[node_idx].values[0]
                except:
                    paper_ids[i, j] = None
            else:
                paper_ids[i, j] = None
    
    return paper_ids

def get_paper_titles(dataset, paper_ids):
    """
    Get paper titles from paper IDs.
    
    Args:
        dataset: Dataset containing paper metadata
        paper_ids: 2D array of paper IDs
        
    Returns:
        2D array of paper titles
    """
    titles = np.zeros_like(paper_ids, dtype=object)
    
    for i in range(paper_ids.shape[0]):
        for j in range(paper_ids.shape[1]):
            paper_id = paper_ids[i, j]
            if paper_id is not None and paper_id in dataset.paper_metadata.index:
                titles[i, j] = dataset.paper_metadata.loc[paper_id, 'title']
            else:
                titles[i, j] = ""
    
    return titles

# Main execution
dataset = OGBNArxivDataset()
k = 5

em1_path = "/fred/oz318/luhanc/fusion/saved_embeddings/ogbn-arxiv/textual/nvembedv2/4096.pt"
em2_path = "/fred/oz318/luhanc/fusion/logs/ASGC/256/100/1/embeddings.pt"
em1 = torch.load(em1_path, map_location=dataset.graph.device)
em2 = torch.load(em2_path, map_location=dataset.graph.device)
# mask = torch.ones_like(dataset.labels).to(dataset.graph.device)
mask = (dataset.labels == 0).to(dataset.graph.device)


# compare_embedding_fig = compare_embeddings(
#     dataset,
#     em1,
#     em2,
#     mask=mask,
#     k=k,
#     model_names=["NVEmbedV2", "ASGC"],
#     use_faiss=True,
#     embedding_paths=[em1_path, em2_path],
#     force_rebuild=False
# )
# compare_embedding_fig.show()

rank_shift_fig = visualize_largest_rank_shifts(
    dataset,
    em1,
    em2,
    mask=mask,
    k=k,
    num_display=10,
    model_names=["NVEmbedV2", "ASGC"],
    use_faiss=True,
    embedding_paths=[em1_path, em2_path],
    force_rebuild=False
)
rank_shift_fig.show()



