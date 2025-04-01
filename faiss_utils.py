import torch
import numpy as np
import faiss
import time
from pathlib import Path

def build_faiss_index(embeddings, normalize=True):
    """
    Build a FAISS index for fast similarity search.
    
    Args:
        embeddings: Node embeddings tensor
        normalize: Whether to normalize embeddings before building index
        
    Returns:
        FAISS index, normalized embeddings as numpy array
    """
    # Convert to numpy and ensure it's C-contiguous
    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.detach().cpu().numpy().astype('float32')
        # Ensure the array is C-contiguous for FAISS
        embeddings_np = np.ascontiguousarray(embeddings_np)
    else:
        embeddings_np = np.ascontiguousarray(embeddings.astype('float32'))
    
    if normalize:
        # L2 normalize embeddings
        faiss.normalize_L2(embeddings_np)
    
    # Create inner product index (for cosine similarity when vectors are normalized)
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatIP(dimension)
    
    # Add vectors to the index
    index.add(embeddings_np)
    
    return index, embeddings_np

def faiss_search_topk(index, query_embeddings, k=5, normalize=True):
    """
    Search for top-k similar items using FAISS index.
    
    Args:
        index: FAISS index
        query_embeddings: Query embedding vectors
        k: Number of nearest neighbors to retrieve
        normalize: Whether to normalize query embeddings
        
    Returns:
        Tuple of (similarity values, indices)
    """
    # Convert to numpy if it's a torch tensor and ensure it's C-contiguous
    if isinstance(query_embeddings, torch.Tensor):
        query_np = query_embeddings.detach().cpu().numpy().astype('float32')
        # Ensure the array is C-contiguous for FAISS
        query_np = np.ascontiguousarray(query_np)
    else:
        query_np = np.ascontiguousarray(query_embeddings.astype('float32'))
    
    if normalize:
        # L2 normalize query vectors
        faiss.normalize_L2(query_np)
    
    # Search for top-k similar items
    similarities, indices = index.search(query_np, k)
    
    return similarities, indices

def build_and_save_faiss_index(embeddings, save_path, normalize=True, force_rebuild=False):
    """
    Build a FAISS index for embeddings and save it to disk.
    
    Args:
        embeddings: Node embeddings tensor
        save_path: Path to save the FAISS index
        normalize: Whether to normalize embeddings before building index
        force_rebuild: Whether to rebuild the index even if it already exists
        
    Returns:
        Tuple of (FAISS index, normalized embeddings as numpy array)
    """
    save_path = Path(save_path)
    index_path = save_path / "faiss_index.bin"
    embeddings_path = save_path / "normalized_embeddings.npy"
    
    # Create directory if it doesn't exist
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Check if index already exists and we're not forcing a rebuild
    if index_path.exists() and embeddings_path.exists() and not force_rebuild:
        print(f"Loading existing FAISS index from {index_path}")
        return load_faiss_index(save_path)
    
    # Build the index
    print(f"Building new FAISS index and saving to {index_path}")
    start_time = time.time()
    index, embeddings_np = build_faiss_index(embeddings, normalize=normalize)
    build_time = time.time() - start_time
    print(f"FAISS index built in {build_time:.3f} seconds")
    
    # Save the index and embeddings
    faiss.write_index(index, str(index_path))
    np.save(embeddings_path, embeddings_np)
    print(f"FAISS index and embeddings saved to {save_path}")
    
    return index, embeddings_np

def load_faiss_index(load_path):
    """
    Load a FAISS index and normalized embeddings from disk.
    
    Args:
        load_path: Path containing the saved FAISS index and embeddings
        
    Returns:
        Tuple of (FAISS index, normalized embeddings as numpy array)
    """
    load_path = Path(load_path)
    index_path = load_path / "faiss_index.bin"
    embeddings_path = load_path / "normalized_embeddings.npy"
    
    if not index_path.exists() or not embeddings_path.exists():
        raise FileNotFoundError(f"FAISS index or embeddings not found in {load_path}")
    
    # Load the index and embeddings
    start_time = time.time()
    index = faiss.read_index(str(index_path))
    embeddings_np = np.load(embeddings_path)
    load_time = time.time() - start_time
    print(f"FAISS index and embeddings loaded in {load_time:.3f} seconds")
    
    return index, embeddings_np

def get_index_path_from_embedding_path(embedding_path, index_root="/home/lcheng/oz318/fusion/faiss_indices"):
    """
    Generate a path for saving a FAISS index based on the embedding path.
    
    Args:
        embedding_path: Path to the embedding file
        index_root: Root directory for storing FAISS indices
        
    Returns:
        Path object for the FAISS index directory
    """
    # Convert to Path object
    embedding_path = Path(embedding_path)
    index_root = Path(index_root)
    
    # Create a directory name based on the embedding path
    # Remove leading slashes and replace path separators with underscores
    rel_path = str(embedding_path).lstrip("/").replace("/", "_")
    
    # Return the full path
    return index_root / rel_path

def compute_similarity_with_saved_faiss(embeddings, mask, embedding_path=None, k=None, normalize=True, 
                                      index_root="/home/lcheng/oz318/fusion/faiss_indices", force_rebuild=False):
    """
    Compute similarities using a saved FAISS index or build and save a new one.
    
    Args:
        embeddings: Node embeddings tensor
        mask: Boolean mask indicating which nodes to compute similarity for
        embedding_path: Path to the embedding file (used to determine index location)
        k: Number of top similar items to return per query (if None, returns all)
        normalize: Whether to normalize embeddings
        index_root: Root directory for storing FAISS indices
        force_rebuild: Whether to rebuild the index even if it already exists
        
    Returns:
        Tuple of (similarities, indices) for top-k similar items
    """
    if embedding_path is None:
        # If no embedding path provided, use in-memory index without saving
        return compute_similarity_with_faiss(embeddings, mask, k=k, normalize=normalize)
    
    # Determine where to save/load the index
    index_path = get_index_path_from_embedding_path(embedding_path, index_root)
    
    # Build and save index or load existing one
    index, embeddings_np = build_and_save_faiss_index(
        embeddings, index_path, normalize=normalize, force_rebuild=force_rebuild
    )
    
    # Get the indices of masked nodes
    masked_indices = torch.where(mask)[0].cpu().numpy()
    
    # Get query embeddings
    query_embeddings = embeddings_np[masked_indices]
    
    # Set default k if not provided
    if k is None:
        k = min(embeddings.shape[0], 100)  # Default to 100 or all nodes if fewer
    
    # Request k+1 results to account for potential self-similarity
    search_k = k + 1
    
    # Perform search
    print(f"Searching for top-{k} similar items for {len(masked_indices)} queries...")
    start_time = time.time()
    similarities, indices = faiss_search_topk(index, query_embeddings, k=search_k, normalize=False)
    search_time = time.time() - start_time
    print(f"Search completed in {search_time:.3f} seconds")
    
    # Filter results (same logic as in compute_similarity_with_faiss)
    filtered_similarities = np.zeros((len(masked_indices), k), dtype=similarities.dtype)
    filtered_indices = np.zeros((len(masked_indices), k), dtype=indices.dtype)
    
    for i, idx in enumerate(masked_indices):
        # Check if the first result is the self node
        if indices[i, 0] == idx:
            # If first result is self, take the next k results
            end_idx = min(search_k, k+1)
            filtered_similarities[i, :end_idx-1] = similarities[i, 1:end_idx]
            filtered_indices[i, :end_idx-1] = indices[i, 1:end_idx]
            
            # Pad if needed
            if end_idx-1 < k:
                remaining = k - (end_idx-1)
                padding_sim = np.zeros(remaining, dtype=similarities.dtype)
                padding_idx = np.ones(remaining, dtype=indices.dtype) * -1
                filtered_similarities[i, end_idx-1:] = padding_sim
                filtered_indices[i, end_idx-1:] = padding_idx
        else:
            # If first result is not self, take the first k results
            end_idx = min(search_k, k)
            filtered_similarities[i, :end_idx] = similarities[i, :end_idx]
            filtered_indices[i, :end_idx] = indices[i, :end_idx]
            
            # Pad if needed
            if end_idx < k:
                remaining = k - end_idx
                padding_sim = np.zeros(remaining, dtype=similarities.dtype)
                padding_idx = np.ones(remaining, dtype=indices.dtype) * -1
                filtered_similarities[i, end_idx:] = padding_sim
                filtered_indices[i, end_idx:] = padding_idx
    
    # Convert to torch tensors
    similarities_torch = torch.from_numpy(filtered_similarities).to(embeddings.device)
    indices_torch = torch.from_numpy(filtered_indices).to(embeddings.device)
    
    return similarities_torch, indices_torch

def compute_similarity_with_faiss(embeddings, mask, k=None, normalize=True):
    """
    Compute similarities using in-memory FAISS index.
    
    Args:
        embeddings: Node embeddings tensor
        mask: Boolean mask indicating which nodes to compute similarity for
        k: Number of top similar items to return per query (if None, returns all)
        normalize: Whether to normalize embeddings
        
    Returns:
        Tuple of (similarities, indices) for top-k similar items
    """
    # Build in-memory index
    index, embeddings_np = build_faiss_index(embeddings, normalize=normalize)
    
    # Get the indices of masked nodes
    masked_indices = torch.where(mask)[0].cpu().numpy()
    
    # Get query embeddings
    query_embeddings = embeddings_np[masked_indices]
    
    # Set default k if not provided
    if k is None:
        k = min(embeddings.shape[0], 100)  # Default to 100 or all nodes if fewer
    
    # Request k+1 results to account for potential self-similarity
    search_k = k + 1
    
    # Perform search
    print(f"Searching for top-{k} similar items for {len(masked_indices)} queries...")
    start_time = time.time()
    similarities, indices = faiss_search_topk(index, query_embeddings, k=search_k, normalize=False)
    search_time = time.time() - start_time
    print(f"Search completed in {search_time:.3f} seconds")
    
    # Filter results to remove self-similarity
    filtered_similarities = np.zeros((len(masked_indices), k), dtype=similarities.dtype)
    filtered_indices = np.zeros((len(masked_indices), k), dtype=indices.dtype)
    
    for i, idx in enumerate(masked_indices):
        # Check if the first result is the self node
        if indices[i, 0] == idx:
            # If first result is self, take the next k results
            end_idx = min(search_k, k+1)
            filtered_similarities[i, :end_idx-1] = similarities[i, 1:end_idx]
            filtered_indices[i, :end_idx-1] = indices[i, 1:end_idx]
            
            # Pad if needed
            if end_idx-1 < k:
                remaining = k - (end_idx-1)
                padding_sim = np.zeros(remaining, dtype=similarities.dtype)
                padding_idx = np.ones(remaining, dtype=indices.dtype) * -1
                filtered_similarities[i, end_idx-1:] = padding_sim
                filtered_indices[i, end_idx-1:] = padding_idx
        else:
            # If first result is not self, take the first k results
            end_idx = min(search_k, k)
            filtered_similarities[i, :end_idx] = similarities[i, :end_idx]
            filtered_indices[i, :end_idx] = indices[i, :end_idx]
            
            # Pad if needed
            if end_idx < k:
                remaining = k - end_idx
                padding_sim = np.zeros(remaining, dtype=similarities.dtype)
                padding_idx = np.ones(remaining, dtype=indices.dtype) * -1
                filtered_similarities[i, end_idx:] = padding_sim
                filtered_indices[i, end_idx:] = padding_idx
    
    # Convert to torch tensors
    similarities_torch = torch.from_numpy(filtered_similarities).to(embeddings.device)
    indices_torch = torch.from_numpy(filtered_indices).to(embeddings.device)
    
    return similarities_torch, indices_torch
