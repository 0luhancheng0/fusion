import torch
import numpy as np
import faiss
import time
from pathlib import Path
# ...existing imports...

# New overarching class to group FAISS-related operations and improve efficiency
class FaissManager:
    def __init__(self, embeddings, index_dir, normalize=True, force_rebuild=False):
        """
        Initialize the FAISS manager, build or load the index.
        
        Args:
            embeddings: Node embeddings tensor.
            index_dir: Directory to store/load FAISS index and related outputs.
            normalize: Whether to L2 normalize embeddings.
            force_rebuild: Force rebuilding the index.
        """
        self.embeddings = embeddings
        self.index_dir = Path(index_dir)
        self.normalize = normalize
        self.force_rebuild = force_rebuild
        self.index = None
        self.embeddings_np = None
        self._build_index()
    
    def _build_index(self):
        self.index_dir.mkdir(parents=True, exist_ok=True)
        paths = self.get_file_paths(self.index_dir)
        if paths["index"].exists() and paths["embeddings"].exists() and not self.force_rebuild:
            print(f"Loading existing FAISS index from {paths['index']}")
            self.index, self.embeddings_np = self.load_index_and_embeddings(self.index_dir)
        else:
            print(f"Building new FAISS index and saving to {paths['index']}")
            start_time = time.time()
            self.index, self.embeddings_np = self.build_faiss_index(self.embeddings, normalize=self.normalize)
            build_time = time.time() - start_time
            print(f"FAISS index built in {build_time:.3f} seconds")
            self.save_index_and_embeddings(self.index, self.embeddings_np, self.index_dir)
            print(f"FAISS index and embeddings saved to {self.index_dir}")

    @staticmethod
    def build_faiss_index(embeddings, normalize=True):
        # ...existing code from build_faiss_index...
        if isinstance(embeddings, torch.Tensor):
            embeddings_np = embeddings.detach().cpu().numpy().astype('float32')
            embeddings_np = np.ascontiguousarray(embeddings_np)
        else:
            embeddings_np = np.ascontiguousarray(embeddings.astype('float32'))
        if normalize:
            faiss.normalize_L2(embeddings_np)
        dimension = embeddings_np.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_np)
        return index, embeddings_np

    @staticmethod
    def faiss_search_topk(index, query_embeddings, k=5, normalize=True):
        # ...existing code from faiss_search_topk...
        if isinstance(query_embeddings, torch.Tensor):
            query_np = query_embeddings.detach().cpu().numpy().astype('float32')
            query_np = np.ascontiguousarray(query_np)
        else:
            query_np = np.ascontiguousarray(query_embeddings.astype('float32'))
        if normalize:
            faiss.normalize_L2(query_np)
        similarities, indices = index.search(query_np, k)
        return similarities, indices

    @staticmethod
    def get_file_paths(directory):
        directory = Path(directory)
        return {
            "index": directory / "faiss_index.bin",
            "embeddings": directory / "normalized_embeddings.npy",
            "all_similarities": directory / "all_similarities.npy",
            "all_indices": directory / "all_indices.npy",
        }

    @staticmethod
    def save_index_and_embeddings(index, embeddings_np, directory):
        paths = FaissManager.get_file_paths(directory)
        faiss.write_index(index, str(paths["index"]))
        np.save(paths["embeddings"], embeddings_np)

    @staticmethod
    def load_index_and_embeddings(directory):
        paths = FaissManager.get_file_paths(directory)
        if not paths["index"].exists() or not paths["embeddings"].exists():
            raise FileNotFoundError(f"FAISS index or embeddings not found in {directory}")
        index = faiss.read_index(str(paths["index"]))
        embeddings_np = np.load(paths["embeddings"])
        return index, embeddings_np

    @staticmethod
    def save_similarity_results(similarities, indices, directory):
        paths = FaissManager.get_file_paths(directory)
        np.save(paths["all_similarities"], similarities)
        np.save(paths["all_indices"], indices)
        print(f"All similarities and indices saved to {directory}")

    def compute_similarity_with_mask(self, mask, k=None):
        """
        Compute top-k similarities for nodes specified by the mask.
        
        Args:
            mask: Boolean mask for nodes.
            k: Number of similar items per query (if None, defaults to 100 or number of nodes).
            
        Returns:
            Tuple of torch tensors for similarities and indices.
        """
        masked_indices = torch.where(mask)[0].cpu().numpy()
        query_embeddings = self.embeddings_np[masked_indices]
        if k is None:
            k = min(self.embeddings.shape[0], 100)
        search_k = k + 1
        print(f"Searching for top-{k} similar items for {len(masked_indices)} queries...")
        start_time = time.time()
        similarities, indices = self.faiss_search_topk(self.index, query_embeddings, k=search_k, normalize=False)
        search_time = time.time() - start_time
        print(f"Search completed in {search_time:.3f} seconds")
        filtered_similarities = np.zeros((len(masked_indices), k), dtype=similarities.dtype)
        filtered_indices = np.zeros((len(masked_indices), k), dtype=indices.dtype)
        for i, idx in enumerate(masked_indices):
            if indices[i, 0] == idx:
                end_idx = min(search_k, k+1)
                filtered_similarities[i, :end_idx-1] = similarities[i, 1:end_idx]
                filtered_indices[i, :end_idx-1] = indices[i, 1:end_idx]
                if end_idx-1 < k:
                    remaining = k - (end_idx-1)
                    padding_sim = np.zeros(remaining, dtype=similarities.dtype)
                    padding_idx = np.ones(remaining, dtype=indices.dtype) * -1
                    filtered_similarities[i, end_idx-1:] = padding_sim
                    filtered_indices[i, end_idx-1:] = padding_idx
            else:
                end_idx = min(search_k, k)
                filtered_similarities[i, :end_idx] = similarities[i, :end_idx]
                filtered_indices[i, :end_idx] = indices[i, :end_idx]
                if end_idx < k:
                    remaining = k - end_idx
                    padding_sim = np.zeros(remaining, dtype=similarities.dtype)
                    padding_idx = np.ones(remaining, dtype=indices.dtype) * -1
                    filtered_similarities[i, end_idx:] = padding_sim
                    filtered_indices[i, end_idx:] = padding_idx
        similarities_torch = torch.from_numpy(filtered_similarities).to(self.embeddings.device)
        indices_torch = torch.from_numpy(filtered_indices).to(self.embeddings.device)
        return similarities_torch, indices_torch

    def compute_all_similarities(self):
        """
        Compute similarity scores for all nodes (excluding self-similarity) and save the results.
        
        Returns:
            Tuple of torch tensors for similarities and indices.
        """
        n = self.embeddings_np.shape[0]
        print(f"Computing all similarities for {n} nodes...")
        similarities, indices = self.faiss_search_topk(self.index, self.embeddings_np, k=n, normalize=False)
        filtered_similarities = np.zeros((n, n-1), dtype=similarities.dtype)
        filtered_indices = np.zeros((n, n-1), dtype=indices.dtype)
        for i in range(n):
            if indices[i, 0] == i:
                filtered_similarities[i, :] = similarities[i, 1:n]
                filtered_indices[i, :] = indices[i, 1:n]
            else:
                filtered_similarities[i, :] = similarities[i, :n-1]
                filtered_indices[i, :] = indices[i, :n-1]
        self.save_similarity_results(filtered_similarities, filtered_indices, self.index_dir)
        similarities_torch = torch.from_numpy(filtered_similarities).to(self.embeddings.device)
        indices_torch = torch.from_numpy(filtered_indices).to(self.embeddings.device)
        return similarities_torch, indices_torch

# ...existing code...
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

def get_file_paths(directory):
    """
    Return a dict of file paths for the FAISS index, embeddings,
    and similarity results inside the given directory.
    """
    directory = Path(directory)
    return {
        "index": directory / "faiss_index.bin",
        "embeddings": directory / "normalized_embeddings.npy",
        "all_similarities": directory / "all_similarities.npy",
        "all_indices": directory / "all_indices.npy",
    }

def save_index_and_embeddings(index, embeddings_np, directory):
    """
    Save the FAISS index and normalized embeddings to the given directory.
    """
    paths = get_file_paths(directory)
    faiss.write_index(index, str(paths["index"]))
    np.save(paths["embeddings"], embeddings_np)

def load_index_and_embeddings(directory):
    """
    Load the FAISS index and normalized embeddings from the given directory.
    """
    paths = get_file_paths(directory)
    if not paths["index"].exists() or not paths["embeddings"].exists():
        raise FileNotFoundError(f"FAISS index or embeddings not found in {directory}")
    index = faiss.read_index(str(paths["index"]))
    embeddings_np = np.load(paths["embeddings"])
    return index, embeddings_np

def save_similarity_results(similarities, indices, directory):
    """
    Save computed similarities and indices to the given directory.
    """
    paths = get_file_paths(directory)
    np.save(paths["all_similarities"], similarities)
    np.save(paths["all_indices"], indices)
    print(f"All similarities and indices saved to {directory}")

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
    paths = get_file_paths(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Check if index already exists and we're not forcing a rebuild
    if paths["index"].exists() and paths["embeddings"].exists() and not force_rebuild:
        print(f"Loading existing FAISS index from {paths['index']}")
        return load_faiss_index(save_path)
    
    # Build the index
    print(f"Building new FAISS index and saving to {paths['index']}")
    start_time = time.time()
    index, embeddings_np = build_faiss_index(embeddings, normalize=normalize)
    build_time = time.time() - start_time
    print(f"FAISS index built in {build_time:.3f} seconds")
    
    # Save the index and embeddings
    save_index_and_embeddings(index, embeddings_np, save_path)
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
    start_time = time.time()
    index, embeddings_np = load_index_and_embeddings(load_path)
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

def compute_all_similarities_with_saved_faiss(embeddings, embedding_path, normalize=True, 
                                              index_root="/home/lcheng/oz318/fusion/faiss_indices", 
                                              force_rebuild=False):
    """
    Compute similarity scores for all nodes using a FAISS index and save the results
    (excluding self-similarity) into the same directory as the index file.
    
    Args:
        embeddings: Node embeddings tensor
        embedding_path: Path to the embedding file (used to determine index location)
        normalize: Whether to normalize embeddings before building the index
        index_root: Root directory for storing FAISS indices
        force_rebuild: Whether to rebuild the index even if it already exists
        
    Returns:
        Tuple of (similarities, indices) as torch tensors (without self-similarity)
    """
    # Determine where to save/load the index
    index_dir = get_index_path_from_embedding_path(embedding_path, index_root)
    
    # Build and save index or load existing one
    index, embeddings_np = build_and_save_faiss_index(
        embeddings, index_dir, normalize=normalize, force_rebuild=force_rebuild
    )
    
    n = embeddings_np.shape[0]
    print(f"Computing all similarities for {n} nodes...")
    
    # Search for all neighbors (retrieve full set)
    similarities, indices = faiss_search_topk(index, embeddings_np, k=n, normalize=False)
    
    # Remove self-similarity
    filtered_similarities = np.zeros((n, n-1), dtype=similarities.dtype)
    filtered_indices = np.zeros((n, n-1), dtype=indices.dtype)
    
    for i in range(n):
        if indices[i, 0] == i:
            filtered_similarities[i, :] = similarities[i, 1:n]
            filtered_indices[i, :] = indices[i, 1:n]
        else:
            filtered_similarities[i, :] = similarities[i, :n-1]
            filtered_indices[i, :] = indices[i, :n-1]
    
    # Save the computed similarities and indices in the same directory as the index file
    save_similarity_results(filtered_similarities, filtered_indices, index_dir)
    
    # Convert to torch tensors before returning
    similarities_torch = torch.from_numpy(filtered_similarities).to(embeddings.device)
    indices_torch = torch.from_numpy(filtered_indices).to(embeddings.device)
    
    return similarities_torch, indices_torch
