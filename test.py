import torch
import dgl
import time
from evaluation import NodeEmbeddingEvaluator
from dataloading import OGBNArxivDataset

def test_hard_negative_sampling():
    """
    Test function for hard negative sampling.
    Creates a test graph and verifies that the hard negative samples are correct.
    
    Returns:
        bool: True if the test passed, False otherwise
    """
    print("Testing hard negative sampling...")
    
    # Create a simple test graph with a known structure
    # 0 -- 1 -- 2 -- 3
    # |         |
    # 4 -- 5 -- 6
    
    src = torch.tensor([0, 1, 1, 2, 2, 0, 4, 4, 5, 5, 6, 6])
    dst = torch.tensor([1, 0, 2, 1, 3, 4, 0, 5, 4, 6, 5, 2])
    
    g = dgl.graph((src, dst))
    print(f"Created test graph with {g.num_nodes()} nodes and {g.num_edges()} edges")
    
    # Get hard negatives
    neg_src, neg_dst = NodeEmbeddingEvaluator.optimized_hard_negative_sampling(
        g, num_samples=None, fallback_random=False, node_sampling_fraction=1.0
    )
    
    print(f"Found {len(neg_src)} hard negative pairs")
    
    # Verify hard negatives
    test_passed = True
    
    # Check if all returned pairs are valid hard negatives
    for i in range(len(neg_src)):
        s, d = neg_src[i].item(), neg_dst[i].item()
        
        # Check they are not directly connected
        if g.has_edges_between(torch.tensor([s]), torch.tensor([d])):
            print(f"Error: ({s}, {d}) are directly connected!")
            test_passed = False
        
        # Check they are 2-hops away
        one_hop = set(g.successors(s).tolist())
        valid_two_hop = False
        for n in one_hop:
            if d in g.successors(n).tolist():
                valid_two_hop = True
                break
        
        if not valid_two_hop:
            print(f"Error: ({s}, {d}) are not 2-hops away!")
            test_passed = False
    
    # Create a dictionary to store all 2-hop neighbors for each node
    two_hop_dict = {}
    for node in range(g.num_nodes()):
        neighbors = set(g.successors(node).tolist())
        two_hop_neighbors = set()
        for n in neighbors:
            two_hop_neighbors.update(g.successors(n).tolist())
        # Remove self and direct neighbors
        two_hop_neighbors.discard(node)
        two_hop_neighbors = two_hop_neighbors - neighbors
        two_hop_dict[node] = two_hop_neighbors
    
    # Check for some expected hard negatives
    expected_pairs = [(0, 2), (0, 6), (4, 2), (4, 3), (1, 4), (1, 5), (1, 6), (2, 4), (2, 5), (3, 6)]
    found_pairs = set([(s.item(), d.item()) for s, d in zip(neg_src, neg_dst)])
    
    print("Examples of found hard negative pairs:")
    for i in range(min(5, len(neg_src))):
        print(f"({neg_src[i].item()}, {neg_dst[i].item()})")
    
    missing = []
    for pair in expected_pairs:
        if pair not in found_pairs and (pair[1], pair[0]) not in found_pairs:
            missing.append(pair)
    
    if missing:
        print(f"Warning: Some expected pairs were not found: {missing}")
        print("This might be normal if there are many possible hard negatives")
    
    if test_passed:
        print("Hard negative sampling test PASSED!")
    else:
        print("Hard negative sampling test FAILED!")
    
    return test_passed

def test_optimized_hard_negative_sampling_arxiv():
    """
    Test optimized hard negative sampling on the OGBN-ArXiv graph.
    This tests the new optimized implementation with multiprocessing.
    
    Returns:
        bool: True if the test passed, False otherwise
    """
    print("\nTesting optimized hard negative sampling on OGBN-ArXiv dataset...")
    
    # Load the arxiv graph
    print("Loading OGBN-ArXiv graph...")
    graph = OGBNArxivDataset(add_reverse=False, add_self_loop=False).graph
    num_nodes = graph.num_nodes()
    num_edges = graph.num_edges()
    print(f"Graph loaded with {num_nodes} nodes and {num_edges} edges")

    # Test different parameters
    test_configs = [
        {"num_samples": 1000, "fallback_random": False, "max_samples_per_node": 10, 
         "num_workers": 2, "node_sampling_fraction": 0.05},
        {"num_samples": 5000, "fallback_random": True, "max_samples_per_node": 20, 
         "num_workers": 4, "node_sampling_fraction": 0.1}
    ]
    
    all_valid = True
    for config in test_configs:
        print(f"\nConfig: {config}")
        
        # Run optimized hard negative sampling with timing
        start_time = time.time()
        neg_src, neg_dst = NodeEmbeddingEvaluator.optimized_hard_negative_sampling(
            graph, **config
        )
        elapsed_time = time.time() - start_time
        
        print(f"Found {len(neg_src)} hard negative pairs in {elapsed_time:.3f} seconds")
        
        # Verify a subset of the hard negatives
        verify_size = min(100, len(neg_src))
        indices = torch.randperm(len(neg_src))[:verify_size]
        
        print(f"Verifying {verify_size} random negative pairs...")
        
        valid_pairs = 0
        # Create a bidirectional version for validation
        bidir_graph = dgl.to_bidirected(graph, copy_ndata=False)
        
        for idx in indices:
            s, d = neg_src[idx].item(), neg_dst[idx].item()
            
            # Check they are not directly connected
            if graph.has_edges_between(torch.tensor([s]), torch.tensor([d])):
                print(f"Error: ({s}, {d}) are directly connected!")
                all_valid = False
                continue
            
            # If fallback_random is True, we might have random negatives, so we'll skip the 2-hop check
            if config["fallback_random"]:
                valid_pairs += 1
                continue
                
            # Check they are 2-hops away using the bidirectional graph
            one_hop = set(bidir_graph.successors(s).tolist())
            two_hop = False
            for n in one_hop:
                if d in bidir_graph.successors(n).tolist():
                    two_hop = True
                    break
            
            if not two_hop:
                print(f"Error: ({s}, {d}) are not 2-hops away!")
                all_valid = False
            else:
                valid_pairs += 1
        
        print(f"Valid pairs found: {valid_pairs}/{verify_size}")
    
    if all_valid:
        print("\nOptimized hard negative sampling on ArXiv PASSED!")
    else:
        print("\nOptimized hard negative sampling on ArXiv FAILED!")
    
    return all_valid

def compare_performance():
    """
    Compare the performance of the original and optimized implementations.
    
    Returns:
        tuple: (original_time, optimized_time)
    """
    print("\nComparing performance of original vs optimized implementations...")
    
    # Load graph
    graph = OGBNArxivDataset(add_reverse=False, add_self_loop=False).graph
    sample_count = 5000
    
    try:
        # Test original implementation if available
        original_available = hasattr(NodeEmbeddingEvaluator, "hard_negative_sampling")
        
        if original_available:
            print(f"Running original implementation to generate {sample_count} samples...")
            start_time = time.time()
            NodeEmbeddingEvaluator.hard_negative_sampling(
                graph, num_samples=sample_count, fallback_random=True
            )
            original_time = time.time() - start_time
            print(f"Original implementation: {original_time:.3f} seconds")
        else:
            original_time = float('inf')
            print("Original implementation not available")
        
        # Test optimized implementation
        print(f"Running optimized implementation to generate {sample_count} samples...")
        start_time = time.time()
        NodeEmbeddingEvaluator.optimized_hard_negative_sampling(
            graph, num_samples=sample_count, fallback_random=True, 
            num_workers=4, node_sampling_fraction=0.1
        )
        optimized_time = time.time() - start_time
        print(f"Optimized implementation: {optimized_time:.3f} seconds")
        
        if original_available and original_time != float('inf'):
            speedup = original_time / optimized_time
            print(f"Speedup: {speedup:.2f}x")
        
        return (original_time, optimized_time)
    
    except Exception as e:
        print(f"Error during performance comparison: {e}")
        import traceback
        traceback.print_exc()
        return (None, None)

if __name__ == "__main__":
    print("Running hard negative sampling tests...")
    test_passed_basic = test_hard_negative_sampling()
    test_passed_arxiv = test_optimized_hard_negative_sampling_arxiv()
    
    # Run performance comparison if tests pass
    if test_passed_basic and test_passed_arxiv:
        print("\nAll tests PASSED! Running performance comparison...")
        compare_performance()
    else:
        print("\nSome tests FAILED!")

