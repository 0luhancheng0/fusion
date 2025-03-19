import os
import torch
import dgl
import typer
import json
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from evaluation import NodeEmbeddingEvaluator
from ogb.nodeproppred import DglNodePropPredDataset
from dataloading import OGBNArxivDataset
from dgl.nn.pytorch.link import EdgePredictor
from tqdm import tqdm
from torchmetrics.functional.classification import auroc

app = typer.Typer(help="Generate hard negative samples for ogbn-arxiv dataset")

def _load_graph(use_gpu: bool = True, add_reverse: bool = False) -> dgl.DGLGraph:
    """Load the ogbn-arxiv dataset graph and place it on the appropriate device."""
    graph = OGBNArxivDataset(add_reverse=add_reverse, load_metadata=False).graph
    device = torch.device('cuda' if (torch.cuda.is_available() and use_gpu) else 'cpu')
    graph = graph.to(device)
    print(f"Graph placed on {device}")
    print(f"Graph statistics: {graph.num_nodes()} nodes, {graph.num_edges()} edges")
    return graph

def _prepare_output_directory(output_dir: str, output_file: str) -> Path:
    """Create output directory if it doesn't exist and return the save path."""
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    save_path = out_dir / output_file
    return save_path

def _determine_sample_size(
    graph: dgl.DGLGraph, 
    edge_multiplier: float = 1.0, 
    absolute_samples: Optional[int] = None
) -> int:
    """Determine number of samples based on parameters."""
    if absolute_samples is not None:
        num_samples = absolute_samples
        print(f"Using specified sample count: {num_samples}")
    else:
        num_samples = int(graph.num_edges() * edge_multiplier)
        print(f"Using {edge_multiplier}x the number of edges: {num_samples} samples")
    return num_samples

def _load_embeddings(
    embed_file: Path, 
    device: torch.device
) -> torch.Tensor:
    """Load embeddings from file and handle different storage formats."""
    try:
        embeddings = torch.load(embed_file)
        if isinstance(embeddings, dict):
            # Handle case where embeddings are stored as a dict
            if 'embeddings' in embeddings:
                embeddings = embeddings['embeddings']
            elif 'embedding' in embeddings:
                embeddings = embeddings['embedding']
            else:
                # Use the first value if keys don't match expected names
                embeddings = next(iter(embeddings.values()))
                
        embeddings = embeddings.to(device)
        print(f"Loaded embeddings with shape: {embeddings.shape}")
        return embeddings
    except Exception as e:
        print(f"Error loading embeddings from {embed_file}: {e}")
        raise

def _load_existing_results(results_path: Path) -> Dict:
    """Load existing results from a JSON file."""
    existing_results = {}
    if results_path.exists():
        try:
            with open(results_path, 'r') as f:
                existing_results = json.load(f)
            print(f"Loaded existing results from {results_path}")
        except Exception as e:
            print(f"Error loading existing results: {e}")
    return existing_results

def _save_results(results_path: Path, results: Dict) -> None:
    """Save results to a JSON file."""
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

def _evaluate_link_prediction(
    embeddings: torch.Tensor,
    pos_src: torch.Tensor,
    pos_dst: torch.Tensor,
    neg_src: torch.Tensor,
    neg_dst: torch.Tensor,
    batch_size: int,
    device: torch.device
) -> float:
    """Evaluate link prediction using batched processing."""
    predictor = EdgePredictor("cos")
    
    # Process positive edges in batches
    all_pos_scores = []
    for i in range(0, len(pos_src), batch_size):
        batch_src = pos_src[i:i+batch_size]
        batch_dst = pos_dst[i:i+batch_size]
        
        with torch.no_grad():
            batch_pos_scores = predictor(embeddings[batch_src], embeddings[batch_dst]).squeeze()
            all_pos_scores.append(batch_pos_scores)
    
    pos_scores = torch.cat(all_pos_scores)
    
    # Process negative edges in batches
    all_neg_scores = []
    for i in range(0, len(neg_src), batch_size):
        batch_neg_src = neg_src[i:i+batch_size]
        batch_neg_dst = neg_dst[i:i+batch_size]
        
        with torch.no_grad():
            batch_neg_scores = predictor(embeddings[batch_neg_src], embeddings[batch_neg_dst]).squeeze()
            all_neg_scores.append(batch_neg_scores)
    
    neg_scores = torch.cat(all_neg_scores)
    
    # Create labels and evaluate
    labels = torch.cat([
        torch.ones(pos_scores.shape[0], device=device),
        torch.zeros(neg_scores.shape[0], device=device)
    ]).type(torch.bool)
    
    scores = torch.cat([pos_scores, neg_scores])
    
    # Calculate AUROC
    return auroc(scores, labels, task="binary").item()

def _evaluate_embeddings(
    embeddings_dir: str,
    negatives_path: str,
    batch_size: int,
    use_gpu: bool,
    file_pattern: str,
    result_key: str
) -> Dict:
    """Common evaluation function for both hard and uniform negative samples."""
    # Load graph
    graph = _load_graph(use_gpu=use_gpu, add_reverse=False)
    device = graph.device
    
    # Load negative samples
    try:
        neg_data = torch.load(negatives_path)
        neg_src = neg_data['src'].to(device)
        neg_dst = neg_data['dst'].to(device)
        print(f"Loaded {len(neg_src)} negative pairs from {negatives_path}")
    except Exception as e:
        print(f"Error loading negatives: {e}")
        return {}
    
    # Get real edges
    pos_src, pos_dst = graph.edges()
    
    # Find all embedding files recursively
    embedding_dir = Path(embeddings_dir)
    embed_files = list(embedding_dir.rglob(file_pattern))
    
    if not embed_files:
        print(f"No embedding files found in {embeddings_dir} or subdirectories matching pattern {file_pattern}")
        return {}
    
    print(f"Found {len(embed_files)} embedding files")
    
    # Prepare results
    overall_results = {}
    
    # Process each embedding file
    for embed_file in tqdm(embed_files, desc="Evaluating embeddings"):
        file_name = embed_file.name
        print(f"\nEvaluating {file_name}...")
        
        # Get directory containing the embedding file
        embedding_parent = embed_file.parent
        results_path = embedding_parent / "results.json"
        
        # Load existing results if available
        existing_results = _load_existing_results(results_path)
        
        # Load embeddings
        try:
            embeddings = _load_embeddings(embed_file, device)
        except Exception:
            continue
        
        # Evaluate link prediction
        try:
            auc_score = _evaluate_link_prediction(
                embeddings=embeddings,
                pos_src=pos_src,
                pos_dst=pos_dst,
                neg_src=neg_src,
                neg_dst=neg_dst,
                batch_size=batch_size,
                device=device
            )
            
            print(f"AUROC: {auc_score:.4f}")
            
            # Add result to existing results
            existing_results[result_key] = float(auc_score)
            overall_results[file_name] = auc_score
            
            # Save updated results
            _save_results(results_path, existing_results)
            
        except Exception as e:
            print(f"Error evaluating {embed_file}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n==== Evaluation Summary ====")
    for name, score in sorted(overall_results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: {score:.4f}")
    
    # Save overall results
    _save_overall_results(embedding_dir, result_key, overall_results)
    
    return overall_results

def _save_overall_results(
    embedding_dir: Path,
    result_key: str,
    overall_results: Dict
) -> None:
    """Save overall results to all_results.json in the embeddings directory."""
    all_results_path = embedding_dir / "all_results.json"
    
    # Load existing overall results if available
    existing_overall_results = _load_existing_results(all_results_path)
    
    # Update with new results
    if result_key not in existing_overall_results:
        existing_overall_results[result_key] = {}
    
    existing_overall_results[result_key].update(
        {name: float(score) for name, score in overall_results.items()}
    )
    
    # Save updated overall results
    _save_results(all_results_path, existing_overall_results)

@app.command()
def generate_hard_negatives(
    output_dir: str = typer.Option("logs", help="Directory to save the generated samples"),
    output_file: str = typer.Option("arxiv_hard_negatives.pt", help="Filename for the samples"),
    edge_multiplier: float = typer.Option(1.0, help="Multiplier for number of samples (relative to graph edges)"),
    absolute_samples: Optional[int] = typer.Option(None, help="Absolute number of samples (overrides multiplier if set)"),
    max_samples_per_node: int = typer.Option(20, help="Maximum samples to generate per source node"),
    use_gpu: bool = typer.Option(True, help="Use GPU if available"),
    force_regenerate: bool = typer.Option(False, help="Force regeneration even if samples exist"),
    fallback_random: bool = typer.Option(True, help="Fall back to random sampling if not enough hard negatives"),
    num_workers: int = typer.Option(None, help="Number of worker processes for multiprocessing"),
    node_sampling_fraction: float = typer.Option(0.2, help="Fraction of nodes to sample for hard negative generation")
):
    """
    Generate and save hard negative samples for the ogbn-arxiv dataset with customizable parameters.
    """
    print("Loading ogbn-arxiv dataset...")
    # Load graph with reverse edges for hard negative sampling
    graph = _load_graph(use_gpu=use_gpu, add_reverse=True)
    
    # Prepare output path
    save_path = _prepare_output_directory(output_dir, output_file)
    
    # Determine sample size
    num_samples = _determine_sample_size(graph, edge_multiplier, absolute_samples)
    
    # Generate hard negatives
    print(f"Generating {num_samples} hard negative samples...")
    src, dst = NodeEmbeddingEvaluator.optimized_hard_negative_sampling(
        graph=graph,
        num_samples=num_samples,
        fallback_random=fallback_random,
        save_path=save_path,
        load_path=save_path if not force_regenerate else None,
        force_regenerate=force_regenerate,
        max_samples_per_node=max_samples_per_node,
        num_workers=num_workers,
        node_sampling_fraction=node_sampling_fraction
    )
    
    print(f"Successfully generated {len(src)} hard negative samples")
    print(f"Saved to {save_path}")
    
    # Return the tensors in case they're needed for further processing
    return src, dst

@app.command()
def generate_uniform_negatives(
    output_dir: str = typer.Option("logs", help="Directory to save the generated samples"),
    output_file: str = typer.Option("arxiv_uniform_negatives.pt", help="Filename for the samples"),
    edge_multiplier: float = typer.Option(1.0, help="Multiplier for number of samples (relative to graph edges)"),
    absolute_samples: Optional[int] = typer.Option(None, help="Absolute number of samples (overrides multiplier if set)"),
    use_gpu: bool = typer.Option(True, help="Use GPU if available")
):
    """
    Generate and save uniformly distributed negative samples for the ogbn-arxiv dataset.
    """
    print("Loading ogbn-arxiv dataset...")
    # Load graph without reverse edges for uniform sampling
    graph = _load_graph(use_gpu=use_gpu, add_reverse=False)
    
    # Prepare output path
    save_path = _prepare_output_directory(output_dir, output_file)
    
    # Determine sample size
    num_samples = _determine_sample_size(graph, edge_multiplier, absolute_samples)
    
    # Generate uniform negative samples
    print(f"Generating {num_samples} uniform negative samples...")
    src, dst = NodeEmbeddingEvaluator.generate_uniform_negative_samples(
        graph=graph,
        num_samples=num_samples,
        save_path=save_path
    )
    
    print(f"Successfully generated {len(src)} uniform negative samples")
    print(f"Saved to {save_path}")
    
    # Return the tensors in case they're needed for further processing
    return src, dst

@app.command()
def evaluate_hard_negatives(
    embeddings_dir: str = typer.Option("embeddings", help="Directory containing embedding files"),
    hard_negatives_path: str = typer.Option("logs/arxiv_hard_negatives.pt", help="Path to pre-generated hard negatives"),
    batch_size: int = typer.Option(100000, help="Batch size for evaluation to avoid OOM"),
    use_gpu: bool = typer.Option(True, help="Use GPU if available"),
    file_pattern: str = typer.Option("*.pt", help="File pattern to match embedding files"),
    result_key: str = typer.Option("lp_hard/auc", help="Key to use for storing hard negative link prediction results")
):
    """
    Evaluate link prediction performance of all embeddings using pre-generated hard negatives.
    Results are added to results.json files in each embedding directory.
    """
    return _evaluate_embeddings(
        embeddings_dir=embeddings_dir,
        negatives_path=hard_negatives_path,
        batch_size=batch_size,
        use_gpu=use_gpu,
        file_pattern=file_pattern,
        result_key=result_key
    )

@app.command()
def evaluate_uniform_negatives(
    embeddings_dir: str = typer.Option("embeddings", help="Directory containing embedding files"),
    uniform_negatives_path: str = typer.Option("logs/arxiv_uniform_negatives.pt", help="Path to pre-generated uniform negatives"),
    batch_size: int = typer.Option(100000, help="Batch size for evaluation to avoid OOM"),
    use_gpu: bool = typer.Option(True, help="Use GPU if available"),
    file_pattern: str = typer.Option("*.pt", help="File pattern to match embedding files"),
    result_key: str = typer.Option("lp_uniform/auc", help="Key to use for storing uniform negative link prediction results")
):
    """
    Evaluate link prediction performance of all embeddings using pre-generated uniform negatives.
    Results are added to results.json files in each embedding directory.
    """
    return _evaluate_embeddings(
        embeddings_dir=embeddings_dir,
        negatives_path=uniform_negatives_path,
        batch_size=batch_size,
        use_gpu=use_gpu,
        file_pattern=file_pattern,
        result_key=result_key
    )

@app.command()
def evaluate_all_negatives(
    embeddings_dir: str = typer.Option("embeddings", help="Directory containing embedding files"),
    hard_negatives_path: str = typer.Option("logs/arxiv_hard_negatives.pt", help="Path to pre-generated hard negatives"),
    uniform_negatives_path: str = typer.Option("logs/arxiv_uniform_negatives.pt", help="Path to pre-generated uniform negatives"),
    batch_size: int = typer.Option(100000, help="Batch size for evaluation to avoid OOM"),
    use_gpu: bool = typer.Option(True, help="Use GPU if available"),
    file_pattern: str = typer.Option("*.pt", help="File pattern to match embedding files")
):
    """
    Evaluate link prediction performance using both hard and uniform negatives in one run.
    """
    print(f"Running evaluation with both hard and uniform negatives")
    
    hard_results = _evaluate_embeddings(
        embeddings_dir=embeddings_dir,
        negatives_path=hard_negatives_path,
        batch_size=batch_size,
        use_gpu=use_gpu,
        file_pattern=file_pattern,
        result_key="lp_hard/auc"
    )
    
    uniform_results = _evaluate_embeddings(
        embeddings_dir=embeddings_dir,
        negatives_path=uniform_negatives_path,
        batch_size=batch_size,
        use_gpu=use_gpu,
        file_pattern=file_pattern,
        result_key="lp_uniform/auc"
    )
    
    # Return combined results
    return {
        "hard_negatives": hard_results,
        "uniform_negatives": uniform_results
    }

if __name__ == "__main__":
    try:
        app()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
