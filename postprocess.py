import os
import torch
import dgl
import typer
import json
from pathlib import Path
from typing import Optional, Dict
from evaluation import NodeEmbeddingEvaluator
from ogb.nodeproppred import DglNodePropPredDataset
from dataloading import OGBNArxivDataset
from dgl.nn.pytorch.link import EdgePredictor
from tqdm import tqdm
from torchmetrics.functional.classification import auroc

app = typer.Typer(help="Generate hard negative samples for ogbn-arxiv dataset")

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
    # Load ogbn-arxiv dataset using OGBNArxivDataset which already adds reverse edges
    graph = OGBNArxivDataset(add_reverse=True, load_metadata=False).graph
    
    # Put graph on GPU if available and requested
    device = torch.device('cuda' if (torch.cuda.is_available() and use_gpu) else 'cpu')
    graph = graph.to(device)
    print(f"Graph placed on {device}")
    
    print(f"Graph statistics: {graph.num_nodes()} nodes, {graph.num_edges()} edges")
    
    # Create output directory
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # Define save path for hard negatives
    save_path = out_dir / output_file
    
    # Set sample size based on parameters
    if absolute_samples is not None:
        num_samples = absolute_samples
        print(f"Using specified sample count: {num_samples}")
    else:
        num_samples = int(graph.num_edges() * edge_multiplier)
        print(f"Using {edge_multiplier}x the number of edges: {num_samples} samples")
    
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
def evaluate_hard_negatives(
    embeddings_dir: str = typer.Option("embeddings", help="Directory containing embedding files"),
    hard_negatives_path: str = typer.Option("logs/arxiv_hard_negatives.pt", help="Path to pre-generated hard negatives"),
    batch_size: int = typer.Option(100000, help="Batch size for evaluation to avoid OOM"),
    use_gpu: bool = typer.Option(True, help="Use GPU if available"),
    file_pattern: str = typer.Option("*.pt", help="File pattern to match embedding files"),
    result_key: str = typer.Option("lp_hard/auc", help="Key to use for storing hard negative link prediction results")
):
    """
    Evaluate link prediction performance of all embeddings in a directory and its subdirectories using pre-generated hard negatives.
    Results are added to results.json files in each embedding directory.
    """
    print(f"Evaluating embeddings from {embeddings_dir} and subdirectories using hard negatives from {hard_negatives_path}")
    
    # Load graph
    graph = OGBNArxivDataset(add_reverse=False, load_metadata=False).graph
    
    # Put graph on GPU if available and requested
    device = torch.device('cuda' if (torch.cuda.is_available() and use_gpu) else 'cpu')
    graph = graph.to(device)
    
    # Load hard negatives
    try:
        hard_neg_data = torch.load(hard_negatives_path)
        hard_neg_src = hard_neg_data['src'].to(device)
        hard_neg_dst = hard_neg_data['dst'].to(device)
        print(f"Loaded {len(hard_neg_src)} hard negative pairs")
    except Exception as e:
        print(f"Error loading hard negatives: {e}")
        return
    
    # Get real edges
    pos_src, pos_dst = graph.edges()
    
    # Create predictor
    predictor = EdgePredictor("cos")
    
    # Find all embedding files recursively
    embedding_dir = Path(embeddings_dir)
    embed_files = list(embedding_dir.rglob(file_pattern))
    
    if not embed_files:
        print(f"No embedding files found in {embeddings_dir} or subdirectories matching pattern {file_pattern}")
        return
    
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
        existing_results = {}
        if results_path.exists():
            try:
                with open(results_path, 'r') as f:
                    existing_results = json.load(f)
                print(f"Loaded existing results from {results_path}")
            except Exception as e:
                print(f"Error loading existing results: {e}")
        
        # Load embeddings
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
        except Exception as e:
            print(f"Error loading embeddings from {embed_file}: {e}")
            continue
        
        # Evaluate link prediction
        try:
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
            for i in range(0, len(hard_neg_src), batch_size):
                batch_neg_src = hard_neg_src[i:i+batch_size]
                batch_neg_dst = hard_neg_dst[i:i+batch_size]
                
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
            auc_score = auroc(scores, labels, task="binary").item()
            
            print(f"AUROC: {auc_score:.4f}")
            
            # Add result to existing results
            existing_results[result_key] = float(auc_score)
            overall_results[file_name] = auc_score
            
            # Save updated results
            with open(results_path, 'w') as f:
                json.dump(existing_results, f, indent=2)
            
            print(f"Updated results saved to {results_path}")
            
        except Exception as e:
            print(f"Error evaluating {embed_file}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n==== Evaluation Summary ====")
    for name, score in sorted(overall_results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: {score:.4f}")
    
    return overall_results

if __name__ == "__main__":
    try:
        app()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
