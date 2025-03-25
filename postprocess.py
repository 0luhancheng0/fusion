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
from rich.progress import track
from rich import print

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


def _load_existing_results(results_path: Path) -> Dict:
    """Load existing results from a JSON file."""
    existing_results = {}
    if results_path.exists():
        with open(results_path, 'r') as f:
            existing_results = json.load(f)

    return existing_results

def _save_results(results_path: Path, results: Dict) -> None:
    """Save results to a JSON file."""
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

@app.command()
def evaluate_link_prediction(
    embeddings_dir: str = "/home/lcheng/oz318/fusion/logs",
    use_gpu: bool = True,
    file_pattern: str = "**/embeddings.pt",
) -> Dict:
    """Common evaluation function for both hard and uniform negative samples."""
    # Load graph
    graph = _load_graph(use_gpu=use_gpu, add_reverse=False)
    device = graph.device
    evaluator = OGBNArxivDataset().evaluator()
    embedding_dir = Path(embeddings_dir)
    embed_files = list(embedding_dir.rglob(file_pattern))
    for embed_file in track(embed_files, description="Evaluating embeddings"):
        
        # Get directory containing the embedding file
        embedding_parent = embed_file.parent
        results_path = embedding_parent / "results.json"
        
        # Load existing results if available
        existing_results = _load_existing_results(results_path)
        
        # Load embeddings
        embeddings = torch.load(embed_file, map_location=device)
        results_dict = evaluator.evaluate_link_prediction(embeddings)
        
        existing_results.update(results_dict)        

        _save_results(results_path, existing_results)

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
def collect_all_results(
    input_dir: str = typer.Option("logs", help="Directory containing results.json files"),
    output_dir: str = typer.Option(None, help="Output directory for the combined results (defaults to input_dir)"),
    output_file: str = typer.Option("all_results.json", help="Filename for the combined results"),
    include_empty: bool = typer.Option(False, help="Include directories without results.json files"),
):
    """
    Collect and combine all results.json files in the input directory and its subdirectories
    into a single all_results.json file. Results are organized by their relative paths.
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Input directory {input_dir} does not exist")
        return
    
    # If output_dir is not specified, use input_dir
    output_path = Path(output_dir) if output_dir else input_path
    output_path.mkdir(exist_ok=True, parents=True)
    save_path = output_path / output_file
    
    # Find all results.json files
    results_files = list(input_path.rglob("results.json"))
    if not results_files:
        print(f"No results.json files found in {input_dir} or subdirectories")
        return
    
    print(f"Found {len(results_files)} results.json files")
    
    # Combine results
    combined_results = {}
    
    for results_file in track(results_files, description="Processing results files"):
        try:
            # Get relative path to make it easier to identify
            rel_path = results_file.relative_to(input_path)
            parent_dir = str(rel_path.parent)
            
            # Load results
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            if not results and not include_empty:
                print(f"Skipping empty results file: {parent_dir}/results.json")
                continue
                
            # Use parent directory path as key, store results directly
            combined_results[parent_dir] = results
            
            print(f"Processed: {parent_dir}")
            
        except Exception as e:
            print(f"Error processing {results_file}: {e}")
    
    # Save combined results
    with open(save_path, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"Combined results saved to {save_path}")
    return combined_results

if __name__ == "__main__":
    try:
        app()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
