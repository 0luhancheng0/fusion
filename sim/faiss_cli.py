#!/usr/bin/env python3

import typer
import numpy as np
from typing import Optional, List
from pathlib import Path
import sys
import os
from ..constants import SAVED_INDEX_DIR
# Add the parent directory to the path so we can import faiss_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import faiss_utils

app = typer.Typer(help="CLI for FAISS vector database operations")

# Ensure the index directory exists
def ensure_index_dir():
    Path(SAVED_INDEX_DIR).mkdir(parents=True, exist_ok=True)
    return Path(SAVED_INDEX_DIR)

# Get full path for an index name
def get_index_path(index_name):
    return ensure_index_dir() / f"{index_name}"

@app.command()
def create_index(
    index_name: str = typer.Argument(..., help="Name of the index (relative to the standard index directory)"),
    dim: int = typer.Argument(..., help="Dimensionality of the vectors"),
    index_type: str = typer.Option("Flat", help="Type of FAISS index (Flat, IVF, HNSW, etc.)"),
    metric: str = typer.Option("L2", help="Distance metric to use (L2, IP, etc.)"),
):
    """Create a new FAISS index and save it to the standard index directory."""
    output_path = get_index_path(index_name)
    typer.echo(f"Creating {index_type} index with dimension {dim} using {metric} metric")
    index = faiss_utils.create_index(dim=dim, index_type=index_type, metric=metric)
    faiss_utils.save_index(index, str(output_path))
    typer.echo(f"Index saved to {output_path} (in {SAVED_INDEX_DIR})")


@app.command()
def search(
    index_name: str = typer.Argument(..., help="Name of the index (relative to the standard index directory)"),
    query_vector: Path = typer.Argument(..., help="Path to query vector(s) file (.npy format)"),
    k: int = typer.Option(10, help="Number of nearest neighbors to return"),
    output_file: Optional[Path] = typer.Option(None, help="Path to save results (optional)"),
):
    """Search for nearest neighbors in the index."""
    index_path = get_index_path(index_name)
    typer.echo(f"Loading index from {index_path}")
    index = faiss_utils.load_index(str(index_path))
    
    typer.echo(f"Loading query vectors from {query_vector}")
    queries = np.load(str(query_vector))
    
    typer.echo(f"Searching for {k} nearest neighbors")
    distances, indices = faiss_utils.search_index(index, queries, k=k)
    
    if output_file:
        np.savez(str(output_file), distances=distances, indices=indices)
        typer.echo(f"Results saved to {output_file}")
    else:
        typer.echo("Search results:")
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            typer.echo(f"Query {i}:")
            for d, j in zip(dist, idx):
                typer.echo(f"  ID: {j}, Distance: {d}")


@app.command()
def info(
    index_name: str = typer.Argument(..., help="Name of the index (relative to the standard index directory)"),
):
    """Display information about a FAISS index."""
    index_path = get_index_path(index_name)
    typer.echo(f"Loading index from {index_path}")
    index = faiss_utils.load_index(str(index_path))
    index_info = faiss_utils.get_index_info(index)
    
    typer.echo("Index Information:")
    for key, value in index_info.items():
        typer.echo(f"{key}: {value}")

@app.command()
def list_indexes():
    """List all indexes in the standard index directory."""
    index_dir = ensure_index_dir()
    indexes = list(index_dir.glob("*"))
    
    if not indexes:
        typer.echo(f"No indexes found in {SAVED_INDEX_DIR}")
        return
    
    typer.echo(f"Indexes in {SAVED_INDEX_DIR}:")
    for index_path in indexes:
        typer.echo(f"  {index_path.name}")

if __name__ == "__main__":
    app()
