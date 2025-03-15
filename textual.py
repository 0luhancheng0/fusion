import os
import torch
import typer
from pathlib import Path
from base import ConfigBase, DriverBase
from dataloading import OGBNArxivDataset
from evaluation import NodeEmbeddingEvaluator
import json
from typing_extensions import Annotated

app = typer.Typer(help="Textual Embeddings Evaluation Script")

class TextualConfig(ConfigBase):
    """Configuration for the textual embeddings evaluation."""
    
    def __init__(
        self,
        model_name: str,
        embedding_path: str,
        embedding_dim: int,
        seed: str = "0",
        prefix: str = None
    ):
        # If no prefix is provided, create one using the model name
        if prefix is None:
            prefix = model_name
            
        super().__init__(seed, prefix)
        self.model_name = model_name
        self.embedding_path = embedding_path
        self.embedding_dim = embedding_dim
        # No max_epochs needed since we don't train


class TextualDriver(DriverBase):
    """Driver for textual embeddings evaluation."""
    
    def __init__(self, config: TextualConfig):
        super().__init__(config)
    
    def setup_trainer(self, callbacks=[], monitor="loss/val", mode="min", **kwargs):
        """Override to do nothing since we don't need a trainer."""
        return None
    
    def setup_model(self):
        """Not using a model to train - just loading embeddings."""
        return None
    
    def setup_datamodule(self):
        """Just need the graph, not an actual datamodule."""
        self.dataset = OGBNArxivDataset()
        self.graph = self.dataset.graph
        return None
    
    def get_node_embeddings(self):
        """Load embeddings from the specified path."""
        if not hasattr(self, "embeddings"):
            embeddings_path = Path(self.config.embedding_path)
            if not embeddings_path.exists():
                raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
            
            print(f"Loading embeddings from {embeddings_path}")
            
            # Use weights_only=True for security and handle the warning
            embeddings = torch.load(embeddings_path, weights_only=True)
            
            # Handle different embedding formats
            if isinstance(embeddings, dict):
                if 'embeddings' in embeddings:
                    embeddings = embeddings['embeddings']
            
            # Ensure embeddings are on CPU for consistent device handling
            self.embeddings = embeddings.cpu()
            print(f"Embeddings loaded with shape {self.embeddings.shape}, memory: {self.embeddings.element_size() * self.embeddings.nelement() / (1024 * 1024):.2f} MB")
            
        return self.embeddings
    
    def run(self):
        """Run the evaluation and save results."""
        try:
            # Get embeddings
            embeddings = self.get_node_embeddings()
            
            # Set up evaluator with CPU tensors
            self.evaluator = NodeEmbeddingEvaluator(
                self.graph.ndata["label"].cpu(),
                {
                    "train": self.graph.ndata["train_mask"].cpu(),
                    "valid": self.graph.ndata["val_mask"].cpu(),
                    "test": self.graph.ndata["test_mask"].cpu()
                }
            )
            
            # Ensure graph is on CPU for link prediction
            cpu_graph = self.graph.to('cpu') if self.graph.device != torch.device('cpu') else self.graph
            
            # Evaluate embeddings
            results = {}
            
            # Node classification tasks
            print("Evaluating node classification (validation)...")
            results["acc/valid"] = self.evaluator.evaluate_arxiv_embeddings(embeddings, split="valid")
            print(f"Validation accuracy: {results['acc/valid']:.4f}")
            
            print("Evaluating node classification (test)...")
            results["acc/test"] = self.evaluator.evaluate_arxiv_embeddings(embeddings, split="test")
            print(f"Test accuracy: {results['acc/test']:.4f}")
            
            # Link prediction task
            print("Evaluating link prediction...")
            results["lp/auc"] = self.evaluator.evaluate_link_prediction(cpu_graph, embeddings)
            print(f"Link prediction AUC: {results['lp/auc']:.4f}")
            
            print(f"All evaluation complete: {results}")
            
            # Save results
            save_dir = Path("/home/lcheng/oz318/fusion/logs/TextualEmbeddings") / self.config.model_name
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save results
            with open(save_dir / 'results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            # Save config
            with open(save_dir / 'config.json', 'w') as f:
                json.dump(self.config.__dict__, f, indent=2)
            
            # Save a symlink to original embeddings
            embeddings_link = save_dir / 'embeddings_link'
            if not embeddings_link.exists():
                try:
                    os.symlink(os.path.abspath(self.config.embedding_path), embeddings_link)
                except Exception as e:
                    print(f"Could not create symlink: {e}")
            
            self.results = results
            return self
        
        finally:
            # Clean up memory even if there's an exception
            if hasattr(self, 'embeddings'):
                print("Cleaning up embeddings from memory...")
                del self.embeddings
                torch.cuda.empty_cache()  # Clear CUDA cache if using CUDA
    
    def fit(self):
        """No training required for evaluation."""
        pass
    
    def validate(self):
        """No validation set."""
        pass
    
    def test(self):
        """No separate test method."""
        pass


@app.command(name="evaluate")
def evaluate_embedding(
    embedding_path: Annotated[str, typer.Argument(help="Path to the embedding file")],
    model_name: Annotated[str, typer.Option(help="Name of the embedding model (optional, will be inferred from path)")] = None,
    embedding_dim: Annotated[int, typer.Option(help="Dimension of the embeddings (optional, will be inferred from path)")] = None
):
    """Evaluate a single textual embedding file."""
    
    # Parse the path
    embedding_path = Path(embedding_path)
    
    # If model_name is not provided, infer it from the path
    if model_name is None:
        # Extract model name from path, assuming path structure like /path/to/model_name/dim.pt
        model_name = embedding_path.parent.name
        print(f"Inferred model name: {model_name}")
    
    # If embedding_dim is not provided, infer it from the filename
    if embedding_dim is None:
        # Extract dimension from filename (assuming filename is dimension.pt)
        try:
            embedding_dim = int(embedding_path.stem)
            print(f"Inferred embedding dimension: {embedding_dim}")
        except ValueError:
            # If we can't parse the dimension, try to load the embedding to get its size
            print("Could not infer embedding dimension from filename. Loading embedding to determine size...")
            temp_embedding = torch.load(embedding_path, weights_only=True)
            if isinstance(temp_embedding, dict) and 'embeddings' in temp_embedding:
                temp_embedding = temp_embedding['embeddings']
            embedding_dim = temp_embedding.shape[1]
            print(f"Determined embedding dimension from file: {embedding_dim}")
            # Free memory
            del temp_embedding
            torch.cuda.empty_cache()
    
    # Create config
    config = TextualConfig(
        model_name=model_name,
        embedding_path=str(embedding_path),
        embedding_dim=embedding_dim,
    )
    
    # Run evaluation
    driver = TextualDriver(config)
    driver.run()
    
    return driver.results


@app.command(name="evaluate-all")
def evaluate_all():
    """Evaluate all textual embeddings in the standard directory."""
    base_dir = Path("/home/lcheng/oz318/fusion/saved_embeddings/ogbn-arxiv/textual")
    results = {}
    
    # Find all model directories
    for model_dir in base_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        model_name = model_dir.name
        print(f"\n=== Evaluating {model_name} embeddings ===")
        
        model_results = {}
        
        # Process one model's embeddings at a time
        for emb_file in model_dir.glob("*.pt"):
            embedding_dim = int(emb_file.stem)  # Assuming filename is the dimension
            print(f"Processing {model_name} with dimension {embedding_dim}")
            
            try:
                config = TextualConfig(
                    model_name=model_name,
                    embedding_path=str(emb_file),
                    embedding_dim=embedding_dim,
                )
                
                # Create a fresh driver for each embedding
                driver = TextualDriver(config)
                
                # Run evaluation and store results - make sure we get the driver back
                driver = driver.run()
                
                # Save the results if they exist
                if hasattr(driver, 'results') and driver.results:
                    model_results[embedding_dim] = driver.results
                else:
                    print(f"Warning: No results returned for {model_name} with dimension {embedding_dim}")
                
                # Explicitly clean up to free memory
                if hasattr(driver, 'embeddings'):
                    del driver.embeddings
                del driver
                torch.cuda.empty_cache()  # Clear CUDA cache if using CUDA
                
                print(f"Successfully evaluated {model_name} with dimension {embedding_dim}")
                
            except Exception as e:
                print(f"Error processing {emb_file}: {e}")
                import traceback
                traceback.print_exc()
        
        # Store all results for this model
        results[model_name] = model_results
        print(f"Completed evaluation of {model_name}")
    
    # Save overall results to a single file
    results_dir = Path("/home/lcheng/oz318/fusion/logs/TextualEmbeddings")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / 'all_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"All results saved to {results_dir / 'all_results.json'}")
    return results


if __name__ == "__main__":
    app()
