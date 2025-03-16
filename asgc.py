import torch
import dgl
from einops import einsum
from cupy.linalg import lstsq
import cupy as cp
from dataloading import OGBNArxivDataset
import dgl.function as fn
from jaxtyping import Float
from torch import Tensor
from base import DriverBase, ConfigBase

import typer
app = typer.Typer()

class SGConv:
    def __init__(
        self,
        k=1,
        cached=False,
        norm=None,
    ):
        self._cached = cached
        self._k = k
        self.norm = norm
        self._cache = []

    def __call__(self, graph, feat, edge_weight=None):
        with graph.local_scope():
            msg_func = fn.copy_u("h", "m")
            if edge_weight is not None:
                graph.edata["_edge_weight"] = dgl.EdgeWeightNorm("both")(
                    graph, edge_weight
                )
                msg_func = fn.u_mul_e("h", "_edge_weight", "m")

            # Initialize list to store features at each step
            all_features = [feat]  # Start with the input features (0-th step)
            current_feat = feat

            if (
                hasattr(self, "_cached_features")
                and self._cached
                and self._cached_features is not None
            ):
                return self._cached_features[
                    -1
                ]  # Return only last feature for compatibility
            else:
                if edge_weight is None:
                    # compute normalization
                    degs = graph.in_degrees().to(feat).clamp(min=1)
                    norm = torch.pow(degs, -0.5)
                    norm = norm.to(feat.device).unsqueeze(1)

                # compute (D^-1 A^k D)^k X
                for i in range(self._k):
                    if edge_weight is None:
                        current_feat = current_feat * norm
                    graph.ndata["h"] = current_feat
                    graph.update_all(msg_func, fn.sum("m", "h"))
                    current_feat = graph.ndata.pop("h")
                    if edge_weight is None:
                        current_feat = current_feat * norm

                    if self.norm is not None:
                        current_feat = self.norm(current_feat)
                    all_features.append(current_feat)
                if self._cached:
                    self._cached_features = all_features

            return all_features  # Return only the last feature for compatibility


class ASGC:
    """Adaptive Spectral Graph Convolution (ASGC) model."""

    def __init__(
        self, input_features, k=8, reg=5, lr=0.01, weight_decay=5e-4, num_classes=40, device="cuda:0"
    ):
        self.device = device
        self.input_features = input_features.to(self.device)
        self.k = k

        self.reg = reg
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.sgc = SGConv(k=k, cached=True)
        
        self.cache = None
        self.coefficients = torch.zeros((self.k + 1, self.input_features.shape[-1]), device=self.device)
        self.R = torch.zeros((self.k + 1), device=self.device)
        self.R[0] = self.reg

        self.dataset = OGBNArxivDataset()
        
        self.graph = self.dataset.graph.to(self.device)
        # self.masks = {"train": self.graph.ndata["train_mask"], "valid": self.graph.ndata["val_mask"], "test": self.graph.ndata["test_mask"]}
        # self.evaluator = NodeEmbeddingEvaluator(self.graph.ndata["label"], self.masks)
        self.evaluator = OGBNArxivDataset.evaluator()
        self.embeddings = None
        
    def compute_coefficients(self, cache: Float[Tensor, "k n d"]):

        for feat_idx in range(self.input_features.shape[-1]):
            A = torch.vstack([cache[..., feat_idx].T, self.R])
            b = torch.vstack(
                [
                    cache[0, :, feat_idx].unsqueeze(-1),
                    torch.tensor([[0.0]], device=self.device),
                ]
            )
            x, _, _, _ = lstsq(cp.asarray(A), cp.asarray(b))
            self.coefficients[:, feat_idx] = torch.as_tensor(x.flatten(), device=self.device)

    def run(self):
        """Forward pass to generate adaptive spectral graph convolution embeddings."""
        cache = torch.stack(self.sgc(self.graph, self.input_features)).to(self.device)
        self.compute_coefficients(cache)
        print(cache.shape, self.coefficients.shape)
        self.embeddings = einsum(cache, self.coefficients.to(cache.device), "k n d, k d -> n d")
        return self.coefficients, self.embeddings
        

class Config(ConfigBase):
    def __init__(self, input_features: str, max_epochs: int, k: int, reg: float, lr: float, weight_decay: float, prefix: str):
        super().__init__("0", prefix)
        self.input_features = input_features
        self.max_epochs = max_epochs
        self.k = k
        self.reg = reg
        self.lr = lr
        self.weight_decay = weight_decay


class ASGCDriver(DriverBase):
    def __init__(self, config: Config):

        super().__init__(config)
    
    def setup_model(self):
        input_features = torch.load(self.config.input_features)
        return ASGC(
            input_features=input_features,
            k=self.config.k,
            reg=self.config.reg,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
    
    def setup_datamodule(self):
        # ASGCDriver doesn't actually use the datamodule
        return None
    
    def get_node_embeddings(self):
        # # Returns the node embeddings from the model
        # if hasattr(self.model, 'embeddings'):
        #     return self.model.embeddings
        # # If embeddings aren't cached, compute them
        # _, embeddings = self.model.run()
        return self.model.embeddings
    
    def run(self):
        # Override the run method to implement custom behavior
        import os
        import json
        
        # Determine the save directory path
        save_dir = os.path.join('/home/lcheng/oz318/fusion/logs/ASGC', 
                              f"{self.model.input_features.shape[-1]}", 
                              f"{self.config.reg}",
                              f"{self.config.k}")
        
        # Check if results already exist
        results_path = os.path.join(save_dir, 'results.json')
        if os.path.exists(results_path):
            print(f"Results already exist at {results_path}, skipping this trial.")
            # Load and return existing results
            with open(results_path, 'r') as f:
                results = json.load(f)
            return results
        
        # Create the directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Run model
        coefficients, embeddings = self.model.run()
        
        # Evaluate results using the appropriate evaluator
        results = {}
        # Use validation set by default
        results["acc/valid"] = self.model.evaluator.evaluate_arxiv_embeddings(embeddings, split="valid")
        results["acc/test"] = self.model.evaluator.evaluate_arxiv_embeddings(embeddings, split="test")
        
        # Add link prediction results
        results["lp/auc"] = self.model.evaluator.evaluate_link_prediction(
            self.model.graph, embeddings
        )
        
        # Save results
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save config
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        # Save coefficients
        torch.save(coefficients, os.path.join(save_dir, 'coefficients.pt'))
        
        # Save embeddings
        torch.save(embeddings, os.path.join(save_dir, 'embeddings.pt'))
        
        return results


@app.command()
def run_experiments():
    from tqdm import tqdm
    
    # Calculate total number of experiments
    dims = [32, 64, 128, 256]
    regs = [0.01, 0.1, 1, 10, 100, 1e3, 1e4]
    ks = [1, 2, 4, 8, 16, 32, 64]
    total_experiments = len(dims) * len(regs) * len(ks)
    
    # Create progress bar
    pbar = tqdm(total=total_experiments, desc="Running experiments")
    
    for dim in dims:
        for reg in regs:
            for k in ks: 
                # Call the single trial function
                result = run(dim=dim, reg=reg, k=k, lr=0.01, weight_decay=5e-4)
                pbar.update(1)
                pbar.set_description(f"Dim: {dim}, Reg: {reg}, K: {k}, Acc: {result.get('acc/valid', 0):.4f}")
    
    pbar.close()
    print("All experiments completed!")
from typing_extensions import Annotated
@app.command()
def run(
    dim: Annotated[int, typer.Argument(help="Feature dimension")],
    reg: Annotated[float, typer.Argument(help="Regularization parameter")],
    k: Annotated[int, typer.Argument(help="Number of hops")],
    lr: Annotated[float, typer.Argument(help="Learning rate")],
    weight_decay: Annotated[float, typer.Argument(help="Weight decay")]
):
    """Run a single trial with specified parameters."""
    config = Config(
        input_features=f"/fred/oz318/luhanc/fusion/saved_embeddings/ogbn-arxiv/relational/node2vec/{dim}.pt",
        max_epochs=100,
        k=k,
        reg=reg,
        lr=lr,
        weight_decay=weight_decay,
        prefix=f"{dim}/{reg}/{k}"
    )

    driver = ASGCDriver(config)
    result = driver.run()
    print(f"Completed single trial with dim={dim}, reg={reg}, k={k}")
    print(f"Results: {result}")
    
    return result


if __name__ == "__main__":
    app()