import torch
import dgl
import lightning as L
from einops import einsum
from cupy.linalg import lstsq
import cupy as cp
from dataloading import OGBNArxivDataset
from dataloading import GraphDataModule
from dataloading import GraphDataModule
from torch import nn
import dgl.function as fn
from jaxtyping import Float
from typing import Literal, List, Optional
from torch import Tensor
from ray import tune
from torchmetrics.functional.classification import multiclass_accuracy
from base import AbstractDriver, AbstractConfig
from callbacks import LoggingCoefficients


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


class ASGC(L.LightningModule):
    """Adaptive Spectral Graph Convolution (ASGC) model."""

    def __init__(
        self, d_feat=128, k=8, reg=5, lr=0.01, weight_decay=5e-4, num_classes=40
    ):
        super().__init__()
        self.save_hyperparameters()
        self.k = k
        self.d_feat = d_feat
        self.reg = reg
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.sgc = SGConv(k=k, cached=True)
        self.coeffs = nn.Parameter(
            torch.zeros((self.k + 1, self.d_feat)), requires_grad=False
        )
        # self.coeffs = None
        self.cache = None
        self.R = nn.Parameter(torch.zeros((self.k + 1)), requires_grad=False)
        self.R[0] = self.reg
        self.fc = nn.Linear(d_feat, num_classes)

    def compute_coefficients(self, cache: Float[Tensor, "k n d"]):

        for feat_idx in range(self.d_feat):
            A = torch.vstack([cache[..., feat_idx].T, self.R])
            b = torch.vstack(
                [
                    cache[0, :, feat_idx].unsqueeze(-1),
                    torch.tensor([[0.0]], device=self.device),
                ]
            )
            x, _, _, _ = lstsq(cp.asarray(A), cp.asarray(b))
            self.coeffs[:, feat_idx] = torch.as_tensor(x.flatten(), device=self.device)

        return self.coeffs

    def forward(self, graph, feats):
        """Forward pass to generate adaptive spectral graph convolution embeddings."""

        if self.cache is None:
            self.cache = torch.stack(self.sgc(graph, feats)).to(self.device)
        if self.coeffs.allclose(torch.zeros_like(self.coeffs)):
            self.compute_coefficients(self.cache)
        embeddings = einsum(self.cache, self.coeffs, "k n d, k d -> n d")
        logits = self.fc(embeddings)
        return logits, embeddings

    def step(self, graph, feats, stage: Literal["train", "val", "test"]):
        logits, embeddings = self(graph, feats)

        masks = graph.ndata[f"{stage}_mask"]
        labels = graph.ndata["label"]
        loss = torch.nn.functional.cross_entropy(logits[masks], labels[masks])
        acc = multiclass_accuracy(
            logits[masks], labels[masks], num_classes=self.num_classes
        )

        self.log(f"loss/{stage}", loss, prog_bar=True)
        self.log(f"acc/{stage}", acc, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        graph, feats = batch[0]
        return self.step(graph, feats, stage="train")

    def validation_step(self, batch, batch_idx):
        graph, feats = batch[0]
        return self.step(graph, feats, stage="val")

    def test_step(self, batch, batch_idx):
        graph, feats = batch[0]
        return self.step(graph, feats, stage="test")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


class Config(AbstractConfig):
    def __init__(self, input_features, max_epochs, k, reg, lr, weight_decay, seed, prefix):
        super().__init__(seed, prefix)
        self.input_features = input_features
        self.max_epochs = max_epochs
        self.k = k
        self.reg = reg
        self.lr = lr
        self.weight_decay = weight_decay



class Driver(AbstractDriver):

    def setup_datamodule(self):
        dataset = OGBNArxivDataset()
        self.graph = dataset.graph
        return GraphDataModule(
            self.graph, self.config.input_features,
        )

    def setup_model(self):
        return ASGC(
            d_feat=self.datamodule.features.shape[1],
            k=self.config.k,
            reg=self.config.reg,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            num_classes=40,
        )

    def setup_trainer(self):
        return super().setup_trainer(
            callbacks=[LoggingCoefficients()], monitor="acc/val", mode="max"
        )

    def get_node_embeddings(self, model=None):
        return einsum(self.model.cache, self.model.coeffs.cuda(), "k n d, k d -> n d")

    # def evaluate(self):
    #     return {
    #         **self.validate(),
    #         **self.test(),
    #         "lp/auc": self.evaluator.evaluate_link_prediction(
    #             self.graph, self.get_node_embeddings()
    #         ),
    #     }

from pprint import pprint

@app.command()
def run_experiments():
    for dim in [32, 64, 128, 256]:
        for reg in [0.01, 0.1, 1.0, 10, 100]:
            for k in [1, 2, 4, 8]: 
                config = Config(
                    input_features=f"/fred/oz318/luhanc/fusion/saved_embeddings/ogbn-arxiv/relational/node2vec/{dim}.pt",
                    max_epochs=5,
                    k=k,
                    reg=reg,
                    lr=0.01,
                    weight_decay=5e-4,
                    seed=k,
                    prefix=f"{dim}/{reg}",
                )
                driver = Driver(config)
                driver.run()
                torch.save(driver.get_node_embeddings(),f"/home/lcheng/oz318/fusion/saved_embeddings/ogbn-arxiv/relational/asgc/{dim}.pt")
                pprint(f"Results for dim={dim}, reg={reg}:")
                pprint(driver.results)
run_experiments()
# if __name__ == "__main__":
#     app()