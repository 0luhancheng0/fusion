from torch_geometric.nn import Node2Vec
import torch
import lightning as L
from torch.utils.data import DataLoader

from torch.utils.data import TensorDataset
from constants import LABEL
from evaluation import NodeEmbeddingEvaluator

from dataloading import OGBNArxivDataset

import dgl.transforms as T
import dgl
from torchinfo import summary
from lightning_fabric.utilities.seed import seed_everything
import constants
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torchmetrics.functional.classification import auroc, roc, precision_recall_curve

from base import AbstractConfig, AbstractDriver
from typer import Typer
app = Typer(help="Node2Vec training script")

class Node2VecLightning(L.LightningModule):
    def __init__(
        self,
        batchsize=256,
        num_workers=4,
        lr=0.01,
        embedding_dim=128,
        walk_length=40,
        context_size=20,
        walks_per_node=10,
        num_negative_samples=1,
        p=1.0,
        q=1.0,
        sparse=True,
    ):
        super().__init__()

        self.dataset = OGBNArxivDataset()
        self.graph = self.dataset.graph
        # self.graph = graph

        self.batchsize = batchsize
        self.num_workers = num_workers
        self.labels = self.graph.ndata[LABEL]
        self.word2vec = Node2Vec(
            torch.stack(self.graph.edges("uv")),
            embedding_dim=embedding_dim,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            num_negative_samples=num_negative_samples,
            p=p,
            q=q,
            sparse=sparse,
        )

        self.lr = lr
        self.evaluator = NodeEmbeddingEvaluator(
            self.labels,
            {
                "train": self.graph.ndata["train_mask"].type(torch.bool),
                "val": self.graph.ndata["val_mask"].type(torch.bool),
                "test": self.graph.ndata["test_mask"].type(torch.bool),
            },
        )

    def configure_optimizers(self):
        return torch.optim.SparseAdam(self.word2vec.parameters(), lr=self.lr)

    def training_step(self, batch, _):
        loss = self.word2vec.loss(*batch)
        self.log(f"loss/train", loss)
        return loss

    def validation_step(self, batch, _):
        self.log(
            "acc/val",
            self.evaluator.evaluate_arxiv_embeddings(
                self.get_node_embeddings(), "val"
            ),
        )

    def test_step(self, batch, _):
        self.log(
            "acc/test",
            self.evaluator.evaluate_arxiv_embeddings(
                self.get_node_embeddings(), "test"
            ),
        )

    def get_node_embeddings(self):
        return self.word2vec().detach()

    def train_dataloader(self):
        return self.word2vec.loader(
            batch_size=self.batchsize, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return self._full_dataloader()

    def test_dataloader(self):
        return self._full_dataloader()

    def _full_dataloader(self):
        node_embeddings = self.get_node_embeddings()
        return DataLoader(
            TensorDataset(node_embeddings), batch_size=node_embeddings.shape[0]
        )


class Config(AbstractConfig):
    def __init__(
        self,
        max_epochs,
        batchsize,
        num_workers,
        lr,
        embedding_dim,
        walk_length,
        context_size,
        walks_per_node,
        num_negative_samples,
        p,
        q,
        sparse,
        seed,
        prefix
    ):
        super().__init__(seed, prefix)
        self.max_epochs = max_epochs
        self.batchsize = batchsize
        self.num_workers = num_workers
        self.lr = lr
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.num_negative_samples = num_negative_samples
        self.p = p
        self.q = q
        self.sparse = sparse



class Driver(AbstractDriver):
    def __init__(self, config):
        super().__init__(config)

    def fit(self):
        self.trainer.fit(self.model)

    def setup_model(self):
        return Node2VecLightning(
            batchsize=self.config.batchsize,
            num_workers=self.config.num_workers,
            lr=self.config.lr,
            embedding_dim=self.config.embedding_dim,
            walk_length=self.config.walk_length,
            context_size=self.config.context_size,
            walks_per_node=self.config.walks_per_node,
            num_negative_samples=self.config.num_negative_samples,
            p=self.config.p,
            q=self.config.q,
            sparse=self.config.sparse,
        )

    def validate(self):
        return self.trainer.validate(ckpt_path=self.best_model_path)[0]

    def test(self):
        return self.trainer.test(ckpt_path=self.best_model_path)[0]


    def get_node_embeddings(self):
        return self.best_model().get_node_embeddings()
    def setup_trainer(self, monitor="acc/val", mode="max"):
        return super().setup_trainer(monitor=monitor, mode=mode)
    def setup_datamodule(self):
        self.dataset = OGBNArxivDataset()
        self.graph = self.dataset.graph
    
@app.command(name="train")
def train(
    max_epochs: int = 5,
    batchsize: int = 256,
    num_workers: int = 4,
    lr: float = 0.01,
    embedding_dim: int = 128,
    walk_length: int = 40,
    context_size: int = 20,
    walks_per_node: int = 10,
    num_negative_samples: int = 1,
    p: float = 1.0,
    q: float = 1.0,
    sparse: bool = True,
    seed: int = 0,
):
    """Train a Node2Vec model with the specified parameters and print the results."""
    
    # Create config with CLI arguments
    config = Config(
        max_epochs=max_epochs,
        batchsize=batchsize,
        num_workers=num_workers,
        lr=lr,
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        num_negative_samples=num_negative_samples,
        p=p,
        q=q,
        sparse=sparse,
        seed=seed
    )
    driver = Driver(config)
    return driver.run()

@app.command(name="run-experiments")
def run_experiments():
    """Run experiments for embedding dimensions [32, 64, 128, 256] and seeds 0-4."""
    for embedding_dim in [32, 64, 128, 256]:
        for seed in range(5):
            print(f"\n=== Running experiment with embedding_dim={embedding_dim}, seed={seed} ===")
            config = Config(
                max_epochs=5,
                batchsize=256,
                num_workers=4,
                lr=0.01,
                embedding_dim=embedding_dim,
                walk_length=40,
                context_size=20,
                walks_per_node=10,
                num_negative_samples=1,
                p=1.0,
                q=1.0,
                sparse=True,
                seed=seed,
                prefix=str(embedding_dim),
            )
            driver = Driver(config)
            results = driver.run()
            print(f"Results for embedding_dim={embedding_dim}, seed={seed}:")
            for metric, value in results.items():
                print(f"  {metric}: {value}")
            print("========================================")
if __name__ == "__main__":
    app()
