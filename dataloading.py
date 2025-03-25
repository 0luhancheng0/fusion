import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union
from evaluation import NodeEmbeddingEvaluator
import dgl
import dgl.data
import dgl.transforms as T
import lightning as L
import pandas as pd
import torch
from dgl.data import adapter
from ogb.linkproppred import DglLinkPropPredDataset
from ogb.nodeproppred import DglNodePropPredDataset
from torch.utils.data import DataLoader, TensorDataset
from constants import FEATURE, LABEL


class BaseDataModule(L.LightningDataModule, ABC):
    
    def __init__(
        self,
        feature_name_or_paths: List[Union[os.PathLike, str]] = ["feat"],
        # transforms: Dict = {"AddSelfLoop": {}, "AddReverse": {}},
        batchsize: Optional[int] = 256,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        # self.dataset_name_or_graph = dataset_name_or_graph
        # self.transforms = transforms
        self.feature_name_or_paths = [Path(i) for i in feature_name_or_paths]
        self.batchsize = batchsize
        self.device = device

        # Initialize attributes to be set in setup
        self.graph: Optional[dgl.DGLGraph] = None
        # self.features: List[torch.Tensor] = []
        # self.labels: Optional[torch.Tensor] = None
        # self.train_mask: Optional[torch.Tensor] = None
        # self.valid_mask: Optional[torch.Tensor] = None
        # self.test_mask: Optional[torch.Tensor] = None


    # def setup(self, stage: Optional[str] = None):
    #     """Set up the data for different stages."""
        self.graph = OGBNArxivDataset.load_graph()
        self.features = self.get_features(self.graph)
        self.labels = self.get_labels(self.graph)
        self.train_mask, self.valid_mask, self.test_mask = self.get_masks(self.graph)

    def get_graph(self) -> dgl.DGLGraph:
        """Load or retrieve the graph."""
        return self.graph

    def get_masks(
        self, graph: dgl.DGLGraph
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve train, validation, and test masks from the graph."""
        train_mask = graph.ndata["train_mask"].type(torch.bool)
        valid_mask = graph.ndata["val_mask"].type(torch.bool)
        test_mask = graph.ndata["test_mask"].type(torch.bool)
        return train_mask, valid_mask, test_mask

    def get_features(self, graph: dgl.DGLGraph) -> List[torch.Tensor]:
        """Load features from specified paths or graph data."""
        features = []

        for path in self.feature_name_or_paths:
            if path.exists():
                
                feat = torch.load(path, weights_only=False)
                if isinstance(feat, dict):
                    feat = feat["embeddings"]
                feat = feat.to(self.device)
            else:
                feat = graph.ndata[str(path)]
            features.append(feat)
        return features

    def get_labels(self, graph: dgl.DGLGraph) -> Optional[torch.Tensor]:
        """Retrieve labels from the graph."""
        return graph.ndata.get(LABEL, None)

    def apply_mask(self, mask: torch.Tensor) -> List[torch.Tensor]:
        """Apply a mask to the features."""
        return [feature[mask] for feature in self.features]

    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        """Abstract method to create the training dataloader."""
        pass

    @abstractmethod
    def val_dataloader(self) -> DataLoader:
        """Abstract method to create the validation dataloader."""
        pass

    @abstractmethod
    def test_dataloader(self) -> DataLoader:
        """Abstract method to create the test dataloader."""
        pass

    @abstractmethod
    def predict_dataloader(self):
        pass


class FeatureModule(BaseDataModule):
    """
    DataModule that handles feature-based datasets.
    """

    def train_dataloader(self) -> DataLoader:
        tensors = self.apply_mask(self.train_mask)
        if self.labels is not None:
            tensors.append(self.labels[self.train_mask])
        return DataLoader(TensorDataset(*tensors), batch_size=self.batchsize)

    def val_dataloader(self) -> DataLoader:
        tensors = self.apply_mask(self.valid_mask)
        if self.labels is not None:
            tensors.append(self.labels[self.valid_mask])
        return DataLoader(
            TensorDataset(*tensors),
            batch_size=self.batchsize or self.valid_mask.sum().item(),
        )

    def test_dataloader(self) -> DataLoader:
        tensors = self.apply_mask(self.test_mask)
        if self.labels is not None:
            tensors.append(self.labels[self.test_mask])
        return DataLoader(
            TensorDataset(*tensors),
            batch_size=self.batchsize or self.test_mask.sum().item(),
        )

    def predict_dataloader(self):
        return DataLoader(
            TensorDataset(*self.features),
            batch_size=self.batchsize or self.test_mask.sum().item(),
        )


import constants


class GraphDataModule(L.LightningDataModule):
    """Lightning DataModule for graph data."""

    def __init__(self, graph, features):
        super().__init__()
        self.graph = graph
        self.features = torch.load(features, weights_only=False)
        if isinstance(self.features, dict):
            self.features = self.features["embeddings"]

            
            

    def train_dataloader(self):
        return DataLoader(
            [(self.graph, self.features)], batch_size=1, collate_fn=lambda x: x
        )

    def val_dataloader(self):
        return DataLoader(
            [(self.graph, self.features)], batch_size=1, collate_fn=lambda x: x
        )

    def test_dataloader(self):
        return DataLoader(
            [(self.graph, self.features)], batch_size=1, collate_fn=lambda x: x
        )



class StandardizeNodeFeatures(dgl.transforms.BaseTransform):
    def __init__(self, eps=1e-6, keys=[FEATURE]):
        super().__init__()
        self.eps = eps
        self.keys = keys

    def __call__(self, g):
        for key in self.keys:
            mean = g.ndata[key].mean()
            variance = g.ndata[key].var()
            g.ndata[key] = (g.ndata[key] - mean) / (variance + self.eps).sqrt()
        return g


class EnsureDevice(dgl.transforms.BaseTransform):
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = device

    def __call__(self, g):
        return g.to(self.device)


class MaskToBool(dgl.transforms.BaseTransform):
    def __call__(self, g):
        for key in g.ndata.keys():
            if key.endswith("_mask"):
                g.ndata[key] = g.ndata[key].bool()
        return g


class LaplacianPositionalEncoding(dgl.transforms.BaseTransform):
    def __init__(self, k=2, pos_key="pos_embedding"):
        super().__init__()
        self.pos_key = pos_key
        self.k = k

    def __call__(self, g):
        pos_embedding = dgl.lap_pe(g, k=self.k)
        g.ndata[self.pos_key] = pos_embedding
        return g


class NodePropPredDataset(DglNodePropPredDataset):
    def __init__(self, name, root="dataset", meta_dict=None):
        super().__init__(name, root, meta_dict)

    def __getitem__(self, idx):
        return self.graph[idx]


class OGBDataset:
    def __init__(
        self,
        name: str,
        task: Literal["NodeClassification", "LinkPrediction"],
        root="/home/lcheng/oz318/datasets/ogb",
        transforms=[MaskToBool(), EnsureDevice()],
        nfeatures={},
        standardize=True,
        **kwargs,
    ):
        self.name = name
        self.task = task
        if self.name.startswith("ogbn"):
            self.dataset_loader = NodePropPredDataset
        else:
            assert self.name.startswith("ogbl")
            self.dataset_loader = DglLinkPropPredDataset

        if self.task == "NodeClassification":
            self.adapter = adapter.AsNodePredDataset
        else:
            assert self.task == "LinkPrediction"
            self.adapter = adapter.AsLinkPredDataset

        self.dataset = self.dataset_loader(name=name, root=root)
        self.dataset = self.adapter(self.dataset, **kwargs)
        self.transforms = transforms
        self.nfeatures = nfeatures
        self.standardize = standardize
        self.kwargs = kwargs
        self.graph = self.apply_transforms(self.dataset[0])

    def apply_transforms(self, graph):
        if self.standardize:
            self.transforms.append(
                StandardizeNodeFeatures(keys=list(self.nfeatures.keys()))
            )
        trans = T.Compose(self.transforms)
        return trans(graph)


class OGBNArxivDataset(OGBDataset):

    def __init__(
        self,
        root="/home/lcheng/oz318/datasets/ogb",
        add_reverse=True,
        add_self_loop=False,
        standardize=True,
        load_metadata=True,
    ):
        transforms = [
            MaskToBool(),
            EnsureDevice(),
        ]
        if add_reverse:
            transforms.append(T.AddReverse())
        if add_self_loop:
            transforms.append(T.AddSelfLoop())
        super().__init__(
            name="ogbn-arxiv",
            task="NodeClassification",
            root=root,
            transforms=transforms,
            nfeatures={"feat": "feat"},
            standardize=standardize,
        )
        self.dataset_root = Path(root)
        if load_metadata:
            self._load_category_mappings()
            self._load_paper_metadata()

    def _load_category_mappings(self):
        """Load ArXiv category mappings from files"""
        arxiv_dataset_path = self.dataset_root / "ogbn_arxiv"
        self.paper_labels = pd.read_csv(
            arxiv_dataset_path / "raw" / "node-label.csv.gz", header=None
        ).values
        self.idx2cat = pd.read_csv(
            arxiv_dataset_path / "mapping" / "labelidx2arxivcategeory.csv.gz",
            index_col="label idx",
        )
        self.node_to_paper_mapping = pd.read_csv(
            arxiv_dataset_path / "mapping" / "nodeidx2paperid.csv.gz",
            index_col="node idx",
        )
        self.paper_to_node_mapping = self.node_to_paper_mapping.reset_index().set_index(
            "paper id"
        )

    def _load_paper_metadata(self):
        """Load paper title and abstract information"""
        self.paper_metadata = pd.read_csv(
            self.dataset_root / "ogbn_arxiv" / "titleabs.tsv",
            sep="\t",
            index_col=0,
            names=["paper id", "title", "abstract"],
        )
        self.titleabs = (
            "Title: "
            + self.paper_metadata.title
            + "\n"
            + "Abstract: "
            + self.paper_metadata.abstract
        )
        self.paper_to_embedding_mapping = (
            pd.Series(self.paper_metadata.index)
            .reset_index()
            .set_index("paper id")
            .rename({"index": "embedding index"}, axis=1)
        )
        self.node_to_embedding_mapping = self.node_to_paper_mapping.join(
            self.paper_to_embedding_mapping, on="paper id"
        )["embedding index"]

    def idx2cat(self, idx: list[int]):
        """Convert label indices to category names"""
        return self.idx2cat.loc[idx].values.flatten()

    @staticmethod
    def load_graph():
        return OGBNArxivDataset().graph

    @staticmethod
    def evaluator():
        from constants import HARD_NEGATIVE_SAMPLES, UNIFORM_NEGATIVE_SAMPLES
        graph = OGBNArxivDataset.load_graph()
        return NodeEmbeddingEvaluator(
            labels=graph.ndata["label"],
            masks={
                "train": graph.ndata["train_mask"].type(torch.bool),
                "valid": graph.ndata["val_mask"].type(torch.bool),
                "test": graph.ndata["test_mask"].type(torch.bool),
            },
            graph=graph,  # Add the graph parameter explicitly
            hard_negative_path=HARD_NEGATIVE_SAMPLES,
            uniform_negative_path=UNIFORM_NEGATIVE_SAMPLES,
        )


def idx2cat(idx: list[int]):
    """Convert label indices to category names"""
    path = (
        Path("/home/lcheng/oz318/datasets/ogb/")
        / "ogbn_arxiv"
        / "mapping"
        / "labelidx2arxivcategeory.csv.gz"
    )
    catmapping = pd.read_csv(path, index_col="label idx")
    return catmapping.loc[idx].values.flatten()

