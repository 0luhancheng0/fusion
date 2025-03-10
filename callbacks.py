import lightning as L
import numpy as np
import seaborn as sns
import torch

from dataloading import idx2cat
from utils import mask_to_index, nsample_per_class
import lovely_tensors as lt
lt.monkey_patch()

class LoggingCoefficients(L.Callback):
    def on_train_end(self, trainer, pl_module):
        trainer.logger.experiment.add_figure("coeffs/dist", pl_module.coeffs.data.plt.fig)
        trainer.logger.experiment.add_figure(
            "coeffs/heatmap",
            sns.heatmap(pl_module.coeffs.cpu().numpy(), cmap="YlGnBu").get_figure(),
        )


class LoggingGateScores(L.Callback):
    def on_train_end(self, trainer, pl_module):
        if hasattr(pl_module, "get_scores"):
            scores = pl_module.get_scores(
                *pl_module.down_project(
                    trainer.datamodule.features[0], trainer.datamodule.features[1]
                )
            )
            pl_module.logger.experiment.add_histogram("Gate Scores", scores)


class LoggingEmbeddingCallback(L.Callback):

    def __init__(self, max_nsample=2000):
        self.max_nsample = max_nsample

    def on_train_end(self, trainer, pl_module) -> None:
        self.log_node_embeddings(trainer, pl_module, "End")

    def log_node_embeddings(self, trainer, pl_module, tag):
        node_embeddings = torch.cat(
            trainer.predict(pl_module, trainer.datamodule.predict_dataloader()), dim=0
        )
        labels = trainer.datamodule.labels.cpu().detach().numpy()
        idx = nsample_per_class(self.max_nsample // 40, labels)
        categories = idx2cat(labels[idx])
        embeddings = node_embeddings[idx].detach().type(torch.float16)
        label_idx = {
            stage: mask_to_index(trainer.datamodule.graph.ndata[f"{stage}_mask"])
            .cpu()
            .detach()
            .numpy()
            for stage in ["train", "val", "test"]
        }
        split_labels = np.empty(
            len(trainer.datamodule.graph.ndata["train_mask"]), dtype=object
        )
        np.put_along_axis(split_labels, label_idx["train"], "train", axis=0)
        np.put_along_axis(split_labels, label_idx["val"], "valid", axis=0)
        np.put_along_axis(split_labels, label_idx["test"], "test", axis=0)
        metadata = list(zip(*[categories, split_labels]))
        metadata_header = ["category", "split"]
        pl_module.logger.experiment.add_embedding(
            embeddings,
            tag=f"Node Embeddings/{tag}",
            metadata=metadata,
            metadata_header=metadata_header,
        )
