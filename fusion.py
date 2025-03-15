from abc import ABC
from pathlib import Path
import lightning as L
import torch
import torchmetrics.functional as F
from einops import einsum
from jaxtyping import Float
from torch import Tensor, nn
from base import ConfigBase, DriverBase
from dataloading import FeatureModule
from callbacks import LoggingEmbeddingCallback, LoggingGateScores
from typing import Dict
from typer import Typer
from itertools import product
import constants
from tqdm import tqdm
from torchinfo import summary


from constants import SAVED_EMBEDDINGS_DIR
from itertools import product


app = Typer()

NUM_CLASSES = 40


class FusionClassification(L.LightningModule, ABC):
    def __init__(self, textual_dim, relational_dim, latent_dim, lr, weight_decay=1e-4):
        super().__init__()
        self.textual_dim = textual_dim
        self.relational_dim = relational_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.textual_down_projection = nn.Linear(textual_dim, self.latent_dim)
        self.relational_down_projection = nn.Linear(relational_dim, self.latent_dim)
        self.decision_layer = nn.Linear(self.latent_dim, NUM_CLASSES)
        self.loss = nn.CrossEntropyLoss()
        # self.save_hyperparameters()  # Save hyperparameters for Ray Tune

    def down_project(self, textual, relational):
        return self.textual_down_projection(textual), self.relational_down_projection(
            relational
        )

    def encode(self, textual, relational):
        fused = self.fusion(*self.down_project(textual, relational))
        return fused

    def forward(self, textual, relational):
        return self.decision_layer(self.encode(textual, relational))

    def aux_loss(self, textual, relational):
        return 0

    def step(self, batch, stage):
        textual, relational, labels = batch
        logits = self(textual, relational)
        loss = self.loss(logits, labels)
        self.log(f"loss/{stage}", loss)
        self.log(
            f"acc/{stage}",
            F.accuracy(
                logits.argmax(dim=1), labels, task="multiclass", num_classes=NUM_CLASSES
            ),
            prog_bar=True,
        )
        return loss + self.aux_loss(textual, relational)

    def training_step(self, batch, _):
        return self.step(batch, "train")

    def validation_step(self, batch, _):
        return self.step(batch, "val")

    def test_step(self, batch, _):
        return self.step(batch, "test")

    def predict_step(self, batch, _):
        return self.encode(*batch)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
                "monitor": "loss/val",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def fusion(self, *features):
        raise NotImplementedError


class AdditionFusion(FusionClassification):
    def __init__(self, textual_dim, relational_dim, latent_dim, lr, weight_decay=1e-4):
        super().__init__(textual_dim, relational_dim, latent_dim, lr, weight_decay)

    def fusion(self, textual, relational):
        return textual + relational


class EarlyFusion(FusionClassification):
    def __init__(self, textual_dim, relational_dim, latent_dim, lr, weight_decay=1e-4):
        super().__init__(textual_dim, relational_dim, latent_dim, lr, weight_decay)
        self.fusion_layer = nn.Linear(2 * self.latent_dim, self.latent_dim)

    def fusion(self, textual, relational):
        return self.fusion_layer(torch.hstack((textual, relational)))


class LowRankFusion(FusionClassification):
    def __init__(
        self, textual_dim, relational_dim, latent_dim, lr, weight_decay=1e-4, rank=4
    ):
        super().__init__(textual_dim, relational_dim, latent_dim, lr, weight_decay)
        self.save_hyperparameters()
        self.rank = rank
        self.textual_factor: Float[Tensor, "rank d_latent d_output"] = nn.Parameter(
            nn.init.kaiming_normal_(
                torch.empty(self.rank, self.latent_dim + 1, self.latent_dim)
            )
        )
        self.relational_factor: Float[Tensor, "rank d_latent d_output"] = nn.Parameter(
            nn.init.kaiming_normal_(
                torch.empty(self.rank, self.latent_dim + 1, self.latent_dim)
            )
        )
        self.fusion_weights: Float[Tensor, "rank"] = nn.Parameter(torch.ones(self.rank))
        self.fusion_bias: Float[Tensor, "d_output"] = nn.Parameter(
            torch.zeros(self.latent_dim)
        )

    def append_ones(self, x):
        return torch.hstack((x, torch.ones(x.shape[0], 1, device=self.device)))

    def fusion(self, textual, relational):
        textual = self.append_ones(textual)
        relational = self.append_ones(relational)
        textual_fusion = einsum(
            textual,
            self.textual_factor,
            "batch d_latent, rank d_latent d_output -> batch rank d_output",
        )
        relational_fusion = einsum(
            relational,
            self.relational_factor,
            "batch d_latent, rank d_latent d_output -> batch rank d_output",
        )
        fused = textual_fusion * relational_fusion
        fused = (
            einsum(
                fused,
                self.fusion_weights,
                "batch rank d_output, rank -> batch d_output",
            )
            + self.fusion_bias
        )
        return fused


class TensorFusion(FusionClassification):
    def __init__(self, textual_dim, relational_dim, latent_dim, lr, weight_decay=1e-4):
        super().__init__(textual_dim, relational_dim, latent_dim, lr, weight_decay)
        self.save_hyperparameters()
        self.fusion_layer = nn.Linear((latent_dim + 1) ** 2, latent_dim)

    def append_ones(self, x):
        return torch.hstack((x, torch.ones(x.shape[0], 1, device=self.device)))

    def fusion(self, textual, relational):
        textual = self.append_ones(textual)
        relational = self.append_ones(relational)
        return self.fusion_layer(
            einsum(textual, relational, "b i, b j -> b i j").flatten(start_dim=1)
        )


class GatedFusion(FusionClassification):
    def __init__(self, textual_dim, relational_dim, latent_dim, lr, weight_decay=1e-4):
        super().__init__(textual_dim, relational_dim, latent_dim, lr, weight_decay)
        self.save_hyperparameters()
        self.score_layer = nn.Linear(2 * latent_dim, latent_dim)

    def fusion(self, textual, relational):
        scores = self.get_scores(textual, relational)
        return scores * textual + (1 - scores) * relational

    def get_scores(self, textual, relational):
        textual = (textual - torch.mean(textual, dim=0)) / torch.std(textual, dim=0)
        relational = (relational - torch.mean(relational, dim=0)) / torch.std(
            relational, dim=0
        )
        scores = torch.sigmoid(self.score_layer(torch.hstack((textual, relational))))
        return scores


class TransformerFusion(FusionClassification):
    def __init__(
        self,
        textual_dim,
        relational_dim,
        latent_dim,
        lr,
        num_layers=3,
        weight_decay=1e-4,
        nhead=4,
        output_modality="both",
    ):
        super().__init__(textual_dim, relational_dim, latent_dim, lr, weight_decay)
        self.save_hyperparameters()
        self.output_modality = output_modality
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=2 * latent_dim,
            dropout=0.1,
            activation="relu",
            norm_first=True,
        )
        if output_modality != "relational":
            self.textual_decoder = nn.TransformerDecoder(
                decoder_layer, num_layers=num_layers
            )
        if output_modality != "textual":
            self.relational_decoder = nn.TransformerDecoder(
                decoder_layer, num_layers=num_layers
            )
        if output_modality == "both":
            self.fusion_layer = nn.Linear(2 * latent_dim, latent_dim)

    def fusion(self, textual, relational):
        if self.output_modality == "textual":
            return self.textual_decoder(textual, relational)
        elif self.output_modality == "relational":
            return self.relational_decoder(relational, textual)

        textual = self.textual_decoder(textual, relational)
        relational = self.relational_decoder(relational, textual)

        return self.fusion_layer(torch.hstack((textual, relational)))


class Config(ConfigBase):
    def __init__(
        self,
        model_cls: str,
        textual_path: str,
        relational_path: str,
        latent_dim: int,
        max_epochs: int,
        lr: float,
        seed: int,
        prefix: str,
        kwargs: Dict,
    ):
        super().__init__(seed, prefix)
        self.textual_path = str(textual_path)
        self.relational_path = str(relational_path)
        self.model_cls = model_cls
        self.latent_dim = latent_dim
        self.max_epochs = max_epochs
        self.lr = lr
        self.kwargs = kwargs


class Driver(DriverBase):

    def setup_datamodule(self):
        datamodule = FeatureModule(
            feature_name_or_paths=[
                self.config.textual_path,
                self.config.relational_path,
            ]
        )

        # datamodule.setup()
        self.graph = datamodule.graph
        self.textual_dim = datamodule.features[0].shape[1]
        self.relational_dim = datamodule.features[1].shape[1]
        return datamodule

    def setup_model(self):
        model_cls = eval(self.config.model_cls)
        return model_cls(
            textual_dim=self.textual_dim,
            relational_dim=self.relational_dim,
            latent_dim=self.config.latent_dim,
            lr=self.config.lr,
            **self.config.kwargs,
        )

    def setup_trainer(self):
        return super().setup_trainer(
            callbacks=[
                # LoggingEmbeddingCallback(max_nsample=2000),
                LoggingGateScores(),
            ]
        )

    def get_node_embeddings(self):
        return (
            self.model.cuda()
            .encode(
                self.datamodule.features[0].cuda(), self.datamodule.features[1].cuda()
            )
            .type(torch.float16)
        )


def run(
    model_cls: str,
    textual_path: Path,
    relational_path: Path,
    latent_dim: int,
    seed: int,
    max_epochs: int,
    kwargs: Dict,
    prefix: str,
):
    # tdim, tname = textual_path.stem, textual_path.parent.stem
    # rdim, rname = relational_path.stem, relational_path.parent.stem
    config = Config(
        model_cls=model_cls,
        textual_path=str(textual_path),
        relational_path=str(relational_path),
        latent_dim=latent_dim,
        max_epochs=max_epochs,
        lr=0.01,
        seed=seed,
        prefix=prefix,
        kwargs=kwargs,
    )
    driver = Driver(config)
    if (driver.logdir / 'results.json').exists():
        print(f"Logdir {driver.logdir} already exists. Skipping.")
        return driver
    return driver.run()


class SingleRun:
    @app.command()
    def early_fusion(
        latent_dim: int,
        textual_path: Path,
        relational_path: Path,
        seed: int,
        max_epochs: int = 5,
    ):
        tdim, tname = textual_path.stem, textual_path.parent.stem
        rdim, rname = relational_path.stem, relational_path.parent.stem
        prefix = f"{tname}_{rname}/{tdim}_{rdim}"
        return run(
            model_cls="EarlyFusion",
            textual_path=textual_path,
            relational_path=relational_path,
            latent_dim=latent_dim,
            seed=seed,
            max_epochs=max_epochs,
            kwargs={},
            prefix=prefix,
        )

    @app.command()
    def addition_fusion(
        latent_dim: int,
        textual_path: Path,
        relational_path: Path,
        seed: int,
        max_epochs: int = 5,
    ):
        tdim, tname = textual_path.stem, textual_path.parent.stem
        rdim, rname = relational_path.stem, relational_path.parent.stem
        prefix = f"{tname}_{rname}/{tdim}_{rdim}"
        return run(
            model_cls="AdditionFusion",
            textual_path=textual_path,
            relational_path=relational_path,
            latent_dim=latent_dim,
            seed=seed,
            max_epochs=max_epochs,
            kwargs={},
            prefix=prefix,
        )

    @app.command()
    def gated_fusion(
        latent_dim: int,
        textual_path: Path,
        relational_path: Path,
        seed: int,
        max_epochs: int = 5,
    ):
        tdim, tname = textual_path.stem, textual_path.parent.stem
        rdim, rname = relational_path.stem, relational_path.parent.stem
        prefix = f"{tname}_{rname}/{tdim}_{rdim}"
        return run(
            model_cls="GatedFusion",
            textual_path=textual_path,
            relational_path=relational_path,
            latent_dim=latent_dim,
            seed=seed,
            max_epochs=max_epochs,
            kwargs={},
            prefix=prefix,
        )

    @app.command()
    def lowrank_fusion(
        latent_dim: int,
        textual_path: Path,
        relational_path: Path,
        seed: int,
        max_epochs: int = 5,
        rank: int = 4,
    ):
        tdim, tname = textual_path.stem, textual_path.parent.stem
        rdim, rname = relational_path.stem, relational_path.parent.stem
        prefix = f"{tname}_{rname}/{tdim}_{rdim}/{rank}"
        return run(
            model_cls="LowRankFusion",
            textual_path=textual_path,
            relational_path=relational_path,
            latent_dim=latent_dim,
            seed=seed,
            max_epochs=max_epochs,
            kwargs={"rank": rank},
            prefix=prefix,
        )

    @app.command()
    def transformer_fusion(
        latent_dim: int,
        textual_path: Path,
        relational_path: Path,
        seed: int,
        max_epochs: int = 5,
        num_layers: int = 1,
        nhead: int = 1,
        output_modality: str = "both",
    ):
        tdim, tname = textual_path.stem, textual_path.parent.stem
        rdim, rname = relational_path.stem, relational_path.parent.stem
        prefix = f"{tname}_{rname}/{tdim}_{rdim}/{num_layers}_{nhead}/{output_modality}"
        return run(
            model_cls="GatedFusion",
            textual_path=textual_path,
            relational_path=relational_path,
            latent_dim=latent_dim,
            seed=seed,
            max_epochs=max_epochs,
            kwargs={
                "num_layers": num_layers,
                "nhead": nhead,
                "output_modality": output_modality,
            },
            prefix=prefix,
        )


class Experiment:

    textuals = (SAVED_EMBEDDINGS_DIR / "textual").glob("**/*.pt")
    relationls = (SAVED_EMBEDDINGS_DIR / "relational").glob("**/*.pt")
    input_features = list(product(textuals, relationls))
    max_epochs = 5
    latent_dim = [32, 64, 128, 256]
    seeds = list(range(5))

    @app.command()
    def transformer_exp():
        nlayers = 3
        nheads = 4
        for output_modality in ["both", "textual", "relational"]:
            variables = product(Experiment.input_features, Experiment.seeds, Experiment.latent_dim)
            for (textual_path, relational_path), seed, latent_dim in tqdm(list(variables)):
                tdim, tname = textual_path.stem, textual_path.parent.stem
                rdim, rname = relational_path.stem, relational_path.parent.stem
                prefix = f"{tname}_{rname}/{tdim}_{rdim}/{nlayers}_{nheads}/{output_modality}"
                run(
                    model_cls="TransformerFusion",
                    textual_path=textual_path,
                    relational_path=relational_path,
                    latent_dim=latent_dim,
                    seed=seed,
                    max_epochs=Experiment.max_epochs,
                    kwargs={
                        "num_layers": nlayers,
                        "nhead": nheads,
                        "output_modality": output_modality,
                    },
                    prefix=prefix,
                )

    @app.command()
    def early_exp():
        variables = product(Experiment.input_features, Experiment.seeds, Experiment.latent_dim)
        for (textual_path, relational_path), seed, latent_dim in tqdm(list(variables)):
            tdim, tname = textual_path.stem, textual_path.parent.stem
            rdim, rname = relational_path.stem, relational_path.parent.stem
            prefix = f"{tname}_{rname}/{tdim}_{rdim}"
            run(
                model_cls="EarlyFusion",
                textual_path=textual_path,
                relational_path=relational_path,
                latent_dim=latent_dim,
                seed=seed,
                max_epochs=Experiment.max_epochs,
                kwargs={},
                prefix=prefix,
            )

    @app.command()
    def addition_exp():
        variables = product(Experiment.input_features, Experiment.seeds, Experiment.latent_dim)
        for (textual_path, relational_path), seed, latent_dim in tqdm(list(variables)):
            tdim, tname = textual_path.stem, textual_path.parent.stem
            rdim, rname = relational_path.stem, relational_path.parent.stem
            prefix = f"{tname}_{rname}/{tdim}_{rdim}"
            run(
                model_cls="AdditionFusion",
                textual_path=textual_path,
                relational_path=relational_path,
                latent_dim=latent_dim,
                seed=seed,
                max_epochs=Experiment.max_epochs,
                kwargs={},
                prefix=prefix,
            )

    @app.command()
    def gated_exp():
        variables = product(Experiment.input_features, Experiment.seeds, Experiment.latent_dim)
        for (textual_path, relational_path), seed, latent_dim in tqdm(list(variables)):
            tdim, tname = textual_path.stem, textual_path.parent.stem
            rdim, rname = relational_path.stem, relational_path.parent.stem
            prefix = f"{tname}_{rname}/{tdim}_{rdim}"
            run(
                model_cls="GatedFusion",
                textual_path=textual_path,
                relational_path=relational_path,
                latent_dim=latent_dim,
                seed=seed,
                max_epochs=Experiment.max_epochs,
                kwargs={},
                prefix=prefix,
            )

    @app.command()
    def lowrank_exp():
        rank = 4
        variables = product(Experiment.input_features, Experiment.seeds, Experiment.latent_dim)
        for (textual_path, relational_path), seed, latent_dim in tqdm(list(variables)):
            tdim, tname = textual_path.stem, textual_path.parent.stem
            rdim, rname = relational_path.stem, relational_path.parent.stem
            prefix = f"{tname}_{rname}/{tdim}_{rdim}/{rank}"
            run(
                model_cls="LowRankFusion",
                textual_path=textual_path,
                relational_path=relational_path,
                latent_dim=latent_dim,
                seed=seed,
                max_epochs=Experiment.max_epochs,
                kwargs={"rank": rank},
                prefix=prefix,
            )



if __name__ == "__main__":
    app()
