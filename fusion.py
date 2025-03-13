from abc import ABC
from pathlib import Path
from typing import Literal
import lightning as L
import torch
import torchmetrics.functional as F
from einops import einsum
from jaxtyping import Float
from torch import Tensor, nn
from base import AbstractConfig, AbstractDriver
from dataloading import FeatureModule
from callbacks import LoggingEmbeddingCallback, LoggingGateScores
from typing import Dict
from typer import Typer
from itertools import product
import constants
from tqdm import tqdm
from torchinfo import summary
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
        output_modality = "both",
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
            norm_first=True
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


class SAE(nn.Module):
    def __init__(self, d_in, d_latent, d_out):
        super().__init__()
        self.W_enc = nn.Parameter(nn.init.kaiming_normal_(torch.randn(d_in, d_latent)))
        self._W_dec = nn.Parameter(
            nn.init.kaiming_normal_(torch.randn(d_latent, d_out))
        )
        self.b_enc = nn.Parameter(torch.zeros(d_latent))
        self.b_dec = nn.Parameter(torch.zeros(d_out))

    def W_dec(self):
        return self._W_dec / (self._W_dec.norm(dim=-1, keepdim=True) + 1e-8)

    def forward(self, x, return_z=False):
        x = einsum(self.W_enc, x, "d_in d_latent, b d_in -> b d_latent") + self.b_enc
        z = torch.nn.functional.relu(x)
        y = einsum(self.W_dec, z, "d_latent d_out, b d_latent -> b d_out") + self.b_dec
        if not return_z:
            return y
        return y, z


class Config(AbstractConfig):
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


class Driver(AbstractDriver):

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
    prefix: str
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
    return driver.run()

@app.command()
def early_fusion(
    latent_dim: int,
    textual_path: Path,
    relational_path: Path,
    seed: int,
    max_epochs = 5
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
    max_epochs = 5
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
    max_epochs = 5,
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
    max_epochs = 5,
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
    max_epochs = 5,
    num_layers: int = 1,
    nhead: int = 1,
    output_modality: Literal["textual", "relational", "both"] = "both",
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



def run_experiments():
    textual = constants.SAVED_EMBEDDINGS_DIR / "textual"
    relational = constants.SAVED_EMBEDDINGS_DIR / "relational"

    textual_embeddings = list(textual.glob("**/*.pt"))
    relational_embeddings = list(relational.glob("**/*.pt"))

    # Define search space parameters
    latent_dims = [16, 32, 64, 128]
    model_classes = [TransformerFusion, LowRankFusion, GatedFusion, EarlyFusion]
    seeds = range(5)
    input_features = list(product(textual_embeddings, relational_embeddings))

    # Use itertools.product to generate the grid search combinations
    for latent_dim, model, (t, r), seed in tqdm(
        list(product(latent_dims, model_classes, input_features, seeds))
    ):
        train(
            model_cls=model.__name__,
            latent_dim=latent_dim,
            textual_path=t,
            relational_path=r,
            seed=seed,
        )

def run_transformer_fusion_experiments():
    textual = constants.SAVED_EMBEDDINGS_DIR / "textual"
    relational = constants.SAVED_EMBEDDINGS_DIR / "relational"

    textual_embeddings = list(textual.glob("**/*.pt"))
    relational_embeddings = list(relational.glob("**/*.pt"))

    # Define search space parameters
    latent_dims = [16, 32, 64, 128]
    seeds = range(5)
    input_features = list(product(textual_embeddings, relational_embeddings))
    output_modalities = ["textual", "relational", "both"]
    num_layers = [1, 3, 6, 12]
    nheads = [4, 8, 12]
    

    # Use itertools.product to generate the grid search combinations
    for latent_dim, (t, r), seed, nlayer, output_modality, nhead in tqdm(
        list(product(latent_dims, input_features, seeds, num_layers, output_modalities, nheads))
    ):
        train(
            model_cls="TransformerFusion",
            latent_dim=latent_dim,
            textual_path=t,
            relational_path=r,
            seed=seed,
            kwargs={
                "num_layers": nlayer,
                "nhead": nhead,
                "output_modality": output_modality,
            },
        )

# relational_path = constants.SAVED_EMBEDDINGS_DIR / "relational" / "node2vec" / "128.pt"
# textual_path = constants.SAVED_EMBEDDINGS_DIR / "textual" / "minilm" / "384.pt"
# seed = "0"
# tdim, tname = textual_path.stem, textual_path.parent.stem
# rdim, rname = relational_path.stem, relational_path.parent.stem
# output_modality = "relational"

# kwargs={
#     "num_layers": 1,
#     "nhead": 1,
#     "output_modality": output_modality,       
# }
# config = Config(
#     model_cls="TransformerFusion",
#     textual_path=textual_path,
#     relational_path=relational_path,
#     latent_dim=128,
#     max_epochs=1,
#     lr=0.01,
#     seed=seed,
#     prefix=f"{tname}_{rname}/{tdim}_{rdim}/{output_modality}",
#     kwargs=kwargs,
# )
# driver = Driver(config)
# driver.run()


if __name__ == "__main__":
    app()