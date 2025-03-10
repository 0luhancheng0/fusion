from abc import ABC, abstractmethod
from pathlib import Path
from pprint import pformat
import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning_fabric.utilities.seed import seed_everything
import constants
from lightning.pytorch.callbacks import ModelCheckpoint
from dataloading import OGBNArxivDataset


class AbstractConfig(ABC):
    """Abstract base class for all configuration objects."""
    def __init__(self, seed, prefix):
        self.seed = seed
        self.prefix = prefix

    def __str__(self):
        """Return a string representation of the configuration."""
        return pformat(self.__dict__)

    def __repr__(self):
        """Return a string representation of the configuration."""
        return self.__str__()


class AbstractDriver(ABC):
    """Abstract base class for all model drivers."""

    def __init__(self, config):
        """Initialize the driver with a configuration."""
        self.config = config
        self.setup()
        seed_everything(self.config.seed)

    def setup(self):

        self.datamodule = self.setup_datamodule()
        self.evaluator = self.setup_evaluator()
        self.model = self.setup_model()
        self.logger = self.setup_logger()
        self.trainer = self.setup_trainer()
        self.results = None

    @property
    def logdir(self):
        return self.trainer.logger.log_dir

    def save_hparams(self):
        hparams_file = Path(self.logdir) / TensorBoardLogger.NAME_HPARAMS_FILE
        hparams_file.unlink(missing_ok=True)
        self.logger.hparams = self.config.__dict__
        self.logger.save()
        # self.logger.log_hyperparams(params=self.config.__dict__)

    def setup_logger(self):
        return TensorBoardLogger(
            save_dir=constants.LOG_DIR,
            name=f"{self.model.__class__.__name__}/{self.config.prefix}",
            version=str(self.config.seed)
        )

    def save_embeddings(self):

        embeddings = {
            "embeddings": self.get_node_embeddings(),
            "metadata": {
                "config": self.config.__dict__,
                "results": self.evaluate(),
            },
        }
        saved_path = Path(self.logdir) / "embeddings.pt"

        torch.save(embeddings, saved_path)
        return saved_path

    def setup_trainer(self, callbacks=[], monitor="loss/val", mode="min"):
        """Set up the Lightning trainer."""

        return L.Trainer(
            max_epochs=self.config.max_epochs,
            logger=self.logger,
            callbacks=[
                *callbacks,
                ModelCheckpoint(
                    dirpath=self.logger.log_dir,
                    monitor=monitor,
                    mode=mode,
                    save_top_k=1,
                    auto_insert_metric_name=False,
                ),
            ],
        )

    def fit(self):
        """Train the model."""
        return self.trainer.fit(self.model, self.datamodule)

    def validate(self):
        """Validate the model."""
        return self.trainer.validate(self.model, self.datamodule)[0]

    def test(self):
        """Test the model."""
        return self.trainer.test(self.model, self.datamodule)[0]

    def best_model(self):
        if self.trainer.checkpoint_callback.best_model_path == "":
            raise FileExistsError(f"Run fit before calling best_model.")
        return self.model.__class__.load_from_checkpoint(
            self.trainer.checkpoint_callback.best_model_path,
            hparams_file=Path(self.logdir) / TensorBoardLogger.NAME_HPARAMS_FILE,
        )

    def evaluate(self):
        self.results = {
            **self.validate(),
            **self.test(),
            "lp/auc": self.evaluator.evaluate_link_prediction(
                self.graph, self.get_node_embeddings()
            ),
        }
        return self.results

    def setup_evaluator(self):
        return OGBNArxivDataset.evaluator()

    @property
    def best_model_path(self):
        return self.trainer.checkpoint_callback.best_model_path

    @abstractmethod
    def setup_model(self):
        pass

    @abstractmethod
    def setup_datamodule(self):
        pass

    @abstractmethod
    def get_node_embeddings(self):
        pass

    def run(self):
        self.fit()
        self.save_hparams()
        self.save_embeddings()
        return self.results
