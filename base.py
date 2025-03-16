from abc import ABC, abstractmethod
from pathlib import Path
from pprint import pformat
import torch
import lightning as L
from torchinfo import summary
from lightning.pytorch.loggers import TensorBoardLogger
from lightning_fabric.utilities.seed import seed_everything
import constants
from lightning.pytorch.callbacks import ModelCheckpoint
from dataloading import OGBNArxivDataset
import json
import logging

log = logging.getLogger("lightning_fabric")
log.propagate = False
log.setLevel(logging.ERROR)


class ConfigBase(ABC):
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

    def to_json(self, file_path=None):
        """Serialize the config to JSON."""
        json_dict = self.__dict__
        if file_path:
            with open(file_path, "w") as f:
                json.dump(json_dict, f, indent=4)
        return json.dumps(json_dict, indent=4)

    @classmethod
    def from_json(cls, json_str=None, file_path=None):
        """Create a config instance from a JSON string or file."""
        if file_path:
            with open(file_path, "r") as f:
                config_dict = json.load(f)
        else:
            config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)


class DriverBase(ABC):
    """Abstract base class for all model drivers."""

    def __init__(self, config: ConfigBase):
        """Initialize the driver with a configuration."""
        self.config = config
        seed_everything(self.config.seed)
        self.setup()
        

    def setup(self):

        self.datamodule = self.setup_datamodule()
        self.evaluator = self.setup_evaluator()
        self.model = self.setup_model()
        self.logger = self.setup_logger()
        self.trainer = self.setup_trainer()
        self.results = None

    @property
    def logdir(self):
        return Path(self.trainer.logger.log_dir)

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
            version=str(self.config.seed),
        )

    def save_embeddings(self):
        saved_path = Path(self.logdir) / "embeddings.pt"
        torch.save(self.get_node_embeddings(), saved_path)
        return saved_path

    def save_config(self):
        """Save the configuration to a file."""
        config_file = Path(self.logdir) / "config.json"
        self.config.to_json(config_file)

    def save_results(self):
        """Save the results to a file."""
        results_file = Path(self.logdir) / "results.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=4)

    def setup_trainer(self, callbacks=[], monitor="loss/val", mode="min", **kwargs):
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
            enable_model_summary=False,
            enable_progress_bar=False,
            **kwargs
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
            # "lp/auc/hard": self.evaluator.evaluate_link_prediction(
            #     self.graph, self.get_node_embeddings(), hard_negative=True
            # ),
            # "lp/auc/random": self.evaluator.evaluate_link_prediction(
            #     self.graph, self.get_node_embeddings(), hard_negative=False
            # ),
            "lp/auc": self.evaluator.evaluate_link_prediction(
                self.graph, self.get_node_embeddings()
            ),
        }
        return self.results

    def setup_evaluator(self):
        return OGBNArxivDataset.evaluator()

    def log_model_summary(self):
        self.trainer.logger.experiment.add_text("summary", str(summary(self.model)))
        
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
        self.save_config()
        self.evaluate()
        self.save_results()
        self.log_model_summary()
        return self
