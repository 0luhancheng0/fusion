from pathlib import Path
FEATURE = "feat"
LABEL = "label"
TEST_MASK = "test_mask"
VALID_MASK = "val_mask"
TRAIN_MASK = "train_mask"

ROOT_DIR = Path("/fred/oz318/luhanc/fusion")


SAVED_EMBEDDINGS_DIR = Path("/home/lcheng/oz318/fusion/saved_embeddings/ogbn-arxiv")



# CONFIG_DIR = ROOT_DIR / "configs"
# DEFAULT_MAX_EPOCHS = 5
# DEFAULT_REPEAT = 5
LOG_DIR = Path("/home/lcheng/oz318/fusion/logs")

FIGURE_PATH = LOG_DIR / "figures"


# NEGATIVE_SAMPLES_DIR = Path("logs/negative_samples")
HARD_NEGATIVE_SAMPLES = Path("/fred/oz318/luhanc/fusion/logs/arxiv_hard_negatives.pt")

UNIFORM_NEGATIVE_SAMPLES = Path("/fred/oz318/luhanc/fusion/logs/arxiv_uniform_negatives.pt")