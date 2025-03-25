import torch
from pathlib import Path
import pandas as pd
import json 

raw = json.load(Path("logs/all_results.json").open("r"))
df = pd.DataFrame(raw).transpose()
mask = df["lp_hard/auc"] == df["lp_hard/auc"].max()
transformer_fusions = df[df.index.str.startswith("TransformerFusion")]
