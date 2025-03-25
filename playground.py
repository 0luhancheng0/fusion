import pandas as pd
import json
import torch
from pathlib import Path
from dataloading import OGBNArxivDataset


all_results = Path("logs/all_results.json")
with all_results.open("r") as f:
    data = json.load(f)
# Load all_results.json into pandas DataFrame
df = pd.read_json("logs/all_results.json").transpose()
mask = df.index.str.startswith("TransformerFusion") & (df['lp_hard/auc'] > 0.8) & (df['lp_hard/auc'] != 1.)
df[mask].index[0]



dataset = OGBNArxivDataset()
evaluator = dataset.evaluator()
path = "logs" / Path(df[mask].index[0]) / "embeddings.pt"
embeddings = torch.load(path, map_location="cpu")

evaluator.evaluate_link_prediction(embeddings)

df[mask].iloc[0]




# logs = Path("logs")
# results = logs.glob("**/results.json")


# for result_path in results:
#     try:
#         with open(result_path, 'r') as f:
#             result_data = json.load(f)
#         if 'lp/auc' in result_data:
#             del result_data['lp/auc']
#             with open(result_path, 'w') as f:
#                 json.dump(result_data, f, indent=4)
#     except Exception as e:
#         print(f"Error processing {result_path}: {e}")



# import torch
# em = torch.load("/home/lcheng/oz318/fusion/logs/TransformerFusion/specter_asgc/768_64/256/3_4/both/2/embeddings.pt", map_location="cpu")
