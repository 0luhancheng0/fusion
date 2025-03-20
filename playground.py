import pandas as pd
import json
import torch
from pathlib import Path
all_results = Path("logs/all_results.json")
with all_results.open("r") as f:
    data = json.load(f)
# Load all_results.json into pandas DataFrame
df = pd.read_json("logs/all_results.json").transpose()
mask = df.index.str.startswith("TransformerFusion") & (df['lp_hard/auc'] > 0.8) & (df['lp_hard/auc'] != 1.)
df[mask].index[0]

from dataloading import OGBNArxivDataset


dataset = OGBNArxivDataset()
evaluator = dataset.evaluator()

embeddings = torch.load("logs" / Path(df[mask].index[0]) / "embeddings.pt", map_location="cpu")
evaluator.evaluate_link_prediction(dataset.graph, embeddings, use_hard_negatives=False, check_invalid_values=False)
df[mask].iloc[0]






logs = Path("logs")
results = logs.glob("**/results.json")

# Iterate through all results.json files
for result_path in results:
    try:
        # Load the JSON file
        with open(result_path, 'r') as f:
            result_data = json.load(f)
        
        # Check if 'lp/auc' key exists and remove it
        if 'lp/auc' in result_data:
            del result_data['lp/auc']
            # Save the modified data back to the file
            with open(result_path, 'w') as f:
                json.dump(result_data, f, indent=4)
    except Exception as e:
        print(f"Error processing {result_path}: {e}")
