from pathlib import Path
import torch
import pandas as pd
embeddings = list(Path("/home/lcheng/oz318/fusion/logs/Node2VecLightning").glob("**/*.pt"))
seeds = [i.parent.stem for i in embeddings]
dims = [i.parent.parent.stem for i in embeddings]
embeddings = [torch.load(i, weights_only=False) for i in embeddings]
results = [i['metadata']['results'] for i in embeddings]
df = pd.DataFrame(results)
df['seeds'] = seeds
df['dims'] = dims
# df.set_index(['dims', 'seeds'], inplace=True)
df.groupby(['dims']).agg(
    {
        'acc/val': 'mean',
        'acc/test': 'mean',
        'lp/auc': 'mean'
    }
)

