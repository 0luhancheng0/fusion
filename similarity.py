%matplotlib inline
from cuvs.distance import pairwise_distance
import torch
import cupy as cp
from dataloading import OGBNArxivDataset
import seaborn as sns
from itertools import product


em1 = cp.asarray(torch.load("/fred/oz318/luhanc/fusion/saved_embeddings/ogbn-arxiv/textual/mpnet/768.pt", map_location="cuda:0"))
em2 = cp.asarray(torch.load("/fred/oz318/luhanc/fusion/saved_embeddings/ogbn-arxiv/textual/roberta/768.pt", map_location="cuda:0"))


arxiv = OGBNArxivDataset()
results = cp.empty((40, 40))
for i, j in product(range(arxiv.labels.unique().shape[0]), range(arxiv.labels.unique().shape[0])):
    mask1 = cp.asarray(arxiv.labels) == i
    mask2 = cp.asarray(arxiv.labels) == j
    
    dist1 = pairwise_distance(em1[mask1], em1[mask2], metric="cosine")
    dist2 = pairwise_distance(em2[mask1], em2[mask2], metric="cosine")
    corr = pairwise_distance(dist1, dist2, metric="correlation")
    results[i, j] = 1 - corr.copy_to_host().mean()
    
    # Flatten the distance matrices
    # flat_dist1 = dist1.copy_to_host().flatten()
    # flat_dist2 = dist2.copy_to_host().flatten()

    # Compute correlation coefficient
    # results[i, j] = spearmanr(flat_dist1, flat_dist2).statistic
results = results.get()
# Find the coordinates of the maximum element
max_idx = results.flatten().argmax()
max_i, max_j = max_idx // results.shape[1], max_idx % results.shape[1]

print(f"Maximum value {results[max_i, max_j]} found at coordinates ({max_i}, {max_j})")

results[30, 1]