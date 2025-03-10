from torch import Tensor
from typing import Optional, Dict, Any, Union
import torch
import numpy as np

def index_to_mask(index: Tensor, size: Optional[int] = None) -> Tensor:
    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = index.new_zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask

def mask_to_index(mask: Tensor) -> Tensor:
    return mask.nonzero(as_tuple=False).view(-1)

def nsample_per_class(n, labels):
    """Sample n instances from each class in labels"""
    unique_labels = np.unique(labels)
    indices = []
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        label_indices = np.random.permutation(label_indices)[:n]
        indices.extend(label_indices)
    indices = np.random.permutation(indices)
    return torch.tensor(indices, dtype=torch.int)
