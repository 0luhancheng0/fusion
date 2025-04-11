%matplotlib inline
%load_ext autoreload
%autoreload 2

from dataloading import OGBNArxivDataset
import pandas as pd
import torch as t



dataset = OGBNArxivDataset()
reviews = dataset.paper_metadata.title[dataset.paper_metadata.title.str.contains("review", case=False) | dataset.paper_metadata.title.str.contains("survey", case=False)]
nodes = dataset.paper2node(reviews.index)
eids = dataset.graph.out_edges(nodes, form="eid")
dataset.graph.edata["review_cite"] = t.zeros(dataset.graph.num_edges(), dtype=t.bool).scatter_(0, eids, True)
 

