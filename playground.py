%load_ext autoreload
%matplotlib inline
%autoreload 2

from analysis.node2vec import Node2VecAnalyzer
from analysis.asgc import ASGCAnalyzer
from analysis.textual import TextualEmbeddingsAnalyzer
from analysis.fusion.early import EarlyFusionAnalyzer
from analysis.fusion.gated import GatedFusionAnalyzer
from analysis.fusion.lowrank import LowRankFusionAnalyzer
from analysis.fusion.transformer import TransformerFusionAnalyzer
from analysis.crossmodel import CrossModelAnalyzer

# results['dimension_impact']
crossmodel = CrossModelAnalyzer()

