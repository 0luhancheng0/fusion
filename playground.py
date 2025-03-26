from analysis.node2vec import Node2VecAnalyzer
from analysis.asgc import ASGCAnalyzer
from analysis.textual import TextualEmbeddingsAnalyzer
from analysis.fusion.early import EarlyFusionAnalyzer
from analysis.fusion.gated import GatedFusionAnalyzer
from analysis.fusion.lowrank import LowRankFusionAnalyzer
from analysis.fusion.transformer import TransformerFusionAnalyzer
from analysis.crossmodel import CrossModelAnalyzer
%matplotlib inline
# node2vec = Node2VecAnalyzer()

# node2vec.performance_by_dimension()
# node2vec.visualize_task_comparison()
# node2vec.visualize_lp_comparison()

# asgc = ASGCAnalyzer()
# asgc.heatmap()
# asgc.visualize_parameter_impact()
# asgc.visualize_coefficient_heatmaps()
# asgc.visualize_hyperparameter_sensitivity()


# lowrank = LowRankFusionAnalyzer()
# results = lowrank.run()

# transformer = TransformerFusionAnalyzer()

# transformer.visualize(x_axis="output_modality", y_axis="textual_names")
# lowrank.visualize(x_axis="rank", y_axis="textual_dim")
# results['dimension_impact']
crossmodel = CrossModelAnalyzer()