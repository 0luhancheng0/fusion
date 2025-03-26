%load_ext autoreload
%matplotlib inline
%autoreload 2

from pathlib import Path
from analysis.crossmodel import CrossModelAnalyzer
import json
crossmodel = CrossModelAnalyzer()

crossmodel.df



# crossmodel.df.iloc[:5].input_features[0]