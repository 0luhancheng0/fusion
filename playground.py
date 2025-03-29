%reload_ext autoreload
%autoreload 2
%matplotlib inline
import pandas as pd
from analysis.crossmodel import CrossModelAnalyzer
from pathlib import Path
analyzer = CrossModelAnalyzer()
# analyzer.per_embedding_combo()
