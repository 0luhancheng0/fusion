import typer
import matplotlib
import os
from pathlib import Path
from typing import Optional, List, Tuple, Union, Dict, Any, Type, Callable
import matplotlib.pyplot as plt
from enum import Enum
import numpy as np

# Import analyzers
from analysis.asgc import ASGCAnalyzer
from analysis.crossmodel import CrossModelAnalyzer
from analysis.textual import TextualEmbeddingsAnalyzer
from analysis.node2vec import Node2VecAnalyzer
from analysis.fusion.early import EarlyFusionAnalyzer
from analysis.fusion.gated import GatedFusionAnalyzer
from analysis.fusion.lowrank import LowRankFusionAnalyzer
from analysis.fusion.transformer import TransformerFusionAnalyzer

# Create the main app
app = typer.Typer(help="Command line interface for fusion analysis tools")

class OutlierMethod(str, Enum):
    """Methods for outlier detection"""
    IQR = "iqr"
    ZSCORE = "zscore"
    PERCENTILE = "percentile"

class ColorMap(str, Enum):
    """Common colormaps"""
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    MAGMA = "magma"
    CIVIDIS = "cividis"
    BLUES = "Blues"
    REDS = "Reds"
    GREENS = "Greens"
    COOLWARM = "coolwarm"

# Define a callback for figsize parsing
def parse_figsize(value: Union[Tuple[float, float], str]) -> Tuple[float, float]:
    """Parse figsize from string like '12,8' to tuple (12.0, 8.0)"""
    if isinstance(value, tuple) and len(value) == 2:
        return value  # Already in correct format
    try:
        width, height = value.split(',')
        return (float(width), float(height))
    except:
        raise typer.BadParameter("figsize must be in the format 'width,height'")

# Common parameters for all analyzers
def common_analyzer_options(f: Callable) -> Callable:
    """Decorator to add common analyzer options to a command."""
    common_options = [
        typer.Option(300, help="DPI for saved figures"),
        typer.Option(ColorMap.VIRIDIS, help="Colormap to use for plots"),
        typer.Option("12,8", help="Figure size as width,height", callback=parse_figsize)
    ]
    
    for option in reversed(common_options):
        f = typer.Option(option.default, help=option.help, callback=option.callback)(f)
    
    return f

def run_analyzer(analyzer_class: Type, **kwargs) -> Any:
    """Common function to run any analyzer with provided parameters."""
    analyzer = analyzer_class(**kwargs)
    return analyzer.run()

@app.command()
def asgc(
    dpi: int = typer.Option(300, help="DPI for saved figures"),
    cmap: ColorMap = typer.Option(ColorMap.VIRIDIS, help="Colormap to use for plots"),
    figsize: str = typer.Option("12,8", help="Figure size as width,height", callback=parse_figsize),
):
    """Run ASGC analyzer"""
    typer.echo("Running ASGC analyzer...")
    return run_analyzer(
        ASGCAnalyzer,
        dpi=dpi,
        cmap=cmap,
        figsize=figsize,
    )

@app.command()
def crossmodel(
    dpi: int = typer.Option(300, help="DPI for saved figures"),
    cmap: ColorMap = typer.Option(ColorMap.VIRIDIS, help="Colormap to use for plots"),
    figsize: str = typer.Option("12,8", help="Figure size as width,height", callback=parse_figsize),
):
    """Run cross-model comparison analyzer"""
    typer.echo("Running cross-model comparison...")
    return run_analyzer(
        CrossModelAnalyzer,
        dpi=dpi,
        cmap=cmap,
        figsize=figsize,
    )

@app.command()
def node2vec(
    dpi: int = typer.Option(300, help="DPI for saved figures"),
    cmap: ColorMap = typer.Option(ColorMap.VIRIDIS, help="Colormap to use for plots"),
    figsize: str = typer.Option("12,8", help="Figure size as width,height", callback=parse_figsize),
):
    """Run Node2Vec analyzer"""
    typer.echo("Running Node2Vec analyzer...")
    return run_analyzer(
        Node2VecAnalyzer,
        dpi=dpi,
        cmap=cmap,
        figsize=figsize,
    )

@app.command()
def textual(
    dpi: int = typer.Option(300, help="DPI for saved figures"),
    cmap: ColorMap = typer.Option(ColorMap.VIRIDIS, help="Colormap to use for plots"),
    figsize: str = typer.Option("12,8", help="Figure size as width,height", callback=parse_figsize),
):
    """Run Textual Embeddings analyzer"""
    typer.echo("Running Textual Embeddings analyzer...")
    return run_analyzer(
        TextualEmbeddingsAnalyzer,
        dpi=dpi,
        cmap=cmap,
        figsize=figsize,
    )

@app.command()
def fusion(
    fusion_type: str = typer.Argument(..., help="Fusion type (addition, early, gated, lowrank, transformer)"),
    dpi: int = typer.Option(300, help="DPI for saved figures"),
    cmap: ColorMap = typer.Option(ColorMap.VIRIDIS, help="Colormap to use for plots"),
    figsize: str = typer.Option("12,8", help="Figure size as width,height", callback=parse_figsize),
):
    """Run a specific fusion analyzer"""
    # Map of fusion type names to analyzers
    fusion_analyzers = {
        "early": EarlyFusionAnalyzer,
        "gated": GatedFusionAnalyzer,
        "lowrank": LowRankFusionAnalyzer,
        "transformer": TransformerFusionAnalyzer
    }
    
    if fusion_type.lower() not in fusion_analyzers:
        typer.echo(f"Unknown fusion type: {fusion_type}")
        typer.echo(f"Available types: {', '.join(fusion_analyzers.keys())}")
        raise typer.Exit(1)
    
    typer.echo(f"Running {fusion_type} fusion analyzer...")
    
    return run_analyzer(
        fusion_analyzers[fusion_type.lower()], 
        dpi=dpi,
        cmap=cmap,
        figsize=figsize,
    )
@app.command()
def all(
    dpi: int = typer.Option(300, help="DPI for saved figures"),
    cmap: ColorMap = typer.Option(ColorMap.VIRIDIS, help="Colormap to use for plots"),
    figsize: str = typer.Option("12,8", help="Figure size as width,height", callback=parse_figsize),
):
    """Run all analyzers"""
    results = {
        "asgc": run_analyzer(ASGCAnalyzer, dpi=dpi, cmap=cmap, figsize=figsize),
        "crossmodel": run_analyzer(CrossModelAnalyzer, dpi=dpi, cmap=cmap, figsize=figsize),
        "node2vec": run_analyzer(Node2VecAnalyzer, dpi=dpi, cmap=cmap, figsize=figsize),
        "textual": run_analyzer(TextualEmbeddingsAnalyzer, dpi=dpi, cmap=cmap, figsize=figsize),
        "early": run_analyzer(EarlyFusionAnalyzer, dpi=dpi, cmap=cmap, figsize=figsize),
        "gated": run_analyzer(GatedFusionAnalyzer, dpi=dpi, cmap=cmap, figsize=figsize),
        "lowrank": run_analyzer(LowRankFusionAnalyzer, dpi=dpi, cmap=cmap, figsize=figsize),
        "transformer": run_analyzer(TransformerFusionAnalyzer, dpi=dpi, cmap=cmap, figsize=figsize),
    }
    return results

if __name__ == "__main__":
    if "--show" not in os.sys.argv and "--no-show" not in os.sys.argv:
        matplotlib.use("Agg")
    
    app()
