import typer
import matplotlib
import os
from pathlib import Path
from typing import Optional, List, Tuple, Union
import matplotlib.pyplot as plt
from enum import Enum
import numpy as np

# Import analyzers
from analysis.asgc import ASGCAnalyzer
from analysis.crossmodel import CrossModelAnalyzer
from analysis.textual import TextualEmbeddingsAnalyzer
from analysis.node2vec import Node2VecAnalyzer
from analysis.fusion.addition import AdditionFusionAnalyzer
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

@app.command()
def asgc(
    metrics: List[str] = typer.Option(["acc/test"], help="Metrics for outlier removal"),
    k_value: Optional[int] = typer.Option(None, help="Filter by specific k value"),
    reg_value: Optional[float] = typer.Option(None, help="Filter by specific regularization value"),
    dim_value: Optional[int] = typer.Option(None, help="Filter by specific dimension"),
    show_coefficients: bool = typer.Option(False, help="Show coefficient visualizations"),
    dpi: int = typer.Option(300, help="DPI for saved figures"),
    cmap: ColorMap = typer.Option(ColorMap.VIRIDIS, help="Colormap to use for plots"),
    figsize: str = typer.Option("12,8", help="Figure size as width,height", callback=parse_figsize),
    remove_outliers: bool = typer.Option(False, help="Whether to remove outliers"),
    outlier_method: OutlierMethod = typer.Option(OutlierMethod.IQR, help="Method for outlier detection"),
    outlier_threshold: float = typer.Option(1.5, help="Threshold for outlier detection"),
    save: bool = typer.Option(True, help="Save figures"),
    show: bool = typer.Option(False, help="Show figures (may not work in headless environments)")
):
    """Run ASGC analyzer"""
    typer.echo("Running ASGC analyzer...")
    
    # Initialize analyzer with common options
    analyzer = ASGCAnalyzer(
        dpi=dpi,
        cmap=cmap,
        figsize=figsize,
        remove_outliers=remove_outliers,
        outlier_params=build_outlier_params(metrics, outlier_method, outlier_threshold)
        if remove_outliers else None
    )
    
    # Filter if requested
    if any(v is not None for v in [k_value, reg_value, dim_value]):
        typer.echo(f"Filtering data: k={k_value}, reg={reg_value}, dim={dim_value}")
        filter_conditions = []
        if k_value is not None:
            filter_conditions.append(f"k == {k_value}")
        if reg_value is not None:
            filter_conditions.append(f"reg == {reg_value}")
        if dim_value is not None:
            filter_conditions.append(f"dim == {dim_value}")
            
        query = " and ".join(filter_conditions)
        analyzer.df = analyzer.df.query(query)
        typer.echo(f"After filtering: {len(analyzer.df)} records remain")
    
    # Run analysis
    results = analyzer.run()
    
    # Show additional coefficient visualizations if requested
    if show_coefficients and not analyzer.df.empty:
        top_idx = analyzer.df["acc/test"].idxmax()
        coef_fig = analyzer.visualize_coefficients(top_idx)
        results["best_coefficients"] = coef_fig
        typer.echo("Created coefficient visualization for best model")
    
    # Handle display options
    if show:
        typer.echo("Showing figures...")
        plt.show()
    
    return results

@app.command()
def crossmodel(
    metrics: List[str] = typer.Option(["acc/test"], help="Metrics for outlier removal"),
    model_filter: Optional[str] = typer.Option(None, help="Filter specific model types (comma-separated)"),
    show_detailed_metrics: bool = typer.Option(True, help="Show detailed metrics analyses"),
    dpi: int = typer.Option(300, help="DPI for saved figures"),
    cmap: ColorMap = typer.Option(ColorMap.VIRIDIS, help="Colormap to use for plots"),
    figsize: str = typer.Option("12,8", help="Figure size as width,height", callback=parse_figsize),
    remove_outliers: bool = typer.Option(False, help="Whether to remove outliers"),
    outlier_method: OutlierMethod = typer.Option(OutlierMethod.IQR, help="Method for outlier detection"),
    outlier_threshold: float = typer.Option(1.5, help="Threshold for outlier detection"),
    save: bool = typer.Option(True, help="Save figures"),
    show: bool = typer.Option(False, help="Show figures (may not work in headless environments)")
):
    """Run cross-model comparison analyzer"""
    typer.echo("Running cross-model comparison...")
    
    # Initialize analyzer with common options
    analyzer = CrossModelAnalyzer(
        dpi=dpi,
        cmap=cmap,
        figsize=figsize,
        remove_outliers=remove_outliers,
        outlier_params=build_outlier_params(metrics, outlier_method, outlier_threshold)
        if remove_outliers else None
    )
    
    # Filter by model type if requested
    if model_filter:
        model_types = [m.strip() for m in model_filter.split(",")]
        typer.echo(f"Filtering for model types: {', '.join(model_types)}")
        analyzer.df = analyzer.df[analyzer.df["model_type"].isin(model_types)]
        typer.echo(f"After filtering: {len(analyzer.df)} records remain")
    
    # Run analysis
    results = analyzer.run()
    
    # Handle display options
    if show:
        typer.echo("Showing figures...")
        plt.show()
    
    return results

@app.command()
def node2vec(
    metrics: List[str] = typer.Option(["acc/test"], help="Metrics for outlier removal"),
    dim_value: Optional[int] = typer.Option(None, help="Filter by specific dimension"),
    show_seed_variance: bool = typer.Option(True, help="Show seed variance analysis"),
    dpi: int = typer.Option(300, help="DPI for saved figures"),
    cmap: ColorMap = typer.Option(ColorMap.VIRIDIS, help="Colormap to use for plots"),
    figsize: str = typer.Option("12,8", help="Figure size as width,height", callback=parse_figsize),
    remove_outliers: bool = typer.Option(False, help="Whether to remove outliers"),
    outlier_method: OutlierMethod = typer.Option(OutlierMethod.IQR, help="Method for outlier detection"),
    outlier_threshold: float = typer.Option(1.5, help="Threshold for outlier detection"),
    save: bool = typer.Option(True, help="Save figures"),
    show: bool = typer.Option(False, help="Show figures (may not work in headless environments)")
):
    """Run Node2Vec analyzer"""
    typer.echo("Running Node2Vec analyzer...")
    
    # Initialize analyzer with common options
    analyzer = Node2VecAnalyzer(
        dpi=dpi,
        cmap=cmap,
        figsize=figsize,
        remove_outliers=remove_outliers,
        outlier_params=build_outlier_params(metrics, outlier_method, outlier_threshold)
        if remove_outliers else None
    )
    
    # Filter by dimension if requested
    if dim_value is not None:
        typer.echo(f"Filtering for dimension: {dim_value}")
        analyzer.df = analyzer.df[analyzer.df["dim"] == dim_value]
        typer.echo(f"After filtering: {len(analyzer.df)} records remain")
    
    # Run analysis
    results = analyzer.run()
    
    # Handle display options
    if show:
        typer.echo("Showing figures...")
        plt.show()
    
    return results

@app.command()
def textual(
    metrics: List[str] = typer.Option(["acc/test"], help="Metrics for outlier removal"),
    model_name: Optional[str] = typer.Option(None, help="Filter by specific model name"),
    dpi: int = typer.Option(300, help="DPI for saved figures"),
    cmap: ColorMap = typer.Option(ColorMap.VIRIDIS, help="Colormap to use for plots"),
    figsize: str = typer.Option("12,8", help="Figure size as width,height", callback=parse_figsize),
    remove_outliers: bool = typer.Option(False, help="Whether to remove outliers"),
    outlier_method: OutlierMethod = typer.Option(OutlierMethod.IQR, help="Method for outlier detection"),
    outlier_threshold: float = typer.Option(1.5, help="Threshold for outlier detection"),
    save: bool = typer.Option(True, help="Save figures"),
    show: bool = typer.Option(False, help="Show figures (may not work in headless environments)")
):
    """Run Textual Embeddings analyzer"""
    typer.echo("Running Textual Embeddings analyzer...")
    
    # Initialize analyzer with common options
    analyzer = TextualEmbeddingsAnalyzer(
        dpi=dpi,
        cmap=cmap,
        figsize=figsize,
        remove_outliers=remove_outliers,
        outlier_params=build_outlier_params(metrics, outlier_method, outlier_threshold)
        if remove_outliers else None
    )
    
    # Filter by model name if requested
    if model_name:
        typer.echo(f"Filtering for model: {model_name}")
        analyzer.df = analyzer.df[analyzer.df["model"] == model_name]
        typer.echo(f"After filtering: {len(analyzer.df)} records remain")
    
    # Run analysis
    results = analyzer.run()
    
    # Find and report best model
    if not analyzer.df.empty:
        best_model = analyzer.find_best_model()
        typer.echo(f"Best model: {best_model['model']} "
                  f"(dimension: {best_model.get('embedding_dim', 'N/A')}) "
                  f"with accuracy: {best_model.get('acc/test', 0):.4f}")
    
    # Handle display options
    if show:
        typer.echo("Showing figures...")
        plt.show()
    
    return results

@app.command()
def fusion(
    fusion_type: str = typer.Argument(..., help="Fusion type (addition, early, gated, lowrank, transformer)"),
    metrics: List[str] = typer.Option(["acc/test"], help="Metrics for outlier removal"),
    latent_dim: Optional[int] = typer.Option(None, help="Filter by specific latent dimension"),
    textual_model: Optional[str] = typer.Option(None, help="Filter by specific textual model"),
    relational_model: Optional[str] = typer.Option(None, help="Filter by specific relational model"),
    dpi: int = typer.Option(300, help="DPI for saved figures"),
    cmap: ColorMap = typer.Option(ColorMap.VIRIDIS, help="Colormap to use for plots"),
    figsize: str = typer.Option("12,8", help="Figure size as width,height", callback=parse_figsize),
    remove_outliers: bool = typer.Option(False, help="Whether to remove outliers"),
    outlier_method: OutlierMethod = typer.Option(OutlierMethod.IQR, help="Method for outlier detection"),
    outlier_threshold: float = typer.Option(1.5, help="Threshold for outlier detection"),
    save: bool = typer.Option(True, help="Save figures"),
    show: bool = typer.Option(False, help="Show figures (may not work in headless environments)")
):
    """Run a specific fusion analyzer"""
    # Map of fusion type names to analyzers
    fusion_analyzers = {
        "addition": AdditionFusionAnalyzer,
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
    
    # Get the appropriate analyzer class
    analyzer_class = fusion_analyzers[fusion_type.lower()]
    
    # Initialize analyzer with common options
    analyzer = analyzer_class(
        dpi=dpi,
        cmap=cmap,
        figsize=figsize,
        remove_outliers=remove_outliers,
        outlier_params=build_outlier_params(metrics, outlier_method, outlier_threshold)
        if remove_outliers else None
    )
    
    # Apply filters if requested
    filters = []
    if latent_dim is not None:
        filters.append(f"latent_dim == '{latent_dim}'")
    if textual_model is not None:
        filters.append(f"textual_name == '{textual_model}'")
    if relational_model is not None:
        filters.append(f"relational_name == '{relational_model}'")
    
    if filters:
        query = " and ".join(filters)
        typer.echo(f"Applying filters: {query}")
        analyzer.df = analyzer.df.query(query)
        typer.echo(f"After filtering: {len(analyzer.df)} records remain")
    
    # Run analysis
    results = analyzer.run()
    
    # Find and report best configuration
    if not analyzer.df.empty:
        best_idx = analyzer.df["acc/test"].idxmax()
        best_config = analyzer.df.loc[best_idx]
        typer.echo("\nBest configuration:")
        for col in ["textual_name", "relational_name", "textual_dim", "relational_dim", "latent_dim", "acc/test"]:
            if col in best_config:
                typer.echo(f"  {col}: {best_config[col]}")
    
    # Handle display options
    if show:
        typer.echo("Showing figures...")
        plt.show()
    
    return results

@app.command("best-hard-lp")
def find_best_hard_lp(
    n_top: int = typer.Option(5, help="Number of top models to show"),
    dpi: int = typer.Option(300, help="DPI for saved figures"),
    cmap: ColorMap = typer.Option(ColorMap.VIRIDIS, help="Colormap to use for plots"),
    figsize: str = typer.Option("12,8", help="Figure size as width,height", callback=parse_figsize),
    remove_outliers: bool = typer.Option(False, help="Whether to remove outliers"),
    outlier_method: OutlierMethod = typer.Option(OutlierMethod.IQR, help="Method for outlier detection"),
    outlier_threshold: float = typer.Option(1.5, help="Threshold for outlier detection"),
    save: bool = typer.Option(True, help="Save figures"),
    show: bool = typer.Option(False, help="Show figures (may not work in headless environments)")
):
    """Find the trial with the highest hard link prediction AUC score"""
    typer.echo(f"Finding the {n_top} trials with highest hard link prediction AUC...")
    
    # Initialize cross-model analyzer
    analyzer = CrossModelAnalyzer(
        dpi=dpi,
        cmap=cmap,
        figsize=figsize
    )
    
    # Check if we have hard LP data
    if 'lp_hard/auc' not in analyzer.df.columns:
        typer.echo("No hard link prediction data found in the analysis results")
        raise typer.Exit(1)
    
    # Filter rows with hard LP data
    valid_rows = analyzer.df[~analyzer.df['lp_hard/auc'].isna()]
    if len(valid_rows) == 0:
        typer.echo("No valid hard link prediction results found")
        raise typer.Exit(1)
    
    # Get top models
    top_models = valid_rows.nlargest(n_top, 'lp_hard/auc')
    
    # Display results
    typer.echo("\n=== Top Models for Hard Link Prediction ===")
    
    # Print in table format
    headers = ["Model Type", "LP Hard AUC", "Path"]
    # Get max width for each column
    col_widths = [max(len(h), max(len(str(row[1]["model_type"])) for row in top_models.iterrows())) for h in headers]
    col_widths[2] = min(80, max(len(h), max(len(str(row[1]["path"])) for row in top_models.iterrows())))
    
    # Print header
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    typer.echo(header_line)
    typer.echo("-" * len(header_line))
    
    # Print rows
    for _, row in top_models.iterrows():
        model_type = str(row["model_type"])
        lp_score = f"{row['lp_hard/auc']:.4f}"
        path = str(row["path"])
        
        # Truncate path if too long
        if len(path) > col_widths[2]:
            path = path[:col_widths[2]-3] + "..."
            
        row_str = f"{model_type.ljust(col_widths[0])} | {lp_score.ljust(col_widths[1])} | {path.ljust(col_widths[2])}"
        typer.echo(row_str)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar chart
    bar_colors = plt.cm.viridis(np.linspace(0, 1, len(top_models)))
    bars = ax.bar(
        top_models["model_type"], 
        top_models["lp_hard/auc"], 
        color=bar_colors
    )
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            height + 0.001,
            f'{height:.4f}', 
            ha='center', 
            va='bottom',
            fontsize=10
        )
    
    ax.set_title('Top Models by Hard Link Prediction AUC')
    ax.set_ylabel('Hard LP AUC Score')
    ax.set_ylim(top_models['lp_hard/auc'].min() - 0.01, top_models['lp_hard/auc'].max() + 0.01)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save figure if requested
    if save:
        from constants import FIGURE_PATH
        save_path = FIGURE_PATH / "CrossModelAnalyzer/top_hard_lp_models.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        typer.echo(f"Saved figure to {save_path}")
    
    # Show if requested
    if show:
        plt.show()
    
    return top_models

# Helper for building outlier parameters
def build_outlier_params(metrics: List[str], method: OutlierMethod, threshold: float) -> dict:
    """Build outlier parameters dictionary"""
    return {
        "metrics": metrics,
        "method": method.value,
        "threshold": threshold,
        "keep_filtered": True  # Always keep filtered outliers for reference
    }

if __name__ == "__main__":
    # Set headless backend if not showing plots
    if "--show" not in os.sys.argv and "--no-show" not in os.sys.argv:
        matplotlib.use("Agg")
    
    app()
