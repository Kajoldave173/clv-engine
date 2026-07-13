"""CLI entrypoint for the CLV pipeline.

Usage:
    python -m src.cli data --config configs/params.yaml
    python -m src.cli validate --config configs/params.yaml
    python -m src.cli features --config configs/params.yaml
    python -m src.cli train --config configs/params.yaml
    python -m src.cli evaluate --config configs/params.yaml
    python -m src.cli segment --config configs/params.yaml
    python -m src.cli score --config configs/params.yaml
"""

import typer
from pathlib import Path
from rich.console import Console

from src.utils.io import load_config

app = typer.Typer(
    name="clv",
    help="CLV Prediction & Segmentation Engine — production ML pipeline.",
    add_completion=False,
)
console = Console()


@app.command()
def data(
    config: Path = typer.Option("configs/params.yaml", help="Path to params.yaml"),
) -> None:
    """Download, clean, and split the transaction data."""
    cfg = load_config(config)
    console.print("[bold blue]Phase 1:[/] Data ingestion and cleaning")

    from src.data.ingest import ingest_data
    from src.data.clean import clean_transactions

    raw_df = ingest_data(cfg)
    console.print(f"  Loaded {len(raw_df):,} raw transactions")

    clean_df = clean_transactions(raw_df, cfg)
    console.print(f"  Cleaned to {len(clean_df):,} transactions")
    console.print("[bold green]✓[/] Data pipeline complete")


@app.command()
def validate(
    config: Path = typer.Option("configs/params.yaml", help="Path to params.yaml"),
) -> None:
    """Run data validation checks on cleaned data."""
    cfg = load_config(config)
    console.print("[bold blue]Validation:[/] Running data quality checks")

    from src.data.validate import run_validation

    report = run_validation(cfg)
    console.print(report)
    console.print("[bold green]✓[/] Validation complete")


@app.command()
def features(
    config: Path = typer.Option("configs/params.yaml", help="Path to params.yaml"),
) -> None:
    """Build RFM and behavioral feature matrices."""
    cfg = load_config(config)
    console.print("[bold blue]Phase 2/3:[/] Feature engineering")

    from src.features.build_features import build_all_features

    feature_df = build_all_features(cfg)
    console.print(f"  Built {feature_df.shape[1]} features for {len(feature_df):,} customers")
    console.print("[bold green]✓[/] Feature pipeline complete")


@app.command()
def train(
    config: Path = typer.Option("configs/params.yaml", help="Path to params.yaml"),
) -> None:
    """Train all models: probabilistic, XGBoost CLV, churn classifier."""
    cfg = load_config(config)
    console.print("[bold blue]Phase 2/3:[/] Model training")

    from src.models.train import train_all_models

    results = train_all_models(cfg)
    console.print("[bold green]✓[/] All models trained and saved")


@app.command()
def evaluate(
    config: Path = typer.Option("configs/params.yaml", help="Path to params.yaml"),
) -> None:
    """Evaluate models against holdout and generate SHAP analysis."""
    cfg = load_config(config)
    console.print("[bold blue]Phase 4:[/] Model evaluation + SHAP")

    from src.evaluation.metrics import evaluate_models
    from src.evaluation.shap_analysis import run_shap_analysis

    evaluate_models(cfg)
    run_shap_analysis(cfg)
    console.print("[bold green]✓[/] Evaluation complete")


@app.command()
def segment(
    config: Path = typer.Option("configs/params.yaml", help="Path to params.yaml"),
) -> None:
    """Run behavioral segmentation via clustering."""
    cfg = load_config(config)
    console.print("[bold blue]Phase 4:[/] Customer segmentation")

    from src.segmentation.cluster import run_segmentation

    segments = run_segmentation(cfg)
    console.print(f"  Assigned {len(segments):,} customers to clusters")
    console.print("[bold green]✓[/] Segmentation complete")


@app.command()
def score(
    config: Path = typer.Option("configs/params.yaml", help="Path to params.yaml"),
    input_file: Path = typer.Option(None, "--input", help="Path to new transaction CSV"),
    output_dir: Path = typer.Option("data/predictions/", "--output", help="Output directory"),
) -> None:
    """Batch score all customers: features → predict → segment → output."""
    cfg = load_config(config)
    console.print("[bold blue]Phase 5:[/] Batch scoring pipeline")

    from src.scoring.batch_scorer import batch_score

    scored = batch_score(cfg, input_file=input_file, output_dir=output_dir)
    console.print(f"  Scored {len(scored):,} customers")
    console.print("[bold green]✓[/] Batch scoring complete")


if __name__ == "__main__":
    app()
