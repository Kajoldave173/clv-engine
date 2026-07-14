"""MLflow configuration and logging utilities.

Handles DagsHub connection, authentication, and provides helper
functions for logging params, metrics, and artifacts. Falls back
to local mlruns/ directory if DagsHub credentials aren't configured.

Usage:
    from src.experiment.mlflow_config import setup_mlflow, log_params, log_metrics

    setup_mlflow(config)  # Call once at start

    with mlflow.start_run(run_name="baseline-xgboost"):
        log_params({"n_estimators": 500, "max_depth": 6})
        log_metrics({"rmse": 1807.97, "mae": 509.44})
        log_model_artifact("models/xgb_clv_model.json")
"""

import os
import logging
from pathlib import Path
from typing import Any

import mlflow

from src.utils.io import PROJECT_ROOT

logger = logging.getLogger(__name__)


def setup_mlflow(config: dict[str, Any]) -> str:
    """Configure MLflow tracking with DagsHub or local fallback.

    Reads tracking URI from params.yaml and token from environment.
    If DagsHub credentials aren't available, falls back to local
    mlruns/ directory so the pipeline never crashes due to missing
    experiment tracking.

    Args:
        config: Pipeline configuration dictionary (needs config["mlflow"]).

    Returns:
        The active tracking URI (DagsHub URL or local path).
    """
    mlflow_config = config.get("mlflow", {})
    tracking_uri = mlflow_config.get("tracking_uri", "")
    experiment_name = mlflow_config.get("experiment_name", "clv-prediction")

    # Try DagsHub authentication via token
    dagshub_token = os.environ.get("DAGSHUB_USER_TOKEN", "")

    if tracking_uri and dagshub_token:
        # Authenticate with DagsHub
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking: DagsHub ({tracking_uri})")
        print(f"  MLflow tracking: DagsHub ({tracking_uri})")

    elif tracking_uri and not dagshub_token:
        # URI configured but no token — fall back to local
        local_uri = str(PROJECT_ROOT / "mlruns")
        mlflow.set_tracking_uri(local_uri)
        tracking_uri = local_uri
        logger.warning(
            "DAGSHUB_USER_TOKEN not set. Falling back to local MLflow tracking. "
            "Set the token in your .env file for DagsHub logging."
        )
        print(
            "  MLflow tracking: LOCAL (set DAGSHUB_USER_TOKEN in .env for DagsHub)"
        )

    else:
        # No MLflow config at all — local fallback
        local_uri = str(PROJECT_ROOT / "mlruns")
        mlflow.set_tracking_uri(local_uri)
        tracking_uri = local_uri
        print("  MLflow tracking: LOCAL (mlruns/)")

    # Set or create the experiment
    mlflow.set_experiment(experiment_name)
    print(f"  MLflow experiment: {experiment_name}")

    return tracking_uri


def load_env_file() -> None:
    """Load environment variables from .env file if it exists.

    Simple implementation that doesn't require python-dotenv.
    Reads key=value pairs, ignoring comments and blank lines.
    """
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            # Parse key=value
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                # Don't overwrite existing env vars
                if key and key not in os.environ:
                    os.environ[key] = value


def log_params(params: dict[str, Any]) -> None:
    """Log a dictionary of parameters to the active MLflow run.

    Handles type conversion — MLflow only accepts strings, ints,
    floats, and bools. Lists and dicts get converted to strings.

    Args:
        params: Dictionary of parameter names and values.
    """
    clean = {}
    for key, value in params.items():
        if isinstance(value, (list, dict)):
            clean[key] = str(value)
        else:
            clean[key] = value
    mlflow.log_params(clean)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log a dictionary of metrics to the active MLflow run.

    Args:
        metrics: Dictionary of metric names and values.
        step: Optional step number (for iterative metrics).
    """
    for key, value in metrics.items():
        if value is not None:
            mlflow.log_metric(key, float(value), step=step)


def log_model_artifact(path: str | Path) -> None:
    """Log a model file or directory as an MLflow artifact.

    Args:
        path: Path to the file or directory to log.
    """
    path = Path(path)
    if path.exists():
        mlflow.log_artifact(str(path))
    else:
        logger.warning(f"Artifact not found, skipping: {path}")


def log_figure(fig, filename: str) -> None:
    """Log a matplotlib figure as an MLflow artifact.

    Saves the figure to a temp location, logs it, then cleans up.

    Args:
        fig: Matplotlib figure object.
        filename: Name for the artifact file (e.g. "roc_curve.png").
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / filename
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        mlflow.log_artifact(str(filepath))


def log_config_as_params(config: dict[str, Any], prefix: str = "") -> None:
    """Recursively log a nested config dict as flat MLflow params.

    Nested keys are joined with dots: {"models": {"xgboost": {"lr": 0.05}}}
    becomes {"models.xgboost.lr": 0.05}.

    Args:
        config: Configuration dictionary (can be nested).
        prefix: Key prefix for recursion (internal use).
    """
    flat = {}

    def _flatten(d: dict, pre: str) -> None:
        for key, value in d.items():
            full_key = f"{pre}.{key}" if pre else key
            if isinstance(value, dict):
                _flatten(value, full_key)
            else:
                flat[full_key] = value

    _flatten(config, prefix)

    # MLflow has a 500-param limit; truncate key names if needed
    truncated = {k[:250]: v for k, v in flat.items()}
    log_params(truncated)
