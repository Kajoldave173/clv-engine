"""Orchestrates the full feature pipeline.

Phase 2: RFM summary table (from rfm.py)
Phase 3: Behavioral features (from behavioral.py)

The two feature sets are saved separately. They get merged with
probabilistic model predictions during the training step, since
the probabilistic models must be fitted before their outputs
(P(alive), predicted_purchases) can be used as ML features.
"""

from typing import Any

import pandas as pd

from src.features.rfm import build_rfm_summary
from src.features.behavioral import build_behavioral_features


def build_all_features(config: dict[str, Any]) -> pd.DataFrame:
    """Build all feature matrices.

    Builds the RFM summary table and behavioral features from
    calibration transactions. Each is saved to data/processed/.

    Args:
        config: Pipeline configuration dictionary.

    Returns:
        Behavioral feature DataFrame (RFM saved separately).
    """
    # Phase 2.1: RFM summary
    print("=" * 50)
    print("RFM SUMMARY TABLE")
    print("=" * 50)
    rfm = build_rfm_summary(config)

    # Phase 3.1: Behavioral features
    print("\n" + "=" * 50)
    print("BEHAVIORAL FEATURES")
    print("=" * 50)
    behavioral = build_behavioral_features(config)

    print(f"\nFeature pipeline complete:")
    print(f"  RFM:        {rfm.shape[1]} features, {len(rfm):,} customers")
    print(f"  Behavioral: {behavioral.shape[1]} features, {len(behavioral):,} customers")

    return behavioral