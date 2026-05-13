"""
Smoke tests for Tab 4 (Method Insight) helpers.

Run:  pytest tests/test_method_insight.py -v
"""

import os
import re
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from streamlit_app import (
    MODEL_DIR,
    compute_global_shap,
    compute_test_predictions,
    load_project_splits,
)


@pytest.fixture(scope="module")
def models():
    m = {}
    m["xgb"] = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
    m["conformal"] = joblib.load(os.path.join(MODEL_DIR, "conformal_calibration.pkl"))
    m["pipeline"] = joblib.load(os.path.join(MODEL_DIR, "feature_pipeline.pkl"))
    m["config"] = joblib.load(os.path.join(MODEL_DIR, "feature_config.pkl"))
    m["shap"] = joblib.load(os.path.join(MODEL_DIR, "shap_explainer.pkl"))
    return m


@pytest.fixture(scope="module")
def splits():
    df_train, df_test = load_project_splits()
    return df_train, df_test


def test_compute_test_predictions_columns(models, splits):
    """compute_test_predictions returns the 7 columns Tab 4 charts read."""
    _, df_test = splits
    preds = compute_test_predictions(models, df_test)

    expected = {"pis_year", "actual", "predicted", "log_residual", "lower", "upper", "in_interval"}
    assert set(preds.columns) == expected
    assert len(preds) == len(df_test)
    assert (preds["lower"] < preds["upper"]).all(), "Conformal lower must be < upper"
    assert preds["in_interval"].dtype == bool

    # Realised coverage on the test set should be within ±5pp of the value
    # baked into the conformal artifact (which itself was measured on this set).
    baked_coverage = models["conformal"]["coverage_test"]
    measured_coverage = preds["in_interval"].mean() * 100
    assert abs(measured_coverage - baked_coverage) < 5.0, (
        f"Coverage drift: measured {measured_coverage:.1f}% vs baked {baked_coverage:.1f}%"
    )


def test_compute_global_shap_columns(models, splits):
    """compute_global_shap returns a sorted (feature, mean_abs_shap, display_name) frame."""
    _, df_test = splits
    shap_df = compute_global_shap(models, df_test, n_sample=50)  # small n for speed

    assert {"feature", "mean_abs_shap", "display_name"}.issubset(shap_df.columns)
    assert (shap_df["mean_abs_shap"] >= 0).all(), "Mean |SHAP| must be non-negative"
    diffs = np.diff(shap_df["mean_abs_shap"].values)
    assert (diffs <= 1e-12).all(), "Output must be sorted descending"
    # Column-transformer prefixes must be stripped from display names.
    assert not shap_df["display_name"].str.startswith(("Num__", "Te__", "Ohe__")).any()


def test_render_trends_has_no_deep_ami():
    """Tab 3 should no longer reference 'Deep Affordability' or 'deep_ami_pct'
    inside render_trends — that subsection was removed when Method Insight landed."""
    app_path = os.path.join(os.path.dirname(__file__), "..", "streamlit_app.py")
    with open(app_path) as f:
        src = f.read()

    # Slice out the body of render_trends.
    start = src.index("def render_trends(")
    end = src.index("\ndef ", start + 1)
    body = src[start:end]

    assert "Deep Affordability" not in body, "Deep Affordability subsection should be deleted"
    assert "deep_ami_pct" not in body, "deep_ami_pct aggregation should be deleted from Tab 3"
    # Sanity: the four kept charts are still there.
    for keep in ["Credit Type", "County Comparison", "Housing Type", "Project Volume"]:
        assert re.search(re.escape(keep), body), f"Expected '{keep}' to remain in Tab 3"
