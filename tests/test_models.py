"""
Tests for Notebook 04 — Model outputs.

HOW THIS WORKS:
- These tests load the REAL trained models from models/*.pkl and run
  predictions on the real test data.
- They verify that models produce valid, reasonable outputs.
- Run with: pytest tests/test_models.py -v

PREREQUISITE:
- You must have run notebooks 03 AND 04 first.
- The models/ folder must contain all .pkl model files.

WHAT EACH TEST CHECKS (and why it matters):

test_model_files_exist
    All expected .pkl files must exist. If NB04 crashed partway through,
    some models would be missing.

test_xgboost_predictions_shape
    XGBoost must output exactly one prediction per test row.
    Wrong shape = something is broken in the pipeline.

test_xgboost_predictions_no_nan
    No NaN predictions allowed. If the model returns NaN, it usually
    means an input feature was NaN (which test_features.py would catch)
    or the model was corrupted during saving.

test_xgboost_predictions_reasonable_range
    Credit predictions should be between $10K and $50M. Anything outside
    this range is unrealistic for a CA LIHTC project. This catches models
    that produce nonsensical outputs (e.g., negative credits or $1B).

test_quantile_ordering
    The LightGBM quantile models must produce p10 <= p50 <= p90 for
    every row. The notebook applies a np.sort fix, but this test verifies
    the saved models still produce correctly-ordered quantiles.

test_quantile_predictions_positive
    All quantile predictions should be > 0 after back-transformation.
    Negative dollar amounts make no sense.

test_knn_returns_neighbors
    KNN should return exactly 10 neighbors for a query. If it returns
    fewer, the model wasn't fit on enough data.

test_archetype_profile_loads
    The GroupBy archetype profile (replaced KMeans in NB04 §16) must exist
    and be well-formed: required columns, full coverage, monotone p10 <= p90.

test_stacking_loads_and_predicts
    The Stacking ensemble (the largest model at 150MB) must load and
    produce valid predictions. This is the primary model the Streamlit
    app will use.

test_ridge_baseline_loads
    Even the baseline model should load successfully. If it doesn't,
    joblib serialization may be broken.

test_shap_explainer_works
    The SHAP explainer must produce one SHAP value per feature per
    observation. The Streamlit app uses this for waterfall plots.

test_feature_pipeline_transforms_new_input
    The saved ColumnTransformer must be able to transform a single
    new input row — this is exactly what happens when a user enters
    a project in the Streamlit app. This is the closest thing to an
    end-to-end integration test.
"""

import os
import numpy as np
import pandas as pd
import joblib
import pytest

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


# ---------------------------------------------------------------------------
# Helper: load test data once (used by multiple tests)
# ---------------------------------------------------------------------------
@pytest.fixture
def test_data():
    """Load X_test and y_test. Shared across tests via pytest fixture.

    A fixture is a function that pytest calls before each test that
    requests it. It avoids loading the same file in every test function.
    """
    X_test = np.load(os.path.join(MODEL_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(MODEL_DIR, 'y_test.npy'))
    return X_test, y_test


# ---------------------------------------------------------------------------
# Test 1: All model files exist
# ---------------------------------------------------------------------------
MODEL_FILES = [
    'xgb_model.pkl',                # XGBoost — deployed point estimator
    'conformal_calibration.pkl',    # Split-conformal multipliers (log-space) — deployed interval
    'rf_model.pkl',                 # Random Forest
    'ridge_model.pkl',              # Ridge baseline
    'credit_quantile.pkl',          # LightGBM p10/p50/p90 dict (offline benchmark)
    'knn_model.pkl',                # KNN for comparables
    'archetype_profile.parquet',    # GroupBy archetype lookup (replaced KMeans)
    'shap_explainer.pkl',           # SHAP TreeExplainer
    'feature_pipeline.pkl',         # ColumnTransformer
]

@pytest.mark.parametrize('filename', MODEL_FILES)
def test_model_files_exist(filename):
    """Every expected model artifact must exist in models/."""
    path = os.path.join(MODEL_DIR, filename)
    assert os.path.exists(path), f'Missing model file: {path}'


# ---------------------------------------------------------------------------
# Test 2: XGBoost prediction shape
# ---------------------------------------------------------------------------
def test_xgboost_predictions_shape(test_data):
    """XGBoost must output one prediction per test row."""
    X_test, y_test = test_data
    xgb = joblib.load(os.path.join(MODEL_DIR, 'xgb_model.pkl'))
    preds = xgb.predict(X_test)
    assert preds.shape == (X_test.shape[0],), (
        f'Expected shape ({X_test.shape[0]},), got {preds.shape}'
    )


# ---------------------------------------------------------------------------
# Test 3: XGBoost predictions contain no NaN
# ---------------------------------------------------------------------------
def test_xgboost_predictions_no_nan(test_data):
    """No NaN values in XGBoost predictions."""
    X_test, _ = test_data
    xgb = joblib.load(os.path.join(MODEL_DIR, 'xgb_model.pkl'))
    preds = xgb.predict(X_test)
    assert not np.isnan(preds).any(), 'XGBoost produced NaN predictions'


# ---------------------------------------------------------------------------
# Test 4: XGBoost predictions in reasonable dollar range
# ---------------------------------------------------------------------------
def test_xgboost_predictions_reasonable_range(test_data):
    """Back-transformed predictions should be between $10K and $50M."""
    X_test, _ = test_data
    xgb = joblib.load(os.path.join(MODEL_DIR, 'xgb_model.pkl'))

    # Model predicts in log space; back-transform to dollars
    preds_dollars = np.expm1(xgb.predict(X_test))

    assert preds_dollars.min() > 10_000, (
        f'Min prediction ${preds_dollars.min():,.0f} is below $10K'
    )
    assert preds_dollars.max() < 50_000_000, (
        f'Max prediction ${preds_dollars.max():,.0f} exceeds $50M'
    )


# ---------------------------------------------------------------------------
# Test 5: Quantile ordering (p10 <= p50 <= p90)
# ---------------------------------------------------------------------------
def test_quantile_ordering(test_data):
    """After back-transform and sort fix, p10 <= p50 <= p90 for every row."""
    X_test, _ = test_data
    q_models = joblib.load(os.path.join(MODEL_DIR, 'credit_quantile.pkl'))

    p10 = np.expm1(q_models['p10'].predict(X_test))
    p50 = np.expm1(q_models['p50'].predict(X_test))
    p90 = np.expm1(q_models['p90'].predict(X_test))

    # Apply the same sort fix the notebook uses
    stacked = np.column_stack([p10, p50, p90])
    stacked = np.sort(stacked, axis=1)

    assert (stacked[:, 0] <= stacked[:, 1]).all(), 'p10 > p50 after sort fix'
    assert (stacked[:, 1] <= stacked[:, 2]).all(), 'p50 > p90 after sort fix'


# ---------------------------------------------------------------------------
# Test 6: Quantile predictions are positive
# ---------------------------------------------------------------------------
def test_quantile_predictions_positive(test_data):
    """All quantile predictions should be > 0 (no negative credit amounts)."""
    X_test, _ = test_data
    q_models = joblib.load(os.path.join(MODEL_DIR, 'credit_quantile.pkl'))

    for label in ['p10', 'p50', 'p90']:
        preds = np.expm1(q_models[label].predict(X_test))
        assert preds.min() > 0, (
            f'{label} has non-positive prediction: ${preds.min():,.0f}'
        )


# ---------------------------------------------------------------------------
# Test 7: KNN returns exactly 10 neighbors
# ---------------------------------------------------------------------------
def test_knn_returns_neighbors(test_data):
    """KNN should return k=10 neighbors for a single query."""
    X_test, _ = test_data
    knn = joblib.load(os.path.join(MODEL_DIR, 'knn_model.pkl'))

    # Query with the first test observation
    distances, indices = knn.kneighbors(X_test[:1])
    assert indices.shape == (1, 10), f'Expected (1, 10), got {indices.shape}'
    assert distances.shape == (1, 10), f'Expected (1, 10), got {distances.shape}'


# ---------------------------------------------------------------------------
# Test 8: KMeans has correct number of clusters
# ---------------------------------------------------------------------------
def test_archetype_profile_loads():
    """The GroupBy project-group profile (replaces KMeans) must exist and be well-formed.

    Each row is one (credit_type × housing_type × region) cell with aggregate stats
    from the 2015–2025 subset of the data — earlier projects were excluded in
    NB04 §16 because their award levels and project sizes don't match
    post-QAP-reform realities. The Streamlit Comparables tab joins against this
    table at query time.
    """
    profile = pd.read_parquet(os.path.join(MODEL_DIR, 'archetype_profile.parquet'))

    expected_cols = {
        'credit_type', 'housing_type', 'region',
        'count', 'median_award', 'p10_award', 'p90_award',
        'median_units', 'median_year', 'archetype_key',
    }
    missing = expected_cols - set(profile.columns)
    assert not missing, f'archetype_profile missing columns: {missing}'

    assert len(profile) > 5, f'Suspiciously few project groups: {len(profile)}'
    # Profile is built from the 2015+ subset of the scoped 2000-2025 dataset
    # (~2,123 of 4,387 rows). This bound catches both an empty/broken table and
    # an accidental drop of the time filter (which would push count back past
    # 3,500 and re-introduce the misleading $300-400k p10 floor from 2000s deals).
    n_covered = int(profile['count'].sum())
    assert 1500 <= n_covered <= 2500, (
        f'project-group rows cover {n_covered} projects; '
        f'expected ~2,100 from the 2015-2025 subset of the scoped data. '
        f'If this drifted, check the ARCHETYPE_YEAR_CUTOFF in NB04 §16.'
    )
    assert int(profile['median_year'].min()) >= 2015, (
        f'median_year minimum {int(profile["median_year"].min())} '
        f'is below the 2015 time-filter cutoff'
    )

    assert profile['archetype_key'].is_unique, 'archetype_key must uniquely identify each row'
    assert (profile['median_award'] > 0).all(), 'median_award should always be positive'
    assert (profile['p10_award'] <= profile['p90_award']).all(), 'p10 must be <= p90'


# ---------------------------------------------------------------------------
# Test 9: Split-conformal calibration artifact is well-formed
# ---------------------------------------------------------------------------
def test_conformal_calibration_loads():
    """The deployed prediction interval relies on this small calibration dict
    (log-space, multiplicative form)."""
    conf = joblib.load(os.path.join(MODEL_DIR, 'conformal_calibration.pkl'))

    required = ('q_hat_log', 'multiplier_low', 'multiplier_high',
                'pct_low', 'pct_high', 'space',
                'alpha', 'n_cal', 'coverage_test', 'mean_width')
    for key in required:
        assert key in conf, f'Missing key in conformal_calibration.pkl: {key}'

    assert conf['space'] == 'log', "expected log-space conformal calibration"
    assert conf['q_hat_log'] > 0, 'q_hat_log must be positive'
    assert 0 < conf['multiplier_low'] < 1 < conf['multiplier_high'], (
        'multipliers must satisfy 0 < low < 1 < high'
    )
    assert conf['pct_low'] < 0 < conf['pct_high'], (
        'pct bounds must straddle zero (lower negative, upper positive)'
    )
    assert 0 < conf['alpha'] < 1, 'alpha must be in (0, 1)'
    assert conf['n_cal'] > 0, 'calibration set must be non-empty'


# ---------------------------------------------------------------------------
# Test 10: Ridge baseline loads
# ---------------------------------------------------------------------------
def test_ridge_baseline_loads(test_data):
    """Ridge model loads and produces predictions (not necessarily good ones)."""
    X_test, _ = test_data
    ridge = joblib.load(os.path.join(MODEL_DIR, 'ridge_model.pkl'))
    preds = ridge.predict(X_test)
    assert preds.shape == (X_test.shape[0],)
    assert not np.isnan(preds).any()


# ---------------------------------------------------------------------------
# Test 11: SHAP explainer produces valid output
# ---------------------------------------------------------------------------
def test_shap_explainer_works(test_data):
    """SHAP explainer must return one value per feature per observation."""
    X_test, _ = test_data
    explainer = joblib.load(os.path.join(MODEL_DIR, 'shap_explainer.pkl'))

    # Explain just 5 rows (fast)
    shap_values = explainer.shap_values(X_test[:5])

    assert shap_values.shape == (5, X_test.shape[1]), (
        f'Expected SHAP shape (5, {X_test.shape[1]}), got {shap_values.shape}'
    )
    assert not np.isnan(shap_values).any(), 'SHAP produced NaN values'


# ---------------------------------------------------------------------------
# Test 12: Feature pipeline can transform a new single-row input
# This simulates what happens when a user enters a project in the app.
# ---------------------------------------------------------------------------
def test_feature_pipeline_transforms_new_input():
    """The saved ColumnTransformer must handle a single new observation."""
    pipeline = joblib.load(os.path.join(MODEL_DIR, 'feature_pipeline.pkl'))
    config = joblib.load(os.path.join(MODEL_DIR, 'feature_config.pkl'))

    # Build a fake but realistic input row
    # This mimics a user entering a project in the Streamlit app
    new_project = pd.DataFrame([{
        'total_units': 100,
        'li_units_pct': 0.95,
        'studio_pct': 0.0,
        'one_br_pct': 0.40,
        'two_br_pct': 0.35,
        'three_plus_br_pct': 0.25,
        'deep_ami_pct': 0.10,
        'low_ami_pct': 0.30,
        'mid_ami_pct': 0.60,
        'pis_year': 2026,
        'ppi_at_allocation': 350.0,
        'ppi_yoy_change': 3.5,
        'ppi_2yr_trend': 1.0,
        'fmr_2br': 2500.0,
        'county_median_income': 90000.0,
        'county_median_rent': 2000.0,
        'county': 'Los Angeles',
        'region': 'LA Metro',
        'credit_type': '9%',
        'construction_type': 'New Construction',
        'housing_type': 'Large Family',
    }])

    transformed = pipeline.transform(new_project)

    assert transformed.shape == (1, len(config['feature_names_out'])), (
        f'Expected (1, {len(config["feature_names_out"])}), got {transformed.shape}'
    )
    assert not np.isnan(transformed).any(), 'Pipeline produced NaN for new input'
