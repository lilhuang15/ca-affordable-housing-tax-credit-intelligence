"""
Tests for Notebook 03 — Feature Engineering outputs.

HOW THIS WORKS:
- These tests load the REAL artifacts that notebook 03 saved to models/.
- They verify the pipeline produced correct, consistent outputs.
- Run with: pytest tests/test_features.py -v
- The -v flag means "verbose" — shows each test name and pass/fail.

PREREQUISITE:
- You must have run notebook 03_feature_engineering.ipynb first.
- The models/ folder must contain: X_train.npy, X_test.npy, y_train.npy,
  y_test.npy, feature_pipeline.pkl, feature_config.pkl, train_data.parquet,
  test_data.parquet.

WHAT EACH TEST CHECKS (and why it matters):

test_artifacts_exist
    The most basic check — do the files exist? If NB03 didn't run or
    crashed partway through, these would be missing.

test_train_test_shapes
    Train should have ~3,536 rows, test ~851 rows. If these numbers are
    wildly off, something went wrong with the scope filter or split.

test_feature_count_matches
    X_train and X_test must have the same number of columns. If they
    don't, the ColumnTransformer was applied inconsistently.

test_no_nan_in_transformed_data
    After imputation + encoding, there should be zero NaN values.
    NaN in transformed data would cause models to fail or produce NaN
    predictions.

test_train_test_no_year_overlap
    Train must be 2000–2021, test must be 2022–2025, with NO overlap.
    If any year appears in both, we have data leakage.

test_target_values_positive
    Annual federal award should always be > 0 (we dropped zero/missing
    in NB01). Negative values would indicate a data corruption.

test_percentage_features_in_range
    All derived percentage features (studio_pct, li_units_pct, etc.)
    should be in [0, 1] after the clamping in NB01. Values outside
    this range would mean the cleaning step failed.

test_feature_names_count
    The feature config should list the same number of output feature
    names as columns in the transformed arrays.

test_scaler_centered_train
    StandardScaler should make training features approximately mean=0.
    If the mean is far from 0, the scaler wasn't fit correctly.

test_no_duplicate_rows
    Check that the split didn't accidentally duplicate rows.
"""

import os
import numpy as np
import pandas as pd
import joblib
import pytest

# ---------------------------------------------------------------------------
# Path to the models/ directory (relative to project root)
# pytest runs from the project root by default
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


# ---------------------------------------------------------------------------
# Test 1: Do all required files exist?
# ---------------------------------------------------------------------------
REQUIRED_FILES = [
    'X_train.npy', 'X_test.npy', 'y_train.npy', 'y_test.npy',
    'feature_pipeline.pkl', 'feature_config.pkl',
    'train_data.parquet', 'test_data.parquet',
]

@pytest.mark.parametrize('filename', REQUIRED_FILES)
def test_artifacts_exist(filename):
    """Each required artifact from NB03 must exist in models/."""
    path = os.path.join(MODEL_DIR, filename)
    assert os.path.exists(path), f'Missing: {path}'


# ---------------------------------------------------------------------------
# Test 2: Train/test row counts are in expected range
# ---------------------------------------------------------------------------
def test_train_test_shapes():
    """Train ~3,536 rows, test ~851 rows (from the 4,387 scoped records)."""
    X_train = np.load(os.path.join(MODEL_DIR, 'X_train.npy'))
    X_test = np.load(os.path.join(MODEL_DIR, 'X_test.npy'))

    # Allow some tolerance — exact counts depend on data updates
    assert 3000 < X_train.shape[0] < 4000, f'Train rows unexpected: {X_train.shape[0]}'
    assert 500 < X_test.shape[0] < 1500, f'Test rows unexpected: {X_test.shape[0]}'


# ---------------------------------------------------------------------------
# Test 3: Train and test have the same number of features
# ---------------------------------------------------------------------------
def test_feature_count_matches():
    """X_train and X_test must have identical column counts."""
    X_train = np.load(os.path.join(MODEL_DIR, 'X_train.npy'))
    X_test = np.load(os.path.join(MODEL_DIR, 'X_test.npy'))
    assert X_train.shape[1] == X_test.shape[1], (
        f'Column mismatch: train={X_train.shape[1]}, test={X_test.shape[1]}'
    )


# ---------------------------------------------------------------------------
# Test 4: No NaN in transformed data
# ---------------------------------------------------------------------------
def test_no_nan_in_transformed_data():
    """After imputation + encoding, zero NaN values should remain."""
    X_train = np.load(os.path.join(MODEL_DIR, 'X_train.npy'))
    X_test = np.load(os.path.join(MODEL_DIR, 'X_test.npy'))
    assert not np.isnan(X_train).any(), 'NaN found in X_train'
    assert not np.isnan(X_test).any(), 'NaN found in X_test'


# ---------------------------------------------------------------------------
# Test 5: No year overlap between train and test (leakage check)
# ---------------------------------------------------------------------------
def test_train_test_no_year_overlap():
    """Train years (<=2021) and test years (>=2022) must not overlap."""
    df_train = pd.read_parquet(os.path.join(MODEL_DIR, 'train_data.parquet'))
    df_test = pd.read_parquet(os.path.join(MODEL_DIR, 'test_data.parquet'))

    train_years = set(df_train['pis_year'].dropna().astype(int))
    test_years = set(df_test['pis_year'].dropna().astype(int))
    overlap = train_years & test_years

    assert len(overlap) == 0, f'Year overlap detected (data leakage!): {overlap}'
    assert max(train_years) <= 2021, f'Train extends past 2021: {max(train_years)}'
    assert min(test_years) >= 2022, f'Test starts before 2022: {min(test_years)}'


# ---------------------------------------------------------------------------
# Test 6: Target values are all positive
# ---------------------------------------------------------------------------
def test_target_values_positive():
    """Annual federal award must be > 0 for all rows."""
    y_train = np.load(os.path.join(MODEL_DIR, 'y_train.npy'))
    y_test = np.load(os.path.join(MODEL_DIR, 'y_test.npy'))
    assert (y_train > 0).all(), f'Non-positive values in y_train'
    assert (y_test > 0).all(), f'Non-positive values in y_test'


# ---------------------------------------------------------------------------
# Test 7: Percentage features clamped to [0, 1]
# ---------------------------------------------------------------------------
PCT_FEATURES = [
    'li_units_pct', 'studio_pct', 'one_br_pct', 'two_br_pct',
    'three_plus_br_pct', 'deep_ami_pct', 'mid_ami_pct',
]

def test_percentage_features_in_range():
    """All pct-derived features should be in [0, 1] in the raw data."""
    df_train = pd.read_parquet(os.path.join(MODEL_DIR, 'train_data.parquet'))
    df_test = pd.read_parquet(os.path.join(MODEL_DIR, 'test_data.parquet'))
    df = pd.concat([df_train, df_test])

    for col in PCT_FEATURES:
        vals = df[col].dropna()
        assert vals.min() >= 0, f'{col} has values below 0: {vals.min()}'
        assert vals.max() <= 1, f'{col} has values above 1: {vals.max()}'


# ---------------------------------------------------------------------------
# Test 8: Feature config lists correct number of output names
# ---------------------------------------------------------------------------
def test_feature_names_count():
    """feature_config output names must match X_train column count."""
    X_train = np.load(os.path.join(MODEL_DIR, 'X_train.npy'))
    config = joblib.load(os.path.join(MODEL_DIR, 'feature_config.pkl'))
    assert len(config['feature_names_out']) == X_train.shape[1], (
        f'Config says {len(config["feature_names_out"])} features, '
        f'but X_train has {X_train.shape[1]} columns'
    )


# ---------------------------------------------------------------------------
# Test 9: StandardScaler approximately centers training data
# ---------------------------------------------------------------------------
def test_scaler_centered_train():
    """Numeric features in training data should have mean close to 0."""
    X_train = np.load(os.path.join(MODEL_DIR, 'X_train.npy'))
    config = joblib.load(os.path.join(MODEL_DIR, 'feature_config.pkl'))

    n_numeric = len(config['numeric_features'])
    numeric_means = X_train[:, :n_numeric].mean(axis=0)

    # After StandardScaler, means should be very close to 0
    for i, mean_val in enumerate(numeric_means):
        assert abs(mean_val) < 0.01, (
            f'Feature {config["numeric_features"][i]} has mean={mean_val:.4f} '
            f'(expected ~0 after StandardScaler)'
        )


# ---------------------------------------------------------------------------
# Test 10: No duplicate rows in train or test
# ---------------------------------------------------------------------------
def test_no_duplicate_rows():
    """Train and test DataFrames should have no duplicate app_numbers."""
    df_train = pd.read_parquet(os.path.join(MODEL_DIR, 'train_data.parquet'))
    df_test = pd.read_parquet(os.path.join(MODEL_DIR, 'test_data.parquet'))

    train_dupes = df_train['app_number'].duplicated().sum()
    test_dupes = df_test['app_number'].duplicated().sum()
    assert train_dupes == 0, f'{train_dupes} duplicate app_numbers in train'
    assert test_dupes == 0, f'{test_dupes} duplicate app_numbers in test'
