"""
Smoke tests for the Streamlit app (streamlit_app.py).

HOW THIS WORKS:
- These tests import the app's helper functions directly and verify
  they work end-to-end with real models — no browser or Streamlit
  server needed.
- They simulate what happens when a user fills in the sidebar and
  clicks through each tab.
- Run with: pytest tests/test_app.py -v

PREREQUISITE:
- All model .pkl files must exist in models/ (run NB03 + NB04 first).
- data/fred_ppi.csv must exist.

WHAT EACH TEST CHECKS (and why it matters):

test_load_models
    All 7 model artifacts must load without error. If any .pkl file
    is corrupted or missing, the app would crash on startup.

test_load_project_data
    The combined train+test data must load and have expected columns.
    The app uses this for comparables and trend analysis.

test_build_lookups
    The county/region/FMR/income lookup tables must be populated.
    The app uses these to auto-fill macro features when the user
    selects a county.

test_build_input_row
    Given user inputs, build_input_row must produce a single-row
    DataFrame with all 20 features the pipeline expects.

test_full_prediction_pipeline
    End-to-end: user inputs → build_input_row → pipeline.transform →
    quantile predictions → XGBoost prediction. If this fails, the
    Credit Estimator tab is broken.

test_knn_comparables
    The KNN model must return 10 neighbors for a new project input.
    If this fails, the Market Comparables tab is broken.

test_kmeans_archetype
    KMeans must assign the project to one of 6 clusters.

test_shap_explanation
    The SHAP helper must return feature importance pairs that the
    app uses for the waterfall chart.

test_trend_data_has_years
    The project data must span multiple years for trend charts to
    render. If all data is from one year, the trend tab would be
    empty.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

# Add project root to path so we can import the app module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from streamlit_app import (
    build_input_row,
    get_shap_explanation,
    CA_COUNTY_COORDS,
    MODEL_DIR,
    DATA_DIR,
)


# ---------------------------------------------------------------------------
# Fixtures: load once, share across tests
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def models():
    """Load all models (same as the app's load_models but without st.cache)."""
    import joblib
    m = {}
    m["xgb"] = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
    m["quantile"] = joblib.load(os.path.join(MODEL_DIR, "credit_quantile.pkl"))
    m["knn"] = joblib.load(os.path.join(MODEL_DIR, "knn_model.pkl"))
    m["kmeans"] = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))
    m["pipeline"] = joblib.load(os.path.join(MODEL_DIR, "feature_pipeline.pkl"))
    m["config"] = joblib.load(os.path.join(MODEL_DIR, "feature_config.pkl"))
    m["shap"] = joblib.load(os.path.join(MODEL_DIR, "shap_explainer.pkl"))
    return m


@pytest.fixture(scope="module")
def project_data():
    """Load combined train+test data."""
    df_train = pd.read_parquet(os.path.join(MODEL_DIR, "train_data.parquet"))
    df_test = pd.read_parquet(os.path.join(MODEL_DIR, "test_data.parquet"))
    df = pd.concat([df_train, df_test], ignore_index=True)
    df["county"] = df["county"].str.strip().str.title()
    return df


@pytest.fixture(scope="module")
def lookups(project_data):
    """Build the lookup tables the app uses."""
    df = project_data
    return {
        "county_region": df.groupby("county")["region"].agg(lambda x: x.mode()[0]).to_dict(),
        "county_fmr": df.groupby("county")["fmr_2br"].median().to_dict(),
        "county_income": df.groupby("county")["county_median_income"].median().to_dict(),
        "county_rent": df.groupby("county")["county_median_rent"].median().to_dict(),
        "counties": sorted(df["county"].dropna().unique()),
    }


@pytest.fixture(scope="module")
def ppi_df():
    """Load PPI data."""
    ppi = pd.read_csv(os.path.join(DATA_DIR, "fred_ppi.csv"))
    ppi["observation_date"] = pd.to_datetime(ppi["observation_date"])
    ppi["year"] = ppi["observation_date"].dt.year
    return ppi


@pytest.fixture(scope="module")
def sample_inputs():
    """A realistic set of user inputs (like what the sidebar would produce)."""
    return {
        "county": "Los Angeles",
        "housing_type": "Large Family",
        "construction_type": "New Construction",
        "credit_type": "9%",
        "total_units": 100.0,
        "studio_pct": 0.0,
        "one_br_pct": 0.40,
        "two_br_pct": 0.35,
        "three_plus_br_pct": 0.25,
        "deep_ami_pct": 0.10,
        "mid_ami_pct": 0.60,
        "li_units_pct": 0.95,
        "pis_year": 2026,
        "unit_mix_valid": True,
    }


# ---------------------------------------------------------------------------
# Test 1: All models load
# ---------------------------------------------------------------------------
def test_load_models(models):
    """All 7 model artifacts must load successfully."""
    assert "xgb" in models
    assert "quantile" in models
    assert "knn" in models
    assert "kmeans" in models
    assert "pipeline" in models
    assert "config" in models
    assert "shap" in models


# ---------------------------------------------------------------------------
# Test 2: Project data loads with expected columns
# ---------------------------------------------------------------------------
def test_load_project_data(project_data):
    """Combined data must have key columns for trend analysis and comparables."""
    assert len(project_data) > 4000
    required_cols = [
        "county", "pis_year", "housing_type", "credit_type",
        "annual_federal_award", "credit_per_unit", "total_units",
    ]
    for col in required_cols:
        assert col in project_data.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# Test 3: Lookups are populated
# ---------------------------------------------------------------------------
def test_build_lookups(lookups):
    """Lookup tables must have entries for Los Angeles at minimum."""
    assert "Los Angeles" in lookups["county_region"]
    assert "Los Angeles" in lookups["county_fmr"]
    assert "Los Angeles" in lookups["county_income"]
    assert len(lookups["counties"]) > 50


# ---------------------------------------------------------------------------
# Test 4: build_input_row creates a valid DataFrame
# ---------------------------------------------------------------------------
def test_build_input_row(sample_inputs, lookups, ppi_df):
    """build_input_row must produce a single-row DataFrame with all 20 features."""
    row = build_input_row(sample_inputs, lookups, ppi_df)
    assert isinstance(row, pd.DataFrame)
    assert row.shape[0] == 1

    expected_cols = [
        "total_units", "li_units_pct", "studio_pct", "one_br_pct",
        "two_br_pct", "three_plus_br_pct", "deep_ami_pct", "mid_ami_pct",
        "pis_year", "ppi_at_allocation", "ppi_yoy_change", "ppi_2yr_trend",
        "fmr_2br", "county_median_income", "county_median_rent",
        "county", "region", "credit_type", "construction_type", "housing_type",
    ]
    for col in expected_cols:
        assert col in row.columns, f"Missing column in input row: {col}"


# ---------------------------------------------------------------------------
# Test 5: Full prediction pipeline (end-to-end)
# ---------------------------------------------------------------------------
def test_full_prediction_pipeline(models, sample_inputs, lookups, ppi_df):
    """User inputs → transform → quantile + XGBoost predictions must produce valid numbers."""
    row = build_input_row(sample_inputs, lookups, ppi_df)
    X = models["pipeline"].transform(row)

    # Quantile predictions
    q = models["quantile"]
    p10 = np.expm1(q["p10"].predict(X)[0])
    p50 = np.expm1(q["p50"].predict(X)[0])
    p90 = np.expm1(q["p90"].predict(X)[0])

    # XGBoost
    xgb_pred = np.expm1(models["xgb"].predict(X)[0])

    # All should be positive and in a reasonable range
    for label, val in [("p10", p10), ("p50", p50), ("p90", p90), ("xgb", xgb_pred)]:
        assert val > 10_000, f"{label} prediction ${val:,.0f} is unreasonably low"
        assert val < 50_000_000, f"{label} prediction ${val:,.0f} is unreasonably high"
        assert not np.isnan(val), f"{label} prediction is NaN"


# ---------------------------------------------------------------------------
# Test 6: KNN returns 10 neighbors
# ---------------------------------------------------------------------------
def test_knn_comparables(models, sample_inputs, lookups, ppi_df):
    """KNN must return exactly 10 neighbors for a new project."""
    row = build_input_row(sample_inputs, lookups, ppi_df)
    X = models["pipeline"].transform(row)

    distances, indices = models["knn"].kneighbors(X)
    assert indices.shape == (1, 10), f"Expected (1, 10), got {indices.shape}"
    assert (distances >= 0).all(), "Negative distances found"


# ---------------------------------------------------------------------------
# Test 7: KMeans assigns a valid cluster
# ---------------------------------------------------------------------------
def test_kmeans_archetype(models, sample_inputs, lookups, ppi_df):
    """KMeans must assign the project to a cluster between 0 and 5."""
    row = build_input_row(sample_inputs, lookups, ppi_df)
    X = models["pipeline"].transform(row)

    cluster_id = models["kmeans"].predict(X)[0]
    assert 0 <= cluster_id < 6, f"Cluster {cluster_id} out of range [0, 5]"


# ---------------------------------------------------------------------------
# Test 8: SHAP explanation returns feature-value pairs
# ---------------------------------------------------------------------------
def test_shap_explanation(models, sample_inputs, lookups, ppi_df):
    """SHAP helper must return a list of (feature_name, shap_value) tuples."""
    row = build_input_row(sample_inputs, lookups, ppi_df)
    X = models["pipeline"].transform(row)

    feature_names = models["config"]["feature_names_out"]
    shap_top = get_shap_explanation(models, X, feature_names, top_n=5)

    assert len(shap_top) == 5, f"Expected 5 SHAP features, got {len(shap_top)}"
    for name, val in shap_top:
        assert isinstance(name, str)
        assert isinstance(val, (float, np.floating))
        assert not np.isnan(val), f"NaN SHAP value for {name}"


# ---------------------------------------------------------------------------
# Test 9: Trend data spans multiple years
# ---------------------------------------------------------------------------
def test_trend_data_has_years(project_data):
    """Project data must span at least 15 years for meaningful trend charts."""
    years = project_data["pis_year"].dropna().unique()
    assert len(years) >= 15, f"Only {len(years)} unique years — trends need more"
    assert years.min() <= 2005, f"Data starts at {years.min()} — expected ≤ 2005"
    assert years.max() >= 2022, f"Data ends at {years.max()} — expected ≥ 2022"


# ---------------------------------------------------------------------------
# Test 10: County coordinates cover all counties in data
# ---------------------------------------------------------------------------
def test_county_coords_coverage(project_data):
    """CA_COUNTY_COORDS should cover the vast majority of counties in the data."""
    data_counties = set(project_data["county"].dropna().unique())
    coord_counties = set(CA_COUNTY_COORDS.keys())
    missing = data_counties - coord_counties
    coverage = len(data_counties & coord_counties) / len(data_counties)
    assert coverage > 0.90, (
        f"Only {coverage:.0%} county coordinate coverage. Missing: {missing}"
    )
