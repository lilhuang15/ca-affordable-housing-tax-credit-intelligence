"""
CA Affordable Housing Tax Credit Intelligence Platform
======================================================
Streamlit app — Phase 1 (4 tabs).

Tab 1: Credit Estimator      — XGBoost point estimate + split-conformal interval + SHAP
Tab 2: Market Comparables    — KNN top-10 + Project Group lookup + map
Tab 3: Credit Trend Analysis — statewide/county/housing-type trends
Tab 4: Method Insight        — time-split, distribution shift, fit, SHAP, conformal coverage

Run:  streamlit run streamlit_app.py
"""

import os
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from theme import (
    inject_theme,
    render_masthead,
    CHART_COLORS,
    HOUSING_TYPE_COLORS,
    PALETTE,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CA LIHTC Credit Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_theme()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ---------------------------------------------------------------------------
# Load models and data (cached so they load only once)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    """Load all .pkl model files into memory once."""
    models = {}
    models["xgb"] = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
    models["conformal"] = joblib.load(os.path.join(MODEL_DIR, "conformal_calibration.pkl"))
    models["knn"] = joblib.load(os.path.join(MODEL_DIR, "knn_model.pkl"))
    # Deterministic archetype lookup (replaced KMeans in NB04 §16). Each row is one
    # (credit_type × housing_type × region) cell with aggregate stats.
    models["archetype_profile"] = pd.read_parquet(
        os.path.join(MODEL_DIR, "archetype_profile.parquet")
    )
    models["pipeline"] = joblib.load(os.path.join(MODEL_DIR, "feature_pipeline.pkl"))
    models["config"] = joblib.load(os.path.join(MODEL_DIR, "feature_config.pkl"))
    models["shap"] = joblib.load(os.path.join(MODEL_DIR, "shap_explainer.pkl"))
    return models


@st.cache_data
def load_model_comparison():
    """Load test-set metrics table written by NB04."""
    return pd.read_csv(os.path.join(MODEL_DIR, "model_comparison.csv"))


@st.cache_data
def load_project_splits():
    """Load the scoped train and test sets separately.

    Tab 4 (Method Insight) uses the splits directly; Tab 2/3 use the concatenated
    view via load_project_data, which is a thin wrapper around this function.
    """
    df_train = pd.read_parquet(os.path.join(MODEL_DIR, "train_data.parquet"))
    df_test = pd.read_parquet(os.path.join(MODEL_DIR, "test_data.parquet"))
    df_train["county"] = df_train["county"].str.strip().str.title()
    df_test["county"] = df_test["county"].str.strip().str.title()
    return df_train, df_test


@st.cache_data
def load_project_data():
    """Load the scoped project data (train + test) for comparables and trends."""
    df_train, df_test = load_project_splits()
    return pd.concat([df_train, df_test], ignore_index=True)


@st.cache_data
def load_ppi():
    """Load FRED PPI data for macro context."""
    ppi = pd.read_csv(os.path.join(DATA_DIR, "fred_ppi.csv"))
    ppi["observation_date"] = pd.to_datetime(ppi["observation_date"])
    ppi["year"] = ppi["observation_date"].dt.year
    return ppi


@st.cache_data
def build_lookups(_df):
    """Build helper lookup tables from project data."""
    # County -> Region mapping (most common region per county)
    county_region = _df.groupby("county")["region"].agg(
        lambda x: x.mode()[0]
    ).to_dict()

    # County -> latest FMR (median across years)
    county_fmr = _df.groupby("county")["fmr_2br"].median().to_dict()

    # County -> latest Census stats (median)
    county_income = _df.groupby("county")["county_median_income"].median().to_dict()
    county_rent = _df.groupby("county")["county_median_rent"].median().to_dict()

    # Sorted county list
    counties = sorted(_df["county"].dropna().unique())

    # Statewide medians — used only as fallback when an unknown county slips through
    default_region = _df["region"].mode()[0]
    default_fmr = float(_df["fmr_2br"].median())
    default_income = float(_df["county_median_income"].median())
    default_rent = float(_df["county_median_rent"].median())

    return {
        "county_region": county_region,
        "county_fmr": county_fmr,
        "county_income": county_income,
        "county_rent": county_rent,
        "counties": counties,
        "default_region": default_region,
        "default_fmr": default_fmr,
        "default_income": default_income,
        "default_rent": default_rent,
    }


# County centroid lat/lon for map (CA counties)
CA_COUNTY_COORDS = {
    "Alameda": (37.65, -121.89), "Alpine": (38.60, -119.82),
    "Amador": (38.44, -120.65), "Butte": (39.67, -121.60),
    "Calaveras": (38.20, -120.56), "Colusa": (39.18, -122.24),
    "Contra Costa": (37.92, -121.95), "Del Norte": (41.74, -123.90),
    "El Dorado": (38.78, -120.52), "Fresno": (36.77, -119.65),
    "Glenn": (39.60, -122.39), "Humboldt": (40.71, -123.90),
    "Imperial": (32.84, -115.57), "Inyo": (36.51, -117.41),
    "Kern": (35.34, -118.73), "Kings": (36.07, -119.81),
    "Lake": (39.10, -122.75), "Lassen": (40.67, -120.59),
    "Los Angeles": (34.05, -118.25), "Madera": (37.22, -119.76),
    "Marin": (38.05, -122.75), "Mariposa": (37.58, -119.97),
    "Mendocino": (39.44, -123.39), "Merced": (37.19, -120.72),
    "Modoc": (41.59, -120.73), "Mono": (37.94, -118.89),
    "Monterey": (36.24, -121.31), "Napa": (38.50, -122.27),
    "Nevada": (39.30, -120.77), "Orange": (33.72, -117.78),
    "Placer": (39.06, -120.72), "Plumas": (40.01, -120.84),
    "Riverside": (33.74, -115.99), "Sacramento": (38.45, -121.35),
    "San Benito": (36.61, -121.08), "San Bernardino": (34.84, -116.18),
    "San Diego": (32.83, -116.77), "San Francisco": (37.78, -122.42),
    "San Joaquin": (37.93, -121.27), "San Luis Obispo": (35.34, -120.44),
    "San Mateo": (37.43, -122.36), "Santa Barbara": (34.54, -119.96),
    "Santa Clara": (37.23, -121.70), "Santa Cruz": (37.06, -122.01),
    "Shasta": (40.76, -122.04), "Sierra": (39.58, -120.52),
    "Siskiyou": (41.59, -122.54), "Solano": (38.27, -121.94),
    "Sonoma": (38.53, -122.93), "Stanislaus": (37.56, -120.99),
    "Sutter": (39.03, -121.69), "Tehama": (40.13, -122.24),
    "Tulare": (36.23, -118.78), "Tuolumne": (38.03, -119.96),
    "Ventura": (34.36, -119.13), "Yolo": (38.69, -121.90),
    "Yuba": (39.27, -121.35),
}


# ---------------------------------------------------------------------------
# Helper: build input DataFrame from user selections
# ---------------------------------------------------------------------------
def build_input_row(inputs, lookups, ppi_df):
    """Create a single-row DataFrame matching the feature pipeline's expected columns."""
    county = inputs["county"]
    region = lookups["county_region"].get(county, lookups["default_region"])
    fmr = lookups["county_fmr"].get(county, lookups["default_fmr"])
    income = lookups["county_income"].get(county, lookups["default_income"])
    rent = lookups["county_rent"].get(county, lookups["default_rent"])

    # Get PPI for the selected year (or latest available)
    ppi_year = ppi_df[ppi_df["year"] <= inputs["pis_year"]]
    if len(ppi_year) > 0:
        ppi_annual = ppi_year.groupby("year")["WPUSI012011"].mean()
        yr = inputs["pis_year"]
        ppi_val = ppi_annual.get(yr, ppi_annual.iloc[-1])
        if yr - 1 in ppi_annual.index:
            ppi_yoy = (ppi_val - ppi_annual[yr - 1]) / ppi_annual[yr - 1] * 100
        else:
            ppi_yoy = 3.0
        # ppi_2yr_trend is a BINARY direction indicator in NB02 (data/02_enrich_data.ipynb:882):
        #     (ppi_at_allocation > ppi_at_allocation.shift(2)).astype(float)
        # An earlier version of this app computed it as a 2-year percentage change (~6.5
        # for 2026), which the trained pipeline's StandardScaler then mapped to z ≈ +18,
        # dominating cosine distance and crushing KNN similarity to ~32%. The fix matches
        # the training feature exactly: 1.0 if PPI is rising vs 2 years ago, else 0.0.
        if yr - 2 in ppi_annual.index:
            ppi_2yr = float(ppi_val > ppi_annual[yr - 2])
        else:
            ppi_2yr = 1.0
    else:
        ppi_val, ppi_yoy, ppi_2yr = 350.0, 3.0, 1.0

    row = pd.DataFrame([{
        "total_units": inputs["total_units"],
        "li_units_pct": inputs["li_units_pct"],
        "studio_pct": inputs["studio_pct"],
        "one_br_pct": inputs["one_br_pct"],
        "two_br_pct": inputs["two_br_pct"],
        "three_plus_br_pct": inputs["three_plus_br_pct"],
        "deep_ami_pct": inputs["deep_ami_pct"],
        "low_ami_pct": inputs["low_ami_pct"],
        "mid_ami_pct": inputs["mid_ami_pct"],
        "pis_year": inputs["pis_year"],
        "ppi_at_allocation": ppi_val,
        "ppi_yoy_change": ppi_yoy,
        "ppi_2yr_trend": ppi_2yr,
        "fmr_2br": fmr,
        "county_median_income": income,
        "county_median_rent": rent,
        "county": county,
        "region": region,
        "credit_type": inputs["credit_type"],
        "construction_type": inputs["construction_type"],
        "housing_type": inputs["housing_type"],
    }])
    return row


# ---------------------------------------------------------------------------
# Display-name mapping for SHAP and other feature-name UIs
# ---------------------------------------------------------------------------
# Sklearn's ColumnTransformer emits feature names with `num__/te__/ohe__`
# prefixes and snake_case fields (e.g. `num__fmr_2br`, `ohe__housing_type_Large
# Family`). Showing these raw confuses non-ML users. `pretty_feature_name`
# rewrites them into plain English for the SHAP bar charts (Tab 1 + Tab 4).
_NUM_LABELS = {
    "total_units": "Total Units",
    "li_units_pct": "Low-Income Unit %",
    "studio_pct": "Studio/SRO Unit %",
    "one_br_pct": "1-BR Unit %",
    "two_br_pct": "2-BR Unit %",
    "three_plus_br_pct": "3-BR+ Unit %",
    "deep_ami_pct": "Deep AMI ≤30% Units",
    "low_ami_pct": "Low AMI 30–50% Units",
    "mid_ami_pct": "Mid AMI 50–60% Units",
    "pis_year": "Placed-in-Service Year",
    "ppi_at_allocation": "PPI (Construction Cost)",
    "ppi_yoy_change": "PPI YoY Change",
    "ppi_2yr_trend": "PPI 2-Year Trend",
    "fmr_2br": "Fair Market Rent (2-BR)",
    "county_median_income": "County Median Income",
    "county_median_rent": "County Median Rent",
}
_TE_LABELS = {
    "county": "County",
}
# Match longest keys first so e.g. `housing_type_*` isn't split as `housing_*`.
_OHE_COL_LABELS = {
    "credit_type": "Credit Type",
    "construction_type": "Construction",
    "housing_type": "Housing Type",
    "region": "Region",
}


def pretty_feature_name(raw: str) -> str:
    """Rewrite a ColumnTransformer feature name into a user-readable label."""
    if raw.startswith("num__"):
        key = raw[len("num__"):]
        return _NUM_LABELS.get(key, key.replace("_", " ").title())
    if raw.startswith("te__"):
        key = raw[len("te__"):]
        return _TE_LABELS.get(key, key.replace("_", " ").title())
    if raw.startswith("ohe__"):
        key = raw[len("ohe__"):]
        for col in sorted(_OHE_COL_LABELS, key=len, reverse=True):
            prefix = col + "_"
            if key.startswith(prefix):
                return f"{_OHE_COL_LABELS[col]}: {key[len(prefix):]}"
        return key
    return raw


# ---------------------------------------------------------------------------
# Helper: SHAP waterfall (top features)
# ---------------------------------------------------------------------------
def get_shap_explanation(models, X_transformed, feature_names, top_n=8):
    """Return SHAP values for a single observation."""
    shap_vals = models["shap"].shap_values(X_transformed)
    sv = shap_vals[0]  # single row

    pairs = list(zip(feature_names, sv))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return [(pretty_feature_name(name), val) for name, val in pairs[:top_n]]


# ---------------------------------------------------------------------------
# Method Insight: cached compute helpers
# ---------------------------------------------------------------------------
# These compute test-set diagnostics once per session and cache the result so
# Tab 4 paints quickly. The leading underscore on `_models` / `_df_test` opts
# the argument out of @st.cache_data hashing — the models dict and DataFrame
# don't have stable hashes, but they're identity-stable for the session.
@st.cache_data
def compute_test_predictions(_models, _df_test):
    """Run the deployed XGBoost on the held-out test set and apply the
    split-conformal multipliers. Returns a single tidy frame used by both
    the Actual-vs-Predicted chart and the Conformal Coverage chart."""
    pipeline = _models["pipeline"]
    xgb = _models["xgb"]
    conf = _models["conformal"]

    X = pipeline.transform(_df_test)
    pred_log = xgb.predict(X)
    pred = np.expm1(pred_log)
    actual = _df_test["annual_federal_award"].to_numpy()

    lower = pred * conf["multiplier_low"]
    upper = pred * conf["multiplier_high"]
    in_interval = (actual >= lower) & (actual <= upper)

    return pd.DataFrame({
        "pis_year": _df_test["pis_year"].to_numpy(),
        "actual": actual,
        "predicted": pred,
        "log_residual": np.log1p(actual) - pred_log,
        "lower": lower,
        "upper": upper,
        "in_interval": in_interval,
    })


@st.cache_data
def compute_global_shap(_models, _df_test, n_sample=200, seed=42):
    """Mean |SHAP| across a 200-row sample of the test set — the static
    counterpart to the per-prediction SHAP shown in Tab 1. Sampling keeps
    first-paint under ~3 s; the cached result is reused on every Tab 4 visit."""
    pipeline = _models["pipeline"]
    explainer = _models["shap"]
    feature_names = _models["config"]["feature_names_out"]

    sample = _df_test.sample(n=min(n_sample, len(_df_test)), random_state=seed)
    X = pipeline.transform(sample)
    shap_vals = explainer.shap_values(X)
    mean_abs = np.abs(shap_vals).mean(axis=0)

    df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
        "display_name": [pretty_feature_name(f) for f in feature_names],
    })
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sidebar: shared project inputs
# ---------------------------------------------------------------------------
def render_sidebar(lookups):
    """Render sidebar inputs and return dict of user selections."""
    st.sidebar.header("Project Inputs")

    county = st.sidebar.selectbox(
        "County",
        lookups["counties"],
        index=lookups["counties"].index("San Francisco")
        if "San Francisco" in lookups["counties"] else 0,
    )

    housing_type = st.sidebar.selectbox(
        "Housing Type",
        ["Large Family", "Senior", "Special Needs", "SRO", "Non-Targeted", "At-Risk"],
    )

    construction_type = st.sidebar.radio(
        "Construction Type",
        ["New Construction", "Rehabilitation"],
    )

    credit_type = st.sidebar.radio(
        "Credit Type",
        ["9%", "4%"],
    )

    total_units = st.sidebar.number_input(
        "Total Units", min_value=10, max_value=500, value=100, step=10,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Unit Mix (%)")
    st.sidebar.caption("Must sum to 100%")
    studio_pct = st.sidebar.slider("SRO/Studio %", 0, 100, 0, 5)
    one_br_pct = st.sidebar.slider("1-BR %", 0, 100, 40, 5)
    two_br_pct = st.sidebar.slider("2-BR %", 0, 100, 35, 5)
    three_br_pct = st.sidebar.slider("3-BR+ %", 0, 100, 25, 5)
    unit_mix_total = studio_pct + one_br_pct + two_br_pct + three_br_pct

    if unit_mix_total != 100:
        st.sidebar.warning(f"Unit mix sums to {unit_mix_total}% — must be 100%")

    st.sidebar.markdown("---")
    st.sidebar.subheader("AMI Targeting (%)")
    st.sidebar.caption("Deep = ≤30% AMI, Low = 30-50% AMI, Mid = 50-60% AMI. Must sum to ≤100%")
    deep_ami = st.sidebar.slider("Deep (≤30% AMI) %", 0, 100, 10, 5)
    low_ami = st.sidebar.slider("Low (30-50% AMI) %", 0, 100, 30, 5)
    mid_ami = st.sidebar.slider("Mid (50-60% AMI) %", 0, 100, 60, 5)

    st.sidebar.markdown("---")
    pis_year = st.sidebar.number_input(
        "Expected PIS Year", min_value=2024, max_value=2030, value=2026,
    )

    li_units_pct = st.sidebar.slider(
        "Low-Income Units %", 20, 100, 95, 5,
        help=(
            "Percentage of total units restricted as low-income. "
            "Federal LIHTC minimum is 20% (20-50 set-aside test); most CA "
            "projects are 95–100%, but inclusionary deals in market-rate "
            "buildings (e.g. SF luxury) can be 20–35%."
        ),
    ) / 100.0

    return {
        "county": county,
        "housing_type": housing_type,
        "construction_type": construction_type,
        "credit_type": credit_type,
        "total_units": float(total_units),
        "studio_pct": studio_pct / 100.0,
        "one_br_pct": one_br_pct / 100.0,
        "two_br_pct": two_br_pct / 100.0,
        "three_plus_br_pct": three_br_pct / 100.0,
        "deep_ami_pct": deep_ami / 100.0,
        "low_ami_pct": low_ami / 100.0,
        "mid_ami_pct": mid_ami / 100.0,
        "li_units_pct": li_units_pct,
        "pis_year": pis_year,
        "unit_mix_valid": unit_mix_total == 100,
    }


# ---------------------------------------------------------------------------
# Tab 1: Credit Estimator
# ---------------------------------------------------------------------------
def render_credit_estimator(models, lookups, ppi_df, inputs):
    st.header("Credit Estimator")
    st.caption(
        "Predict the annual federal tax credit allocation for a proposed CA LIHTC project."
    )

    if not inputs["unit_mix_valid"]:
        st.error("Please adjust the unit mix in the sidebar so it sums to 100%.")
        return

    # Build input and transform
    row = build_input_row(inputs, lookups, ppi_df)
    X = models["pipeline"].transform(row)

    # --- XGBoost point estimate + split-conformal interval (multiplicative) ---
    # Calibrated on 2020-2021 holdout residuals in log-space (Section 8b of NB04),
    # applied as a constant multiplicative band on the dollar prediction:
    #     [pred × multiplier_low,  pred × multiplier_high]
    # Multipliers are model-level constants; dollar bounds adapt to project size.
    xgb_pred = float(np.expm1(models["xgb"].predict(X)[0]))
    mult_low = models["conformal"]["multiplier_low"]
    mult_high = models["conformal"]["multiplier_high"]
    pct_low = models["conformal"]["pct_low"]     # e.g. -50.0
    pct_high = models["conformal"]["pct_high"]   # e.g. +100.0
    coverage_target = int(round((1 - models["conformal"]["alpha"]) * 100))
    p10 = xgb_pred * mult_low
    p50 = xgb_pred
    p90 = xgb_pred * mult_high

    # --- Key metrics: dollar amount + percentage shift in the title ---
    # Percentage shifts (e.g. -42% / +73%) are constant per model — the dollar
    # width adapts to project size but the percentage is the same for every
    # query. Showing it in the metric title avoids a duplicate `delta` field.
    lower_label = f"Lower ({pct_low:+.0f}%)"
    upper_label = f"Upper ({pct_high:+.0f}%)"
    col1, col2, col3 = st.columns(3)
    col1.metric(lower_label, f"${p10:,.0f}")
    col2.metric("Point estimate", f"${p50:,.0f}")
    col3.metric(upper_label, f"${p90:,.0f}")
    st.caption(
        f"There's a {coverage_target}% chance the actual award falls in this range. "
        f"The lower and upper bounds are a constant {pct_low:+.0f}% / {pct_high:+.0f}% shift "
        f"from the point estimate — same percentage for every project, but the dollar width grows "
        f"with project size. On the 2022–2025 test set this range covered the actual award "
        f"{models['conformal']['coverage_test']:.1f}% of the time."
    )

    # --- Prediction range: dot + error bar (one estimate with its interval) ---
    # The three numbers are ONE estimate and its uncertainty range, not three
    # quantities to compare — so they're drawn as a dot with a whisker, dollar
    # amounts on the y-axis, instead of three side-by-side bars.
    st.subheader("Annual Federal Credit — Prediction Range")
    fig_range = go.Figure()
    fig_range.add_trace(go.Scatter(
        x=["Your project"], y=[p50],
        mode="markers",
        marker=dict(size=18, color=CHART_COLORS["upper"]),
        error_y=dict(
            type="data", symmetric=False,
            array=[p90 - p50], arrayminus=[p50 - p10],
            color=CHART_COLORS["point"], thickness=4, width=22,
        ),
        customdata=[[p10, p90]],
        hovertemplate=(
            "P95: $%{customdata[1]:,.0f}<br>"
            "Point: $%{y:,.0f}<br>"
            "P5: $%{customdata[0]:,.0f}<extra></extra>"
        ),
    ))
    for yv, txt, bold in (
        (p10, f"P5&nbsp;&nbsp;${p10:,.0f}", False),
        (p50, f"Point&nbsp;&nbsp;${p50:,.0f}", True),
        (p90, f"P95&nbsp;&nbsp;${p90:,.0f}", False),
    ):
        fig_range.add_annotation(
            x="Your project", y=yv, xshift=110, xanchor="left", showarrow=False,
            text=f"<b>{txt}</b>" if bold else txt,
            font=dict(size=13),
        )
    fig_range.update_layout(
        yaxis_title="Annual Federal Credit ($)",
        yaxis_tickformat="$,.0f",
        yaxis_range=[0, p90 * 1.15],
        xaxis_range=[-0.9, 1.7],
        height=420,
        showlegend=False,
    )
    st.plotly_chart(fig_range, width="stretch")

    # --- 10-year credit stream & equity ---
    st.subheader("10-Year Credit Stream & Equity Estimate")
    credit_price = st.slider(
        "Credit Price ($/credit)", 0.70, 1.00, 0.92, 0.01,
        help="Investor price per dollar of tax credit. Typical range: $0.85–$0.95.",
    )

    stream_col1, stream_col2, stream_col3 = st.columns(3)
    for col, label, annual in [
        (stream_col1, f"Lower ({pct_low:+.0f}%)", p10),
        (stream_col2, "Point estimate", p50),
        (stream_col3, f"Upper ({pct_high:+.0f}%)", p90),
    ]:
        ten_yr = annual * 10
        annual_equity = annual * credit_price
        equity = ten_yr * credit_price
        # Escape every "$" as "\$" so streamlit doesn't enter LaTeX math mode
        # (two unescaped $ on a line trigger `$...$` math, which makes the text
        # in between render in a math font and breaks `**bold**` markers around
        # adjacent dollar amounts). Drop bold markers to keep the bullets plain.
        col.markdown(f"**{label}**")
        col.markdown(f"- Annual credit: \\${annual:,.0f}")
        col.markdown(f"- Annual equity @ \\${credit_price:.2f}: \\${annual_equity:,.0f}")
        col.markdown(f"- 10-year stream: \\${ten_yr:,.0f}")
        col.markdown(f"- 10-year equity @ \\${credit_price:.2f}: \\${equity:,.0f}")

    # --- Implied Eligible Basis ---
    st.subheader("Implied Eligible Basis (Cost Proxy)")
    st.caption(
        "**Eligible Basis = Annual Credit ÷ (Credit Rate × Low-Income Unit Fraction).** "
        "This is the *whole building's* depreciable construction cost — for "
        "mixed-income deals it includes the market-rate portion, so it will "
        "look larger than the Low-Income-only build cost. The credit itself is computed "
        "on Qualified Basis = Eligible Basis × Low-Income Unit Fraction. "
        "Benchmarking proxy only, not a direct development cost figure."
    )
    credit_rate = 0.09 if inputs["credit_type"] == "9%" else 0.04
    basis_p50 = p50 / (credit_rate * inputs["li_units_pct"])
    basis_per_unit = basis_p50 / inputs["total_units"]

    basis_col1, basis_col2, basis_col3 = st.columns(3)
    basis_col1.metric("Implied Eligible Basis (point)", f"${basis_p50:,.0f}")
    basis_col2.metric("Implied Basis per Unit", f"${basis_per_unit:,.0f}")
    basis_col3.metric("Credit Rate Used", f"{credit_rate:.0%}")

    # --- SHAP explanation ---
    st.subheader("What's Driving This Prediction?")
    st.caption("SHAP values show which features pushed the credit estimate up or down.")

    feature_names = models["config"]["feature_names_out"]
    shap_top = get_shap_explanation(models, X, feature_names)

    names = [n for n, _ in shap_top]
    values = [v for _, v in shap_top]
    colors = [CHART_COLORS["positive"] if v > 0 else CHART_COLORS["negative"] for v in values]

    fig_shap = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
    ))
    fig_shap.update_layout(
        xaxis_title="SHAP Value (impact on log-credit prediction)",
        height=max(300, len(names) * 40),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=10),
    )
    st.plotly_chart(fig_shap, width="stretch")

    # --- Plain-English summary ---
    top_pos = [(n, v) for n, v in shap_top if v > 0]
    top_neg = [(n, v) for n, v in shap_top if v < 0]
    summary_parts = []
    if top_pos:
        drivers = ", ".join(n for n, _ in top_pos[:3])
        summary_parts.append(f"**{drivers}** pushed the credit estimate higher")
    if top_neg:
        drags = ", ".join(n for n, _ in top_neg[:3])
        summary_parts.append(f"**{drags}** pulled it lower")
    if summary_parts:
        st.info(" — while ".join(summary_parts) + ".")

    # --- CSV download ---
    st.subheader("Export Results")
    export_data = pd.DataFrame([{
        "County": inputs["county"],
        "Housing Type": inputs["housing_type"],
        "Construction Type": inputs["construction_type"],
        "Credit Type": inputs["credit_type"],
        "Total Units": int(inputs["total_units"]),
        "PIS Year": inputs["pis_year"],
        "Point Estimate": round(p50),
        f"Lower ({pct_low:+.0f}%)": round(p10),
        f"Upper ({pct_high:+.0f}%)": round(p90),
        "Coverage Target": f"{coverage_target}%",
        "10yr Point Stream": round(p50 * 10),
        "Implied Eligible Basis (Point)": round(basis_p50),
        "Implied Basis per Unit": round(basis_per_unit),
    }])
    st.download_button(
        "Download Prediction (CSV)",
        export_data.to_csv(index=False),
        "credit_prediction.csv",
        "text/csv",
    )


# ---------------------------------------------------------------------------
# Tab 2: Market Comparables
# ---------------------------------------------------------------------------
def render_comparables(models, df, lookups, ppi_df, inputs):
    st.header("Market Comparables")
    st.caption(
        "The 10 most similar historical CA LIHTC projects based on project characteristics."
    )

    if not inputs["unit_mix_valid"]:
        st.error("Please adjust the unit mix in the sidebar so it sums to 100%.")
        return

    # Build input and transform
    row = build_input_row(inputs, lookups, ppi_df)
    X = models["pipeline"].transform(row)

    # --- KNN: find 10 nearest neighbors ---
    # KNN was trained on the full scoped dataset (train + test, 4,387 rows)
    distances, indices = models["knn"].kneighbors(X)
    neighbor_indices = indices[0]

    # Map back to full scoped data (must match what KNN was fitted on)
    df_train = pd.read_parquet(os.path.join(MODEL_DIR, "df_all_scoped.parquet"))
    df_train["county"] = df_train["county"].str.strip().str.title()
    comps = df_train.iloc[neighbor_indices].copy()
    comps["distance"] = distances[0]
    comps["similarity"] = 1 - comps["distance"]  # cosine: 1 = identical

    # Display table
    st.subheader("Top 10 Similar Projects")
    display_cols = [
        "project_name", "county", "pis_year", "housing_type",
        "construction_type", "credit_type", "total_units",
        "annual_federal_award", "credit_per_unit",
        "developer", "similarity",
    ]
    display_df = comps[display_cols].copy()
    # Basis/Unit dropped: it's a formula-derived echo of the award (basis =
    # award / (rate × LI)), and removing it + shorter headers lets every
    # column fit without horizontal scrolling.
    display_df.columns = [
        "Project", "County", "Year", "Type",
        "Constr.", "Credit", "Units",
        "Annual Credit", "$/Unit",
        "Developer", "Similarity",
    ]
    display_df["Annual Credit"] = display_df["Annual Credit"].apply(
        lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
    )
    display_df["$/Unit"] = display_df["$/Unit"].apply(
        lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
    )
    display_df["Similarity"] = display_df["Similarity"].apply(lambda x: f"{x:.2%}")

    st.dataframe(display_df, width="stretch", hide_index=True)

    # --- Project Group (deterministic GroupBy on credit × housing × region) ---
    # Replaced KMeans in NB04 §16. Looks up the user's exact (credit, housing, region)
    # cell, with two graceful fallbacks for sparse cells (n < 10) so the UI never
    # shows a single-project "group" with meaningless statistics.
    st.subheader("Project Group")
    st.caption(
        "Historical context from CA LIHTC projects placed in service 2015–2025. "
        "Earlier projects are excluded because "
        "their award levels and project sizes don't reflect post-2015 cost realities; "
        "for a 2026 query, a $300K award from a 2002 project is not a useful reference."
    )
    region = lookups["county_region"].get(inputs["county"], lookups["default_region"])
    profile = models["archetype_profile"]

    group_match = profile[
        (profile["credit_type"] == inputs["credit_type"]) &
        (profile["housing_type"] == inputs["housing_type"]) &
        (profile["region"] == region)
    ]
    fallback_label = None
    if len(group_match) == 0 or group_match.iloc[0]["count"] < 10:
        # Drop region — combine across CA for this credit+housing combo.
        group_match = profile[
            (profile["credit_type"] == inputs["credit_type"]) &
            (profile["housing_type"] == inputs["housing_type"])
        ]
        if len(group_match) > 0:
            fallback_label = "Statewide (region-agnostic)"
        if group_match["count"].sum() < 10:
            # Last fallback: housing_type only.
            group_match = profile[profile["housing_type"] == inputs["housing_type"]]
            fallback_label = f"All {inputs['housing_type']} projects (any credit, any region)"

    if len(group_match) > 0:
        # GroupBy may return multiple rows when we widen the query — re-aggregate.
        n = int(group_match["count"].sum())
        med_award = float((group_match["median_award"] * group_match["count"]).sum() / n)
        p10 = float((group_match["p10_award"] * group_match["count"]).sum() / n)
        p90 = float((group_match["p90_award"] * group_match["count"]).sum() / n)
        med_units = float((group_match["median_units"] * group_match["count"]).sum() / n)

        group_label = (
            f"{inputs['credit_type']} {inputs['housing_type']} in {region}"
            if fallback_label is None
            else f"{inputs['credit_type']} {inputs['housing_type']} ({fallback_label})"
        )
        # Escape "$" as "\$" so streamlit's markdown engine doesn't treat them as
        # LaTeX math delimiters. Bullets split lower/upper end into their own lines
        # to mirror the user-friendly column names used in the expander table below
        # ("Lower end award (p10)" / "Upper end award (p90)") so the inline summary
        # and the full table use the same vocabulary.
        st.info(
            f"**{group_label}** — {n:,} historical projects (2015–2025)\n\n"
            f"- Typical size: ~{med_units:.0f} units\n"
            f"- Median annual credit: \\${med_award:,.0f}\n"
            f"- Lower end award (p10): \\${p10:,.0f}\n"
            f"- Upper end award (p90): \\${p90:,.0f}"
        )
    else:
        st.warning(
            "No historical projects (2015–2025) match this group closely. "
            "Treat the point estimate above with extra caution."
        )

    # --- Show all CA LIHTC project groups for context (collapsed by default) ---
    with st.expander(f"See all {len(profile)} CA LIHTC project groups for context"):
        st.caption(
            "Each row is one (credit × housing × region) cell with its aggregate stats "
            "from 2015–2025. The lower-end / upper-end columns are the 10th and 90th "
            "percentiles of award in each cell — 80% of historical projects in that "
            "cell sit between them. Your project group is highlighted."
        )
        # Highlight the user's project-group row(s) — must be computed against
        # the original column names BEFORE the rename below.
        user_mask = (
            (profile["credit_type"] == inputs["credit_type"]) &
            (profile["housing_type"] == inputs["housing_type"]) &
            (profile["region"] == region)
        )
        display_profile = profile[[
            "credit_type", "housing_type", "region",
            "count", "median_award", "p10_award", "p90_award",
            "median_units", "median_year",
        ]].rename(columns={
            "credit_type":  "Credit",
            "housing_type": "Housing",
            "region":       "Region",
            "count":        "Count",
            "median_award": "Median award",
            "p10_award":    "Lower end award (p10)",
            "p90_award":    "Upper end award (p90)",
            "median_units": "Median units",
            "median_year":  "Median year",
        })
        styled = (
            display_profile.style
            .hide(axis="index")
            .format({
                "Count":                  "{:,d}",
                "Median award":           "${:,.0f}",
                "Lower end award (p10)":  "${:,.0f}",
                "Upper end award (p90)":  "${:,.0f}",
                "Median units":           "{:.0f}",
                "Median year":            "{:.0f}",
            })
            .apply(
                lambda r: [f"background-color: {PALETTE['paper_alt']}; color: {PALETTE['oxblood_dk']}; font-weight: 600" if user_mask.iloc[r.name] else ""] * len(r),
                axis=1,
            )
        )
        st.dataframe(styled, width="stretch", hide_index=True)

    # --- Trend chart: credit per unit over time for this county + housing type ---
    st.subheader("Credit Trend — Your County & Housing Type")
    st.caption(
        "Annual medians are noisy at the county level (small n per year), so the "
        "Project Group panel above aggregates 2015–2025 to provide a more stable reference."
    )
    county_ht = df[
        (df["county"] == inputs["county"]) &
        (df["housing_type"] == inputs["housing_type"])
    ].copy()

    if len(county_ht) > 2:
        trend = county_ht.groupby("pis_year").agg(
            median_credit=("annual_federal_award", "median"),
            count=("annual_federal_award", "count"),
        ).reset_index()

        fig_trend = px.line(
            trend, x="pis_year", y="median_credit",
            markers=True,
            labels={"pis_year": "Year", "median_credit": "Median Annual Credit ($)"},
            title=f"{inputs['housing_type']} in {inputs['county']} — Median Annual Credit Over Time",
        )
        fig_trend.update_layout(yaxis_tickformat="$,.0f")
        st.plotly_chart(fig_trend, width="stretch")
    else:
        st.warning(
            f"Not enough historical data for {inputs['housing_type']} "
            f"in {inputs['county']} to show a trend."
        )

    # --- Map of comparable projects ---
    st.subheader("Geographic Distribution of Comparables")
    map_data = comps[["project_name", "county", "annual_federal_award", "pis_year"]].copy()
    map_data["lat"] = map_data["county"].map(lambda c: CA_COUNTY_COORDS.get(c, (36.7, -119.8))[0])
    map_data["lon"] = map_data["county"].map(lambda c: CA_COUNTY_COORDS.get(c, (36.7, -119.8))[1])

    # Add the user's project
    user_coords = CA_COUNTY_COORDS.get(inputs["county"], (36.7, -119.8))
    user_row = pd.DataFrame([{
        "project_name": "YOUR PROJECT",
        "county": inputs["county"],
        "annual_federal_award": 0,
        "pis_year": inputs["pis_year"],
        "lat": user_coords[0],
        "lon": user_coords[1],
    }])
    map_data = pd.concat([map_data, user_row], ignore_index=True)
    map_data["type"] = ["Comparable"] * (len(map_data) - 1) + ["Your Project"]

    fig_map = px.scatter_map(
        map_data,
        lat="lat", lon="lon",
        color="type",
        hover_name="project_name",
        hover_data={"county": True, "pis_year": True, "lat": False, "lon": False},
        color_discrete_map={"Comparable": PALETTE["navy"], "Your Project": PALETTE["oxblood"]},
        size_max=15,
        zoom=5,
        center={"lat": 36.7, "lon": -119.8},
        map_style="carto-positron",
        title="Comparable Projects Map",
    )
    fig_map.update_traces(marker=dict(size=12))
    st.plotly_chart(fig_map, width="stretch")

    # --- Download comparables ---
    st.download_button(
        "Download Comparables (CSV)",
        display_df.to_csv(index=False),
        "comparables.csv",
        "text/csv",
    )


# ---------------------------------------------------------------------------
# Tab 3: Credit Trend Analysis
# ---------------------------------------------------------------------------
def render_trends(df):
    st.header("Credit Trend Analysis")
    st.caption(
        "Historical trends in annual federal tax credit allocations across California."
    )

    # --- Statewide trend by credit type ---
    st.subheader("Statewide: Annual Credit per Unit by Credit Type")
    state_trend = df.groupby(["pis_year", "credit_type"]).agg(
        median_cpu=("credit_per_unit", "median"),
        count=("credit_per_unit", "count"),
    ).reset_index()

    fig_state = px.line(
        state_trend, x="pis_year", y="median_cpu", color="credit_type",
        markers=True,
        labels={
            "pis_year": "Year",
            "median_cpu": "Median Credit per Unit ($)",
            "credit_type": "Credit Type",
        },
        color_discrete_map={"9%": PALETTE["oxblood"], "4%": PALETTE["navy"]},
    )
    fig_state.update_layout(yaxis_tickformat="$,.0f")
    st.plotly_chart(fig_state, width="stretch")
    st.caption(
        "Per-unit credits have roughly tripled since 2000, with 9% credits "
        "consistently exceeding 4% bond deals by 30–50% — a trend the model "
        "learns to predict 2026 values."
    )

    # --- County comparison ---
    st.subheader("County Comparison")
    all_counties = sorted(df["county"].dropna().unique())
    default_counties = ["Los Angeles", "San Francisco", "San Diego"]
    defaults_available = [c for c in default_counties if c in all_counties]

    selected_counties = st.multiselect(
        "Select counties to compare (2-5 recommended)",
        all_counties,
        default=defaults_available[:3],
    )

    if selected_counties:
        county_trend = df[df["county"].isin(selected_counties)].groupby(
            ["pis_year", "county"]
        ).agg(
            median_credit=("annual_federal_award", "median"),
        ).reset_index()

        fig_county = px.line(
            county_trend, x="pis_year", y="median_credit", color="county",
            markers=True,
            labels={
                "pis_year": "Year",
                "median_credit": "Median Annual Credit ($)",
                "county": "County",
            },
        )
        fig_county.update_layout(yaxis_tickformat="$,.0f")
        st.plotly_chart(fig_county, width="stretch")
        st.caption(
            "Pick 2–5 counties to compare. Yearly medians can be spiky in "
            "smaller counties — fewer projects per year."
        )

    # --- Housing type breakdown ---
    st.subheader("Credit per Unit by Housing Type")
    ht_credit_type = st.radio(
        "Credit Type",
        options=["9%", "4%"],
        horizontal=True,
        key="trends_ht_credit_type",
        help="9% credits are competitive and typically larger; 4% credits are bond-financed and smaller. Filter to compare housing-type trends within one credit class.",
    )
    ht_df = df[df["credit_type"] == ht_credit_type]

    # Default to 3 of 6 types — all six at once is unreadable spaghetti, and
    # some types are sparse within a credit class (9% SRO mostly stops after
    # 2019; 9% Non-Targeted ends in 2004). The rest are one click away.
    all_types = [t for t in HOUSING_TYPE_COLORS if t in set(ht_df["housing_type"].unique())]
    default_types = [t for t in ("Large Family", "Senior", "SRO") if t in all_types]
    selected_types = st.multiselect(
        "Housing types to show",
        all_types,
        default=default_types,
        key="trends_ht_types",
    )

    if selected_types:
        ht_trend = ht_df[ht_df["housing_type"].isin(selected_types)].groupby(
            ["pis_year", "housing_type"]
        ).agg(
            median_cpu=("credit_per_unit", "median"),
        ).reset_index()

        fig_ht = px.line(
            ht_trend, x="pis_year", y="median_cpu", color="housing_type",
            markers=True,
            labels={
                "pis_year": "Year",
                "median_cpu": "Median Credit per Unit ($)",
                "housing_type": "Housing Type",
            },
            color_discrete_map=HOUSING_TYPE_COLORS,
        )
        fig_ht.update_traces(line=dict(width=2.2), marker=dict(size=7))
        fig_ht.update_layout(
            yaxis_tickformat="$,.0f",
            title=f"Median Credit per Unit · {ht_credit_type} credits only",
        )
        st.plotly_chart(fig_ht, width="stretch")
        st.caption(
            "Showing 3 of 6 types by default — add the rest above. Short or broken "
            "lines just mean few projects of that type in this credit class."
        )
    else:
        st.info("Pick at least one housing type above.")

    # --- Volume over time ---
    st.subheader("Project Volume by Year")
    # Stacked by credit type: volume AND composition in one chart, colors
    # matching the statewide credit-type chart above.
    vol = df.groupby(["pis_year", "credit_type"]).size().reset_index(name="Projects")
    fig_vol = px.bar(
        vol, x="pis_year", y="Projects", color="credit_type",
        labels={"pis_year": "Year", "Projects": "Number of Projects", "credit_type": "Credit Type"},
        color_discrete_map={"9%": PALETTE["oxblood"], "4%": PALETTE["navy"]},
    )
    fig_vol.update_layout(barmode="stack")
    st.plotly_chart(fig_vol, width="stretch")
    st.caption(
        "Volume dipped after the 2008 financial crisis and surged in 2023–24 — "
        "with 4% bond deals driving most of the recent growth."
    )


# ---------------------------------------------------------------------------
# Tab 4: Method Insight
# ---------------------------------------------------------------------------
TRAIN_END_YEAR = 2021  # the boundary used by NB03's time-based split


def _render_train_test_timeline(df_train, df_test):
    """§1 chart 1 — Gantt-style horizontal bar showing the 2000–2021 train
    window vs the 2022–2025 test window. Establishes 'time-based split, not
    random' before the distribution-shift chart that follows."""
    train_start, train_end = int(df_train["pis_year"].min()), int(df_train["pis_year"].max())
    test_start, test_end = int(df_test["pis_year"].min()), int(df_test["pis_year"].max())

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=["Train"], x=[train_end - train_start + 1], base=[train_start],
        orientation="h", marker_color=PALETTE["deep"],
        text=[f"  n = {len(df_train):,}  ·  {train_end - train_start + 1} yrs  "],
        textposition="inside", insidetextanchor="middle",
        textfont=dict(color=PALETTE["paper"], size=12),
        hovertemplate=f"Train · {train_start}–{train_end} · n={len(df_train):,}<extra></extra>",
        showlegend=False,
    ))
    fig.add_trace(go.Bar(
        y=["Test"], x=[test_end - test_start + 1], base=[test_start],
        orientation="h", marker_color=PALETTE["signal"],
        text=[f"  n = {len(df_test):,}  ·  {test_end - test_start + 1} yrs  "],
        textposition="inside", insidetextanchor="middle",
        textfont=dict(color=PALETTE["paper"], size=12),
        hovertemplate=f"Test · {test_start}–{test_end} · n={len(df_test):,}<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(
        height=220,
        xaxis=dict(range=[train_start - 0.5, test_end + 0.5], dtick=2, title="Placed-in-Service Year"),
        yaxis=dict(categoryorder="array", categoryarray=["Test", "Train"]),
        margin=dict(l=20, r=20, t=20, b=50),
        bargap=0.4,
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Time-based split — earlier years train, later years test. A random shuffle "
        "would leak future information into past predictions; this is a forecasting "
        "problem."
    )


def _render_distribution_shift(df_train, df_test):
    """§1 chart 2 — anchor figure of the methodology story. Median annual
    federal award by year, with each bar colored by train/test membership.
    The visual jump in 2022 is the central technical challenge."""
    df = pd.concat([df_train, df_test], ignore_index=True)
    yearly = (
        df.groupby("pis_year")["annual_federal_award"]
        .median().reset_index()
        .sort_values("pis_year")
    )
    colors = [
        PALETTE["deep"] if y <= TRAIN_END_YEAR else PALETTE["signal"]
        for y in yearly["pis_year"]
    ]

    train_mean = df_train["annual_federal_award"].mean()
    test_mean = df_test["annual_federal_award"].mean()
    shift_ratio = test_mean / train_mean

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=yearly["pis_year"], y=yearly["annual_federal_award"] / 1e6,
        marker_color=colors,
        hovertemplate="Year %{x}<br>Median award: $%{y:.2f}M<extra></extra>",
        showlegend=False,
    ))
    # Train/test boundary marker.
    fig.add_vline(
        x=TRAIN_END_YEAR + 0.5,
        line=dict(color=PALETTE["ink_mute"], width=1, dash="dash"),
        annotation_text="Train / Test split",
        annotation_position="top right",
        annotation=dict(font_size=11, font_color=PALETTE["ink_soft"]),
    )
    # Manual legend traces (invisible bars at zero just to register the keys).
    fig.add_trace(go.Bar(
        x=[None], y=[None], marker_color=PALETTE["deep"],
        name=f"Train (≤{TRAIN_END_YEAR})", showlegend=True,
    ))
    fig.add_trace(go.Bar(
        x=[None], y=[None], marker_color=PALETTE["signal"],
        name=f"Test (≥{TRAIN_END_YEAR + 1})", showlegend=True,
    ))
    fig.update_layout(
        height=380,
        xaxis_title="Placed-in-Service Year",
        yaxis_title="Median Annual Federal Credit ($M)",
        yaxis_tickformat="$.2fM",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=50),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Test mean is **{shift_ratio:.2f}×** the train mean — post-COVID construction-cost "
        f"escalation. Bounds calibrated only to historical data systematically under-cover "
        f"modern projects, which is why the deployed interval uses split conformal (§3) "
        f"rather than naive quantile regression."
    )


def _render_actual_vs_predicted(preds, model_comparison):
    """§2 chart 1 — Plotly scatter of test-set predictions on log-log axes,
    colored by year so the post-2022 drift below the diagonal is visible.
    Title carries R²/MAPE/within-10% from model_comparison.csv for trust."""
    xgb_row = model_comparison[model_comparison["Model"] == "XGBoost"].iloc[0]
    r2 = xgb_row["R2"]
    mape = xgb_row["MAPE (%)"]

    fig = px.scatter(
        preds, x="actual", y="predicted", color="pis_year",
        color_continuous_scale="Plasma", opacity=0.6,
        labels={"actual": "Actual Award ($)", "predicted": "Predicted Award ($)", "pis_year": "PIS Year"},
    )
    fig.update_traces(marker=dict(size=6))

    # Linear axes with evenly spaced $1M ticks ($0 to $8M covers test set spread
    # of ~$220K – $7.4M). Linear (not log) so tick *intervals* are equal — the
    # earlier log-decade ticks were unevenly spaced (10× per step) and confusing.
    axis_max = 8e6
    tick_step = 1e6
    tickvals = [i * tick_step for i in range(int(axis_max / tick_step) + 1)]
    ticktext = [f"${int(v / 1e6)}M" if v > 0 else "$0" for v in tickvals]
    linear_axis = dict(
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        range=[0, axis_max],
    )

    # y = x reference line spanning the full plot, drawn AFTER axes so it
    # picks up the linear range cleanly.
    fig.add_trace(go.Scatter(
        x=[0, axis_max], y=[0, axis_max], mode="lines",
        line=dict(color=PALETTE["ink_mute"], dash="dash", width=1),
        name="y = x", showlegend=False, hoverinfo="skip",
    ))
    fig.update_layout(
        height=460,
        title=f"XGBoost on 2022–2025 test set  ·  R² = {r2:.2f}  ·  MAPE = {mape:.1f}%",
        margin=dict(l=60, r=20, t=70, b=60),
    )
    fig.update_xaxes(**linear_axis)
    fig.update_yaxes(**linear_axis)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Each point is one held-out 2022–2025 project. Points below the diagonal = model "
        "under-predicted; above = over-predicted. The systematic drift of recent (lighter) "
        "points reflects the distribution shift seen in §1."
    )


def _render_global_shap(shap_df, n_sample, top_n=15):
    """§2 chart 2 — top-N features by mean |SHAP|, sorted descending.
    The static counterpart to the per-prediction SHAP shown in Tab 1."""
    top = shap_df.head(top_n).copy()

    fig = go.Figure(go.Bar(
        x=top["mean_abs_shap"][::-1],
        y=top["display_name"][::-1],
        orientation="h",
        marker_color=PALETTE["deep"],
        text=[f"{v:.3f}" for v in top["mean_abs_shap"][::-1]],
        textposition="outside",
        hovertemplate="%{y}<br>mean |SHAP| = %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        height=max(360, top_n * 28),
        xaxis_title="Mean |SHAP|  (impact on log-credit prediction)",
        yaxis=dict(autorange=None),  # already pre-reversed
        margin=dict(l=20, r=60, t=30, b=50),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Mean absolute SHAP across {n_sample} test samples. Higher = bigger average "
        f"impact on the log-credit prediction. The county target-encoding and total-units "
        f"features dominate; macro PPI and FMR features show up but smaller."
    )


def _render_conformal_coverage(preds, conformal):
    """§3 — the deployed prediction interval, applied to the 2022–2025 test
    set. Each x-position is one project sorted by actual; the shaded band is
    the 90% conformal interval around the XGBoost point estimate; black dots
    are realized awards. Coverage = the fraction of dots inside the band."""
    sorted_preds = preds.sort_values("actual").reset_index(drop=True)
    x = np.arange(len(sorted_preds))
    coverage = sorted_preds["in_interval"].mean() * 100
    target = int(round((1 - conformal["alpha"]) * 100))
    mult_low = conformal["multiplier_low"]
    mult_high = conformal["multiplier_high"]

    fig = go.Figure()
    # Upper bound (invisible line, anchor for fill).
    fig.add_trace(go.Scatter(
        x=x, y=sorted_preds["upper"] / 1e6, mode="lines",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
    ))
    # Lower bound + fill to the upper trace.
    fig.add_trace(go.Scatter(
        x=x, y=sorted_preds["lower"] / 1e6, mode="lines",
        line=dict(color="rgba(0,0,0,0)"),
        fill="tonexty", fillcolor="rgba(168, 200, 224, 0.35)",
        name=f"{target}% conformal band  ·  pred × [{mult_low:.2f}, {mult_high:.2f}]",
        hoverinfo="skip",
    ))
    # XGBoost point estimate.
    fig.add_trace(go.Scatter(
        x=x, y=sorted_preds["predicted"] / 1e6, mode="lines",
        line=dict(color=PALETTE["deep"], width=1.4), name="XGBoost point estimate",
        hovertemplate="Predicted: $%{y:.2f}M<extra></extra>",
    ))
    # Actual awards.
    fig.add_trace(go.Scatter(
        x=x, y=sorted_preds["actual"] / 1e6, mode="markers",
        marker=dict(color=PALETTE["ink"], size=4, opacity=0.7),
        name="Actual award",
        hovertemplate="Actual: $%{y:.2f}M<extra></extra>",
    ))
    fig.update_layout(
        height=440,
        title=f"Split Conformal on XGBoost  ·  {coverage:.1f}% coverage on a {target}% target interval",
        xaxis_title="Test observations (sorted by actual award)",
        yaxis_title="Annual Federal Credit ($M)",
        yaxis_tickformat="$.1fM",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=70, b=50),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Each x-position is one of {len(sorted_preds)} held-out 2022–2025 projects, sorted "
        f"by realized award. The blue band is the 90% prediction interval the Estimator "
        f"tab shows users. Calibrated on 2020–2021 holdout residuals (n = {conformal['n_cal']}); "
        f"on the test set it covers {coverage:.1f}% of actuals — close to the {target}% nominal "
        f"target despite a 2.26× distribution shift."
    )


def _render_overfit_shift_table():
    """§2 table — the three-stage R² decomposition from NB04 §10a.

    train−CV isolates in-distribution overfit; CV−test isolates what the 2.26×
    distribution shift actually cost. Values are hardcoded from the NB04 §10a
    run — regenerate there if the models are retrained.
    """
    st.markdown(
        "**Overfit vs. distribution shift — where the test gap really comes from (R²)**"
    )
    diag = pd.DataFrame({
        "Model": ["Random Forest", "XGBoost (deployed)", "LightGBM p50", "Stacking v2"],
        "Train": [0.936, 0.867, 0.898, 0.897],
        "CV": [0.402, 0.597, 0.550, 0.601],
        "Test": [0.399, 0.620, 0.546, 0.656],
        "Overfit (Train−CV)": [0.535, 0.270, 0.347, 0.296],
        "Shift (CV−Test)": [0.002, -0.023, 0.004, -0.054],
    })
    st.dataframe(diag, hide_index=True, width="stretch")
    st.caption(
        "For every tree model the shift column is basically zero — the gap between "
        "train and test R² is in-distribution overfit, not the 2.26× shift. That's "
        "why improvement effort goes to features and data volume, not shift "
        "correction, and XGBoost's controlled overfit is one reason it's the "
        "deployed model. (Ridge is omitted: its CV R² of −28.9 comes from linear "
        "extrapolation blow-ups on the log scale; full table in NB04 §10a.)"
    )


def render_method_insight(models, df_train, df_test, model_comparison):
    st.header("Method Insight")
    st.caption("How the model was built and validated — for the technical reader.")

    # § 1
    st.subheader("§ 1   Data & The Time-Split Decision")
    _render_train_test_timeline(df_train, df_test)
    _render_distribution_shift(df_train, df_test)

    # § 2
    st.subheader("§ 2   Modeling — Does It Fit?")
    preds = compute_test_predictions(models, df_test)
    _render_actual_vs_predicted(preds, model_comparison)
    _render_overfit_shift_table()
    shap_df = compute_global_shap(models, df_test)
    _render_global_shap(shap_df, n_sample=200)

    # § 3
    st.subheader("§ 3   Uncertainty Quantification")
    _render_conformal_coverage(preds, models["conformal"])


# ---------------------------------------------------------------------------
# Model Card
# ---------------------------------------------------------------------------
def render_model_card(models, model_comparison):
    st.markdown("---")
    with st.expander("Model Card — Data & Methodology"):
        xgb_row = model_comparison[model_comparison["Model"] == "XGBoost"].iloc[0]
        conf = models["conformal"]
        coverage_target = int(round((1 - conf["alpha"]) * 100))

        st.markdown(f"""
**Training Data:** 3,536 CA LIHTC projects, placed in service 2000–2021
**Test Data:** 851 projects, placed in service 2022–2025
**Target Variable:** Annual Federal Tax Credit Award (log-transformed for training)

**Deployed point estimator:** XGBoost — R² = {xgb_row['R2']:.2f}, MAPE = {xgb_row['MAPE (%)']:.1f}%, within 10% of actual on {xgb_row['Within 10%']:.1f}% of test rows.

**Deployed prediction interval:** Split conformal on XGBoost (log-space) — multiplicative band `pred × [{conf['multiplier_low']:.2f}, {conf['multiplier_high']:.2f}]` ({conf['pct_low']:+.0f}% / {conf['pct_high']:+.0f}%), {coverage_target}% target coverage, {conf['coverage_test']:.1f}% achieved on the 2022–2025 test set; mean width ${conf['mean_width']:,.0f}; calibrated on the 2020–2021 holdout (n_cal = {conf['n_cal']}). Stacking v2 and the LightGBM quantile model are reported as offline benchmarks but not loaded by the app.

**Data Sources:**
- CTCAC Project Database (primary — 6,103 projects, 1989–2025)
- FRED PPI WPUSI012011 (construction cost index)
- HUD Fair Market Rents (county-level rent ceilings)
- Census ACS 5-Year (county income, rent, poverty)

**Known Limitations:**
- The 2022–2025 test period saw a **2.26× distribution shift** in credit amounts vs training (post-COVID construction cost escalation). Model accuracy may vary for projects unlike the training distribution.
- The conformal interval is *marginally* calibrated across the whole test set — coverage on individual subgroups (county, region, credit type) may differ. A Mondrian (group-conditional) variant is on the roadmap.
- County-level features use historical medians as proxies for future values.
- Implied Eligible Basis is a derived proxy, not a direct cost figure.
        """)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    render_masthead(
        title_main="Affordable Housing",
        title_em="Tax Credit Intelligence",
        dek=(
            "Forecasting annual federal LIHTC allocations for California — "
            "with calibrated prediction intervals, cost proxies, and market comparables, "
            "drawn from twenty-five years of CTCAC filings."
        ),
        kicker="California · LIHTC Intelligence",
    )

    # Load everything
    models = load_models()
    df_train, df_test = load_project_splits()
    df = load_project_data()
    ppi_df = load_ppi()
    lookups = build_lookups(df)
    model_comparison = load_model_comparison()

    # Sidebar inputs (shared across tabs)
    inputs = render_sidebar(lookups)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "I · Estimator",
        "II · Comparables",
        "III · Trends",
        "IV · Method Insight",
    ])

    with tab1:
        render_credit_estimator(models, lookups, ppi_df, inputs)

    with tab2:
        render_comparables(models, df, lookups, ppi_df, inputs)

    with tab3:
        render_trends(df)

    with tab4:
        render_method_insight(models, df_train, df_test, model_comparison)

    # Model card at the bottom
    render_model_card(models, model_comparison)


if __name__ == "__main__":
    main()
