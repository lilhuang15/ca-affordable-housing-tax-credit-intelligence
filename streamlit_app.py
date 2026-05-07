"""
CA Affordable Housing Tax Credit Intelligence Platform
======================================================
Streamlit app — Phase 1 (3 tabs).

Tab 1: Credit Estimator      — XGBoost point estimate + split-conformal interval + SHAP
Tab 2: Market Comparables    — KNN top-10 + Project Group lookup + map
Tab 3: Credit Trend Analysis — statewide/county/housing-type trends

Run:  streamlit run streamlit_app.py
"""

import os
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CA LIHTC Credit Intelligence",
    page_icon="🏠",
    layout="wide",
)

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
def load_project_data():
    """Load the scoped project data (train + test) for comparables and trends."""
    df_train = pd.read_parquet(os.path.join(MODEL_DIR, "train_data.parquet"))
    df_test = pd.read_parquet(os.path.join(MODEL_DIR, "test_data.parquet"))
    df = pd.concat([df_train, df_test], ignore_index=True)
    df["county"] = df["county"].str.strip().str.title()
    return df


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

    return {
        "county_region": county_region,
        "county_fmr": county_fmr,
        "county_income": county_income,
        "county_rent": county_rent,
        "counties": counties,
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
    region = lookups["county_region"].get(county, "Other Northern CA")
    fmr = lookups["county_fmr"].get(county, 1500.0)
    income = lookups["county_income"].get(county, 80000.0)
    rent = lookups["county_rent"].get(county, 1500.0)

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
# Helper: SHAP waterfall (top features)
# ---------------------------------------------------------------------------
def get_shap_explanation(models, X_transformed, feature_names, top_n=8):
    """Return SHAP values for a single observation."""
    shap_vals = models["shap"].shap_values(X_transformed)
    sv = shap_vals[0]  # single row

    # Pair feature names with SHAP values
    pairs = list(zip(feature_names, sv))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    top = pairs[:top_n]

    # Clean feature names for display
    clean_names = []
    for name, val in top:
        # Remove prefix like num__, te__, ohe__
        display = name.split("__", 1)[-1] if "__" in name else name
        display = display.replace("_", " ").title()
        clean_names.append((display, val))

    return clean_names


# ---------------------------------------------------------------------------
# Sidebar: shared project inputs
# ---------------------------------------------------------------------------
def render_sidebar(lookups):
    """Render sidebar inputs and return dict of user selections."""
    st.sidebar.header("Project Inputs")

    county = st.sidebar.selectbox(
        "County",
        lookups["counties"],
        index=lookups["counties"].index("Los Angeles")
        if "Los Angeles" in lookups["counties"] else 0,
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
        "Low-Income Units %", 50, 100, 95, 5,
        help="Percentage of total units restricted as low-income",
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

    # --- Prediction range bar chart ---
    st.subheader("Annual Federal Credit — Prediction Range")
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=[lower_label, "Point estimate", upper_label],
        y=[p10, p50, p90],
        marker_color=["#73a8cb", "#418bb0", "#184776"],
        text=[f"${p10:,.0f}", f"${p50:,.0f}", f"${p90:,.0f}"],
        textposition="outside",
    ))
    fig_bar.update_layout(
        yaxis_title="Annual Federal Credit ($)",
        yaxis_tickformat="$,.0f",
        height=400,
        showlegend=False,
    )
    st.plotly_chart(fig_bar, width="stretch")

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
        equity = ten_yr * credit_price
        # Escape every "$" as "\$" so streamlit doesn't enter LaTeX math mode
        # (two unescaped $ on a line trigger `$...$` math, which makes the text
        # in between render in a math font and breaks `**bold**` markers around
        # adjacent dollar amounts). Drop bold markers to keep the bullets plain.
        col.markdown(f"**{label}**")
        col.markdown(f"- Annual credit: \\${annual:,.0f}")
        col.markdown(f"- 10-year stream: \\${ten_yr:,.0f}")
        col.markdown(f"- Equity @ \\${credit_price:.2f}: \\${equity:,.0f}")

    # --- Implied Eligible Basis ---
    st.subheader("Implied Eligible Basis (Cost Proxy)")
    st.caption(
        "Derived as Annual Credit ÷ Credit Rate. This is a benchmarking proxy, "
        "not a direct development cost figure."
    )
    credit_rate = 0.09 if inputs["credit_type"] == "9%" else 0.04
    basis_p50 = p50 / credit_rate
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
    colors = ["#228ecd" if v > 0 else "#e77e3c" for v in values]

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
        "implied_basis_per_unit", "developer", "similarity",
    ]
    display_df = comps[display_cols].copy()
    display_df.columns = [
        "Project", "County", "Year", "Housing Type",
        "Construction", "Credit Type", "Units",
        "Annual Credit", "Credit/Unit",
        "Basis/Unit", "Developer", "Similarity",
    ]
    display_df["Annual Credit"] = display_df["Annual Credit"].apply(
        lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
    )
    display_df["Credit/Unit"] = display_df["Credit/Unit"].apply(
        lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
    )
    display_df["Basis/Unit"] = display_df["Basis/Unit"].apply(
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
        "Historical context from CA LIHTC projects placed in service 2015–2025 — "
        "the modern post-QAP-reform era. Earlier projects are excluded because "
        "their award levels and project sizes don't reflect post-2015 cost realities; "
        "for a 2026 query, a $300K award from a 2002 project is not a useful reference."
    )
    region = lookups["county_region"].get(inputs["county"], "Other Northern CA")
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
                lambda r: ["background-color: #fff3cd" if user_mask.iloc[r.name] else ""] * len(r),
                axis=1,
            )
        )
        st.dataframe(styled, width="stretch", hide_index=True)

    # --- Trend chart: credit per unit over time for this county + housing type ---
    st.subheader("Credit Trend — Your County & Housing Type")
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
        color_discrete_map={"Comparable": "#5499c7", "Your Project": "#df7020"},
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
        color_discrete_map={"9%": "#df7020", "4%": "#5499c7"},
    )
    fig_state.update_layout(yaxis_tickformat="$,.0f")
    st.plotly_chart(fig_state, width="stretch")

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

    # --- Housing type breakdown ---
    st.subheader("Credit per Unit by Housing Type")
    ht_trend = df.groupby(["pis_year", "housing_type"]).agg(
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
    )
    fig_ht.update_layout(yaxis_tickformat="$,.0f")
    st.plotly_chart(fig_ht, width="stretch")

    # --- AMI depth trend ---
    st.subheader("Deep Affordability Targeting Over Time")
    st.caption("Average share of units targeting ≤30% AMI households, by year.")
    ami_trend = df.groupby("pis_year")["deep_ami_pct"].mean().reset_index()
    ami_trend.columns = ["Year", "Avg Deep AMI %"]
    ami_trend["Avg Deep AMI %"] *= 100

    fig_ami = px.bar(
        ami_trend, x="Year", y="Avg Deep AMI %",
        labels={"Avg Deep AMI %": "Avg % Units at ≤30% AMI"},
        color_discrete_sequence=["#5499c7"],
    )
    fig_ami.update_layout(yaxis_tickformat=".1f")
    st.plotly_chart(fig_ami, width="stretch")

    # --- Volume over time ---
    st.subheader("Project Volume by Year")
    vol = df.groupby("pis_year").size().reset_index(name="Projects")
    fig_vol = px.bar(
        vol, x="pis_year", y="Projects",
        labels={"pis_year": "Year", "Projects": "Number of Projects"},
        color_discrete_sequence=["#5499c7"],
    )
    st.plotly_chart(fig_vol, width="stretch")


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
    st.title("CA Affordable Housing Tax Credit Intelligence")
    st.markdown(
        "Predict annual federal LIHTC credit allocations for California projects "
        "— with confidence ranges, cost proxies, and market comparables."
    )

    # Load everything
    models = load_models()
    df = load_project_data()
    ppi_df = load_ppi()
    lookups = build_lookups(df)
    model_comparison = load_model_comparison()

    # Sidebar inputs (shared across tabs)
    inputs = render_sidebar(lookups)

    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "💰 Credit Estimator",
        "🏘️ Market Comparables",
        "📈 Credit Trends",
    ])

    with tab1:
        render_credit_estimator(models, lookups, ppi_df, inputs)

    with tab2:
        render_comparables(models, df, lookups, ppi_df, inputs)

    with tab3:
        render_trends(df)

    # Model card at the bottom
    render_model_card(models, model_comparison)


if __name__ == "__main__":
    main()
