"""
CA Affordable Housing Tax Credit Intelligence Platform
======================================================
Streamlit app — Phase 1 (3 tabs).

Tab 1: Credit Estimator     — predict p10/p50/p90 annual federal credit
Tab 2: Market Comparables   — KNN top-10 + KMeans archetype + map
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
    models["quantile"] = joblib.load(os.path.join(MODEL_DIR, "credit_quantile.pkl"))
    models["knn"] = joblib.load(os.path.join(MODEL_DIR, "knn_model.pkl"))
    models["kmeans"] = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))
    models["pipeline"] = joblib.load(os.path.join(MODEL_DIR, "feature_pipeline.pkl"))
    models["config"] = joblib.load(os.path.join(MODEL_DIR, "feature_config.pkl"))
    models["shap"] = joblib.load(os.path.join(MODEL_DIR, "shap_explainer.pkl"))
    return models


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
        if yr - 2 in ppi_annual.index:
            ppi_2yr = (ppi_val - ppi_annual[yr - 2]) / ppi_annual[yr - 2] * 100
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
    st.sidebar.caption("Deep = ≤30% AMI, Mid = 50-60% AMI. Must sum to ≤100%")
    deep_ami = st.sidebar.slider("Deep (≤30% AMI) %", 0, 100, 10, 5)
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

    # --- Quantile predictions (p10 / p50 / p90) ---
    q = models["quantile"]
    p10_log = q["p10"].predict(X)[0]
    p50_log = q["p50"].predict(X)[0]
    p90_log = q["p90"].predict(X)[0]

    # Enforce ordering
    ordered = np.sort([p10_log, p50_log, p90_log])
    p10 = np.expm1(ordered[0])
    p50 = np.expm1(ordered[1])
    p90 = np.expm1(ordered[2])

    # XGBoost point estimate
    xgb_pred = np.expm1(models["xgb"].predict(X)[0])

    # --- Key metrics ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("P10 (Conservative)", f"${p10:,.0f}")
    col2.metric("P50 (Median)", f"${p50:,.0f}")
    col3.metric("P90 (Optimistic)", f"${p90:,.0f}")
    col4.metric("XGBoost Estimate", f"${xgb_pred:,.0f}")

    # --- Confidence range bar chart ---
    st.subheader("Annual Federal Credit — Confidence Range")
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=["P10\n(Conservative)", "P50\n(Median)", "P90\n(Optimistic)"],
        y=[p10, p50, p90],
        marker_color=["#5499c7", "#2ecc71", "#e67e22"],
        text=[f"${p10:,.0f}", f"${p50:,.0f}", f"${p90:,.0f}"],
        textposition="outside",
    ))
    fig_bar.update_layout(
        yaxis_title="Annual Federal Credit ($)",
        yaxis_tickformat="$,.0f",
        height=400,
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- 10-year credit stream & equity ---
    st.subheader("10-Year Credit Stream & Equity Estimate")
    credit_price = st.slider(
        "Credit Price ($/credit)", 0.70, 1.00, 0.92, 0.01,
        help="Investor price per dollar of tax credit. Typical range: $0.85–$0.95.",
    )

    stream_col1, stream_col2, stream_col3 = st.columns(3)
    for col, label, annual in [
        (stream_col1, "Conservative (P10)", p10),
        (stream_col2, "Median (P50)", p50),
        (stream_col3, "Optimistic (P90)", p90),
    ]:
        ten_yr = annual * 10
        equity = ten_yr * credit_price
        col.markdown(f"**{label}**")
        col.markdown(f"- Annual credit: **${annual:,.0f}**")
        col.markdown(f"- 10-year stream: **${ten_yr:,.0f}**")
        col.markdown(f"- Equity @ ${credit_price:.2f}: **${equity:,.0f}**")

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
    basis_col1.metric("Implied Eligible Basis (P50)", f"${basis_p50:,.0f}")
    basis_col2.metric("Implied Basis per Unit", f"${basis_per_unit:,.0f}")
    basis_col3.metric("Credit Rate Used", f"{credit_rate:.0%}")

    # --- SHAP explanation ---
    st.subheader("What's Driving This Prediction?")
    st.caption("SHAP values show which features pushed the credit estimate up or down.")

    feature_names = models["config"]["feature_names_out"]
    shap_top = get_shap_explanation(models, X, feature_names)

    names = [n for n, _ in shap_top]
    values = [v for _, v in shap_top]
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in values]

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
    st.plotly_chart(fig_shap, use_container_width=True)

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
        "P10 Annual Credit": round(p10),
        "P50 Annual Credit": round(p50),
        "P90 Annual Credit": round(p90),
        "XGBoost Estimate": round(xgb_pred),
        "10yr P50 Stream": round(p50 * 10),
        "Implied Eligible Basis (P50)": round(basis_p50),
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
    # KNN was trained on the full training set transformed features
    X_train = np.load(os.path.join(MODEL_DIR, "X_train.npy"))
    distances, indices = models["knn"].kneighbors(X)
    neighbor_indices = indices[0]

    # Map back to training data rows
    df_train = pd.read_parquet(os.path.join(MODEL_DIR, "train_data.parquet"))
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

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # --- KMeans archetype ---
    st.subheader("Project Archetype")
    cluster_id = models["kmeans"].predict(X)[0]

    # Get archetype stats from training data
    X_train_all = np.load(os.path.join(MODEL_DIR, "X_train.npy"))
    train_clusters = models["kmeans"].predict(X_train_all)
    cluster_mask = train_clusters == cluster_id
    cluster_projects = df_train[cluster_mask]

    if len(cluster_projects) > 0:
        avg_credit = cluster_projects["annual_federal_award"].mean()
        med_credit = cluster_projects["annual_federal_award"].median()
        min_credit = cluster_projects["annual_federal_award"].min()
        max_credit = cluster_projects["annual_federal_award"].max()
        avg_units = cluster_projects["total_units"].mean()
        top_county = cluster_projects["county"].mode().iloc[0] if len(cluster_projects) > 0 else "N/A"
        top_housing = cluster_projects["housing_type"].mode().iloc[0] if len(cluster_projects) > 0 else "N/A"

        st.info(
            f"**Cluster {cluster_id + 1}** — {len(cluster_projects)} similar projects in training data\n\n"
            f"- Typical profile: **{top_housing}** in **{top_county}** area, ~{avg_units:.0f} units\n"
            f"- Annual credit range: **${min_credit:,.0f}** – **${max_credit:,.0f}**\n"
            f"- Median annual credit: **${med_credit:,.0f}**"
        )

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
        st.plotly_chart(fig_trend, use_container_width=True)
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

    fig_map = px.scatter_mapbox(
        map_data,
        lat="lat", lon="lon",
        color="type",
        hover_name="project_name",
        hover_data={"county": True, "pis_year": True, "lat": False, "lon": False},
        color_discrete_map={"Comparable": "#5499c7", "Your Project": "#e74c3c"},
        size_max=15,
        zoom=5,
        center={"lat": 36.7, "lon": -119.8},
        mapbox_style="carto-positron",
        title="Comparable Projects Map",
    )
    fig_map.update_traces(marker=dict(size=12))
    st.plotly_chart(fig_map, use_container_width=True)

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
        color_discrete_map={"9%": "#2ecc71", "4%": "#5499c7"},
    )
    fig_state.update_layout(yaxis_tickformat="$,.0f")
    st.plotly_chart(fig_state, use_container_width=True)

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
        st.plotly_chart(fig_county, use_container_width=True)

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
    st.plotly_chart(fig_ht, use_container_width=True)

    # --- AMI depth trend ---
    st.subheader("Deep Affordability Targeting Over Time")
    st.caption("Average share of units targeting ≤30% AMI households, by year.")
    ami_trend = df.groupby("pis_year")["deep_ami_pct"].mean().reset_index()
    ami_trend.columns = ["Year", "Avg Deep AMI %"]
    ami_trend["Avg Deep AMI %"] *= 100

    fig_ami = px.bar(
        ami_trend, x="Year", y="Avg Deep AMI %",
        labels={"Avg Deep AMI %": "Avg % Units at ≤30% AMI"},
        color_discrete_sequence=["#8e44ad"],
    )
    fig_ami.update_layout(yaxis_tickformat=".1f")
    st.plotly_chart(fig_ami, use_container_width=True)

    # --- Volume over time ---
    st.subheader("Project Volume by Year")
    vol = df.groupby("pis_year").size().reset_index(name="Projects")
    fig_vol = px.bar(
        vol, x="pis_year", y="Projects",
        labels={"pis_year": "Year", "Projects": "Number of Projects"},
        color_discrete_sequence=["#1abc9c"],
    )
    st.plotly_chart(fig_vol, use_container_width=True)


# ---------------------------------------------------------------------------
# Model Card
# ---------------------------------------------------------------------------
def render_model_card():
    st.markdown("---")
    with st.expander("Model Card — Data & Methodology"):
        st.markdown("""
**Training Data:** 3,536 CA LIHTC projects, placed in service 2000–2021
**Test Data:** 851 projects, placed in service 2022–2025
**Target Variable:** Annual Federal Tax Credit Award (log-transformed for training)
**Best Model:** XGBoost (R² = 0.63, MAPE = 26.0% on test set)
**Quantile Model:** LightGBM (p10/p50/p90 confidence range)

**Data Sources:**
- CTCAC Project Database (primary — 6,103 projects, 1989–2025)
- FRED PPI WPUSI012011 (construction cost index)
- HUD Fair Market Rents (county-level rent ceilings)
- Census ACS 5-Year (county income, rent, poverty)

**Known Limitations:**
- The 2022–2025 test period saw a **2.26× distribution shift** in credit amounts vs training (post-COVID construction cost escalation). Model accuracy may vary for projects unlike the training distribution.
- County-level features use historical medians as proxies for future values.
- Implied Eligible Basis is a derived proxy, not a direct cost figure.

**Last Trained:** April 2026 | **Pipeline Version:** v1.0
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
    render_model_card()


if __name__ == "__main__":
    main()
