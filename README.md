# CA Affordable Housing Tax Credit Intelligence Platform

A data-driven ML platform that predicts **annual federal tax credit allocations** for California Low-Income Housing Tax Credit (LIHTC) projects — with confidence ranges, cost proxies, and market comparables.

> Built for developers, investors, and housing finance agencies who need a data-driven sanity check on credit requests before submitting to CTCAC.

---

## What It Does

Given a project's characteristics (county, size, unit mix, credit type, housing type), the platform outputs:

1. **Expected annual federal credit** — p10 / p50 / p90 confidence range
2. **Implied eligible basis** — a development cost proxy derived from the predicted credit (`credit / rate`)
3. **Market comparables** — the 10 most similar historical CA LIHTC projects
4. **SHAP explanation** — which features drove the prediction and by how much

---

## Results (Test Set: 2022–2025)

| Model | R² | MAPE | Within 10% |
|---|---|---|---|
| Ridge (baseline) | -0.06 | 43.8% | 19.3% |
| Random Forest | 0.39 | 29.6% | 14.9% |
| **XGBoost** | **0.63** | **26.0%** | **22.6%** |
| Stacking Ensemble | 0.59 | 29.0% | 26.3% |
| LightGBM p50 | 0.54 | 26.8% | 21.6% |

Test set (2022–2025) has a **2.26× distribution shift** vs training (2000–2021) — real post-COVID construction cost escalation, not a data issue.

---

## Data Sources

| Source | What | Used For |
|---|---|---|
| [CTCAC Project Database](https://www.treasurer.ca.gov/ctcac/projects.asp) | 6,103 CA LIHTC projects, 1989–2025 | Primary — target variable + all project features |
| [FRED WPUSI012011](https://fred.stlouisfed.org/series/WPUSI012011) | Producer Price Index for construction | Temporal cost environment features |
| [HUD Fair Market Rents](https://www.huduser.gov/portal/datasets/fmr.html) | Annual 2BR FMR by county | Local rent ceiling proxy |
| [Census ACS 5-Year](https://api.census.gov/data/key_signup.html) | County-level income, rent, poverty | Local economic conditions |

---

## Pipeline

```
01_data_process.ipynb    →  ctcac_clean.csv         (5,496 rows)
02_enrich_data.ipynb     →  lihtc_ca_clean.parquet  (+ PPI, FMR, Census)
03_feature_engineering.ipynb  →  feature_pipeline.pkl   (ColumnTransformer, fit on train only)
04_modeling.ipynb        →  models/*.pkl            (Ridge, RF, XGBoost, Stacking, LightGBM, KNN, KMeans)
streamlit_app.py         →  interactive demo
```

**Train/test split:** Time-based — train 2000–2021 (3,536 rows), test 2022–2025 (851 rows). No random splitting — this is a temporal prediction problem.

---

## Models

| Model | Role |
|---|---|
| Ridge | Interpretable baseline |
| Random Forest | Non-linear baseline |
| XGBoost | Best single predictor (gradient boosting) |
| Stacking (Ridge + RF + XGBoost → Ridge meta) | Ensemble |
| LightGBM Quantile (p10/p50/p90) | Confidence range output |
| KNN (cosine, k=10) | Market comparables retrieval |
| KMeans (k=6) | Project archetype clustering |

---

## Feature Engineering Highlights

- **TargetEncoder** for `county` and `region` — avoids 50+ sparse one-hot columns
- **StandardScaler + median imputation** for 15 numeric features
- **OneHotEncoder** for `credit_type`, `construction_type`, `housing_type`
- **ColumnTransformer fit on training split only** — strict leakage prevention
- **Log-transform target** `log1p(annual_federal_award)` — reduces skew from 4.31 → -0.28
- **Walk-forward TimeSeriesSplit(n_splits=4)** for hyperparameter tuning

---

## Setup

```bash
# Clone
git clone https://github.com/lilhuang15/ca-affordable-housing-tax-credit-intelligence.git
cd ca-affordable-housing-tax-credit-intelligence

# Install dependencies
pip install -r requirements.txt

# Set Census API key (free signup at api.census.gov)
export CENSUS_API_KEY=your_key_here

# Run pipeline in order (downloads data first — see data sources above)
jupyter nbconvert --to notebook --execute data/01_data_process.ipynb
jupyter nbconvert --to notebook --execute data/02_enrich_data.ipynb
jupyter nbconvert --to notebook --execute data/03_feature_engineering.ipynb
jupyter nbconvert --to notebook --execute data/04_modeling.ipynb

# Launch app
streamlit run streamlit_app.py
```

---

## Repository Structure

```
data/
├── 01_data_process.ipynb          # Clean CTCAC data
├── 02_enrich_data.ipynb           # Join PPI, FMR, Census
├── 03_feature_engineering.ipynb   # Scope filter, train/test split, ColumnTransformer
├── 04_modeling.ipynb              # Train all models, SHAP, save .pkl files
└── fred_ppi.csv                   # FRED PPI (committed — small, stable)

models/
├── feature_pipeline.pkl           # Fitted ColumnTransformer
├── feature_config.pkl             # Feature names and split parameters
├── xgb_model.pkl                  # XGBoost (best single model)
├── credit_quantile.pkl            # LightGBM p10/p50/p90
├── knn_model.pkl                  # KNN for comparables
├── kmeans_model.pkl               # KMeans for archetypes
├── ridge_model.pkl                # Ridge baseline
└── shap_explainer.pkl             # SHAP TreeExplainer for XGBoost

docs/
├── Pipeline_Technical_Guide.md    # Full technical explanation of every pipeline decision
└── Pipeline_Technical_Guide.docx

streamlit_app.py                   # Streamlit demo (Phase 1)
requirements.txt
Blueprint_v3.md                    # Full project specification
```

> Large files excluded from git: `data/ctcac_projects.xlsx`, `data/FMR_All_1983_2026.csv`, `models/credit_model.pkl` (150MB Stacking ensemble), `models/rf_model.pkl` (75MB). Re-generate by running the notebooks.

---

## Phase 2 (Planned)

An AI Credit Advisor tab using Anthropic tool-calling (`claude-haiku-4-5`) with:
- 4 tools: `predict_credit_amount`, `get_comparables`, `get_construction_index`, `compliance_qa`
- ChromaDB + BM25 hybrid RAG over CTCAC/IRS compliance documents
- Conversation memory and structured citations

---

## Tech Stack

`pandas` · `scikit-learn` · `xgboost` · `lightgbm` · `category_encoders` · `shap` · `plotly` · `streamlit`
