"""
Hydrography theme — light-blue / white / orange-on-alert.

Aesthetic: cool research-journal. White paper, layered blues for hierarchy,
hairline rules in pale blue, deep navy for type weight. Orange appears only
as a signal color: alerts, the user's own project, and one binary contrast
pair (SHAP +/-) where contrast is the whole point.

Exports:
    inject_theme()           — call once at app startup
    render_masthead(...)     — modern minimal header
    PALETTE                  — color tokens (keys preserved for back-compat
                                with streamlit_app.py)
    CHART_COLORS             — semantic chart color tokens
    PLOTLY_TEMPLATE          — registered as default plotly template
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st


# ---------------------------------------------------------------------------
# Color tokens
# ---------------------------------------------------------------------------
# Hydrography blues:
#   white → tint → mist → sky → tide → deep → ink
# Signal orange used only when contrast is the goal.
#
# Legacy key names (oxblood, navy, brass, moss, rust, paper_alt, ink_soft,
# ink_mute, oxblood_dk) are preserved and remapped so streamlit_app.py keeps
# working without changes.
PALETTE = {
    # Surfaces
    "paper":      "#FFFFFF",   # main canvas
    "paper_alt":  "#F4F8FC",   # sidebar / subtle panels (legacy key)
    "mist":       "#DCE9F4",   # metric hover, highlighted row bg

    # Blues (light → dark)
    "sky":        "#A8C8E0",
    "tide":       "#5B8FB9",
    "deep":       "#1F4E79",

    # Type
    "ink":        "#0E2238",   # body text — deep blue, not pure black
    "ink_soft":   "#1F4E79",   # secondary text (legacy key)
    "ink_mute":   "#5B8FB9",   # captions (legacy key)

    # Rules
    "rule":       "#C7D9E8",   # hairlines (pale blue, replaces black)
    "rule_strong":"#1F4E79",   # masthead rule

    # Signal — used only on alerts + the user's own project + SHAP contrast
    "signal":     "#E8742C",
    "signal_dk":  "#B85A1A",

    # ---- Legacy key aliases (back-compat for streamlit_app.py) ----
    "oxblood":    "#1F4E79",   # was primary accent; now deep blue
    "oxblood_dk": "#0E2238",   # was darker accent; now ink
    "navy":       "#A8C8E0",   # 4% credit & comparable map color → sky blue
    "brass":      "#5B8FB9",   # tertiary chart → tide
    "moss":       "#A8C8E0",   # lower-bound bar → sky
    "rust":       "#E8742C",   # SHAP negative — kept as orange for contrast
}


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,300;0,6..72,400;0,6..72,600;0,6..72,800;1,6..72,400;1,6..72,600&family=Public+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {{
  --paper:        {PALETTE['paper']};
  --paper-alt:    {PALETTE['paper_alt']};
  --mist:         {PALETTE['mist']};
  --sky:          {PALETTE['sky']};
  --tide:         {PALETTE['tide']};
  --deep:         {PALETTE['deep']};
  --ink:          {PALETTE['ink']};
  --ink-soft:     {PALETTE['ink_soft']};
  --ink-mute:     {PALETTE['ink_mute']};
  --rule:         {PALETTE['rule']};
  --rule-strong:  {PALETTE['rule_strong']};
  --signal:       {PALETTE['signal']};
  --signal-dk:    {PALETTE['signal_dk']};

  --serif:  'Newsreader', 'Source Serif 4', Georgia, serif;
  --sans:   'Public Sans', 'Helvetica Neue', system-ui, sans-serif;
  --mono:   'JetBrains Mono', 'SF Mono', Menlo, monospace;
}}

/* ---------- Canvas ---------- */
html, body, [data-testid="stAppViewContainer"], .stApp {{
  background: var(--paper) !important;
  color: var(--ink);
  font-family: var(--sans);
  font-feature-settings: "ss01";
}}

[data-testid="stHeader"] {{ background: transparent !important; border-bottom: none; }}
[data-testid="stToolbar"] {{ display: none; }}

.block-container {{
  padding-top: 1.6rem !important;
  padding-bottom: 5rem !important;
  max-width: 1180px;
}}

/* ---------- Typography ---------- */
h1, h2, h3, h4 {{
  font-family: var(--serif);
  color: var(--ink);
  letter-spacing: -0.012em;
  font-weight: 600;
}}
h1 {{ font-weight: 700; font-size: 2.4rem; line-height: 1.05; }}
h2 {{
  font-size: 1.55rem;
  font-weight: 600;
  margin-top: 2.4rem !important;
  padding-top: 1.0rem;
  border-top: 1px solid var(--rule);
  color: var(--deep);
}}
h3 {{ font-size: 1.1rem; font-weight: 600; margin-top: 1.6rem !important; color: var(--ink); }}

p, li, label, .stMarkdown, [data-testid="stMarkdownContainer"] {{
  font-family: var(--sans);
  color: var(--ink);
  font-size: 0.96rem;
  line-height: 1.55;
}}

[data-testid="stCaptionContainer"], .stCaption, small {{
  font-family: var(--sans) !important;
  color: var(--ink-mute) !important;
  font-size: 0.84rem !important;
  letter-spacing: 0.005em;
}}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {{
  background: var(--paper-alt) !important;
  border-right: 1px solid var(--rule);
}}
[data-testid="stSidebar"] > div:first-child {{ padding-top: 1.2rem; }}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
  font-family: var(--sans);
  border-top: none !important;
  padding-top: 0 !important;
  margin-top: 1.2rem !important;
}}
[data-testid="stSidebar"] h2 {{
  font-size: 0.74rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.18em;
  color: var(--deep);
}}
[data-testid="stSidebar"] hr {{
  border: none;
  border-top: 1px solid var(--rule);
  margin: 1.4rem 0 !important;
}}
[data-testid="stSidebar"] label {{
  font-size: 0.78rem !important;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--ink-soft) !important;
}}

/* ---------- Inputs ---------- */
[data-baseweb="select"] > div,
[data-baseweb="input"] input,
.stNumberInput input,
.stTextInput input {{
  background: var(--paper) !important;
  border: 1px solid var(--rule) !important;
  border-radius: 4px !important;
  font-family: var(--mono) !important;
  font-size: 0.88rem !important;
  color: var(--ink) !important;
}}
[data-baseweb="select"] > div:focus-within,
.stNumberInput input:focus,
.stTextInput input:focus {{
  border-color: var(--deep) !important;
  box-shadow: 0 0 0 2px rgba(31, 78, 121, 0.12);
}}

/* slider */
.stSlider [role="slider"] {{
  background: var(--deep) !important;
  border: 2px solid var(--paper) !important;
  box-shadow: 0 0 0 1px var(--deep);
}}
.stSlider [data-baseweb="slider"] > div > div > div {{
  background: var(--tide) !important;
}}

/* ---------- Buttons & download — sky-blue surface, deep-blue ink ---------- */
.stButton > button, .stDownloadButton > button {{
  background: var(--sky) !important;
  color: var(--deep) !important;
  border: 1px solid var(--tide) !important;
  border-radius: 4px !important;
  font-family: var(--sans) !important;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  font-size: 0.74rem !important;
  padding: 0.65rem 1.4rem !important;
  transition: all 0.15s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
  background: var(--tide) !important;
  color: var(--paper) !important;
  border-color: var(--tide) !important;
  transform: translateY(-1px);
}}
.stButton > button p, .stDownloadButton > button p {{
  color: inherit !important;
  font-weight: 700 !important;
}}

/* ---------- Metrics: airy ledger style ---------- */
[data-testid="stMetric"] {{
  background: transparent;
  border: none;
  border-top: 2px solid var(--deep);
  padding: 0.85rem 0.4rem 0.4rem 0;
}}
[data-testid="stMetricLabel"] {{
  font-family: var(--sans) !important;
  font-size: 0.7rem !important;
  font-weight: 700;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--ink-mute) !important;
}}
[data-testid="stMetricValue"] {{
  font-family: var(--serif) !important;
  font-size: 2.1rem !important;
  font-weight: 600;
  color: var(--deep) !important;
  letter-spacing: -0.015em;
  line-height: 1.05 !important;
  font-feature-settings: "lnum", "tnum";
}}
[data-testid="stMetricValue"] > div {{ font-family: var(--serif) !important; }}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {{
  gap: 0;
  background: transparent;
  border-bottom: 1px solid var(--rule);
  margin-bottom: 1.5rem;
}}
.stTabs [data-baseweb="tab"] {{
  background: transparent !important;
  border: none !important;
  border-radius: 0 !important;
  padding: 0.85rem 1.6rem !important;
  font-family: var(--sans) !important;
  font-size: 0.78rem !important;
  font-weight: 600;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--ink-mute) !important;
  position: relative;
  transition: color 0.15s;
}}
.stTabs [data-baseweb="tab"]:hover {{ color: var(--deep) !important; }}
.stTabs [aria-selected="true"] {{ color: var(--deep) !important; }}
.stTabs [aria-selected="true"]::after {{
  content: "";
  position: absolute;
  left: 0; right: 0; bottom: -1px;
  height: 3px;
  background: var(--deep);
}}
.stTabs [data-baseweb="tab-highlight"] {{ display: none !important; }}

/* ---------- Alerts: orange = warning, others stay blue ---------- */
[data-testid="stAlert"] {{
  background: var(--paper-alt) !important;
  border: 1px solid var(--rule) !important;
  border-left: 4px solid var(--deep) !important;
  border-radius: 4px !important;
  padding: 0.9rem 1.1rem !important;
  font-family: var(--sans);
}}
/* Warning + Error → orange signal */
[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]),
[data-testid="stAlert"]:has([data-testid="stAlertContentError"]) {{
  background: rgba(232, 116, 44, 0.06) !important;
  border-left-color: var(--signal) !important;
}}
[data-testid="stAlert"] [data-testid="stAlertContentInfo"],
[data-testid="stAlert"] [data-testid="stAlertContentWarning"],
[data-testid="stAlert"] [data-testid="stAlertContentError"] {{
  background: transparent !important;
}}

/* ---------- Expander ---------- */
[data-testid="stExpander"] {{
  background: transparent;
  border: 1px solid var(--rule) !important;
  border-radius: 4px !important;
  margin-top: 1rem;
}}
[data-testid="stExpander"] summary {{
  font-family: var(--sans) !important;
  font-size: 0.78rem !important;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  padding: 0.85rem 1.1rem !important;
  color: var(--ink) !important;
}}
[data-testid="stExpander"] summary:hover {{ color: var(--deep) !important; }}

/* ---------- Dataframes ---------- */
[data-testid="stDataFrame"] {{
  border: 1px solid var(--rule);
  border-radius: 4px;
}}
[data-testid="stDataFrame"] * {{
  font-family: var(--mono) !important;
  font-size: 0.82rem !important;
}}

/* ---------- Plotly chart container ---------- */
[data-testid="stPlotlyChart"] {{
  background: transparent;
  margin-top: 0.8rem;
}}

/* ---------- Masthead — modern minimal ---------- */
.masthead {{
  padding: 1.4rem 0 1.6rem 0;
  margin-bottom: 1.8rem;
  border-bottom: 2px solid var(--deep);
  position: relative;
}}
.masthead .kicker {{
  font-family: var(--sans);
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.32em;
  text-transform: uppercase;
  color: var(--deep);
  margin-bottom: 0.85rem;
  display: flex;
  align-items: center;
  gap: 0.7rem;
}}
.masthead .kicker::before {{
  content: "";
  width: 28px;
  height: 2px;
  background: var(--signal);
  display: inline-block;
}}
.masthead h1.title {{
  font-family: var(--serif);
  font-weight: 700;
  font-size: clamp(2.2rem, 4.4vw, 3.4rem);
  line-height: 1.0;
  letter-spacing: -0.022em;
  color: var(--ink);
  margin: 0 0 0.7rem 0;
}}
.masthead h1.title em {{
  font-style: italic;
  color: var(--deep);
  font-weight: 400;
}}
.masthead .dek {{
  font-family: var(--serif);
  font-style: italic;
  font-weight: 300;
  font-size: 1.1rem;
  color: var(--ink-soft);
  max-width: 720px;
  line-height: 1.4;
}}

/* hide Streamlit footer */
footer, [data-testid="stStatusWidget"] {{ display: none !important; }}
</style>
"""


def inject_theme() -> None:
    """Inject Hydrography theme CSS. Call once at app startup."""
    st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Masthead — modern minimal
# ---------------------------------------------------------------------------
def render_masthead(
    title_main: str = "Tax Credit",
    title_em: str = "Intelligence",
    dek: str = (
        "Forecasting annual federal LIHTC allocations for California — "
        "with calibrated prediction intervals, cost proxies, and market comparables."
    ),
    kicker: str = "California · LIHTC Intelligence",
    # Kept for back-compat with existing callers; ignored in modern minimal layout.
    edition: str | None = None,
    bureau: str | None = None,
) -> None:
    """Modern minimal page header — kicker rule, title, dek."""
    html = f"""
    <div class="masthead">
      <div class="kicker">{kicker}</div>
      <h1 class="title">{title_main} <em>{title_em}</em></h1>
      <div class="dek">{dek}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Plotly template
# ---------------------------------------------------------------------------
def _build_plotly_template() -> go.layout.Template:
    return go.layout.Template(
        layout=go.Layout(
            font=dict(
                family="'Public Sans', sans-serif",
                color=PALETTE["ink"],
                size=12,
            ),
            title=dict(
                font=dict(
                    family="'Newsreader', serif",
                    size=18,
                    color=PALETTE["ink"],
                ),
                x=0.0,
                xanchor="left",
                pad=dict(t=8, b=12),
            ),
            paper_bgcolor=PALETTE["paper"],
            plot_bgcolor=PALETTE["paper"],
            colorway=[
                PALETTE["deep"],
                PALETTE["tide"],
                PALETTE["sky"],
                PALETTE["signal"],
                PALETTE["ink_soft"],
                PALETTE["mist"],
            ],
            xaxis=dict(
                showgrid=False,
                showline=True,
                linecolor=PALETTE["rule"],
                linewidth=1,
                ticks="outside",
                tickcolor=PALETTE["rule"],
                tickfont=dict(family="'JetBrains Mono', monospace", size=11),
                title=dict(
                    font=dict(family="'Public Sans', sans-serif", size=11),
                    standoff=12,
                ),
                automargin=True,
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(31, 78, 121, 0.08)",
                gridwidth=1,
                showline=False,
                ticks="outside",
                tickcolor=PALETTE["rule"],
                tickfont=dict(family="'JetBrains Mono', monospace", size=11),
                title=dict(
                    font=dict(family="'Public Sans', sans-serif", size=11),
                    standoff=12,
                ),
                automargin=True,
            ),
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                bordercolor=PALETTE["rule"],
                borderwidth=0,
                font=dict(size=11),
            ),
            margin=dict(l=20, r=20, t=60, b=70, autoexpand=True),
            hoverlabel=dict(
                bgcolor=PALETTE["deep"],
                bordercolor=PALETTE["deep"],
                font=dict(family="'JetBrains Mono', monospace", color=PALETTE["paper"], size=11),
            ),
        )
    )


PLOTLY_TEMPLATE = _build_plotly_template()
pio.templates["hydrography"] = PLOTLY_TEMPLATE
pio.templates.default = "hydrography"


# ---------------------------------------------------------------------------
# Semantic chart colors
# ---------------------------------------------------------------------------
# Bar chart [lower / point / upper] uses a blue gradient (sky → tide → deep)
# so it reads as a sequence, not a comparison.
# SHAP +/- uses (deep, signal-orange) — orange here is contrast, not alarm.
# Map "Your Project" uses signal-orange — it is THE point of focus.
CHART_COLORS = {
    "primary":   PALETTE["deep"],
    "secondary": PALETTE["tide"],
    "tertiary":  PALETTE["sky"],

    # Prediction-range bar (sequence)
    "lower":     PALETTE["sky"],
    "point":     PALETTE["tide"],
    "upper":     PALETTE["deep"],

    # SHAP — true binary contrast, orange permitted
    "positive":  PALETTE["deep"],
    "negative":  PALETTE["signal"],

    # Map
    "user":      PALETTE["signal"],
    "comparable":PALETTE["tide"],
}


# Categorical palette for housing types — 6 muted, distinct hues.
# Spread across the color wheel (blue / plum / teal / bronze / burgundy /
# olive) so adjacent series never read as variants of the same color.
# All values are dark enough to read on white paper without being garish.
HOUSING_TYPE_COLORS = {
    "Large Family":  "#1F4E79",   # deep navy blue
    "Senior":        "#6B4978",   # muted plum
    "Special Needs": "#3F7A78",   # slate teal
    "SRO":           "#8A6A2E",   # bronze
    "Non-Targeted":  "#7A3947",   # burgundy
    "At-Risk":       "#5C6B2E",   # deep olive
}
