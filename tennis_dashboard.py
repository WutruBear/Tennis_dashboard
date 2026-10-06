import base64
import io
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ── paths ──────────────────────────────────────────────────────────────────────
# Resolve data files relative to this script, not the process's working
# directory, so the app behaves the same no matter where it's launched from.
APP_DIR = Path(__file__).resolve().parent

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Tennis Dashboard", page_icon="🎾", layout="wide")

# ── initialize session state ───────────────────────────────────────────────────────
if "selected_player" not in st.session_state:
    st.session_state.selected_player = "Wouter"

# ── global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }

.dash-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 26px; font-weight: 600; letter-spacing: -0.5px;
    margin: 0 0 2px; color: var(--text-color, #fff);
}
.dash-subtitle {
    font-size: 13px; opacity: .45; letter-spacing: .05em;
    text-transform: uppercase; margin-bottom: 1.6rem;
}

.kpi-row { display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; margin-bottom: 1.8rem; }
.kpi-card {
    background: rgba(255,255,255,.04);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 12px; padding: 18px 20px;
    position: relative; overflow: hidden;
}
.kpi-card::before {
    content: ''; position: absolute; inset: 0;
    background: linear-gradient(135deg, rgba(255,255,255,.03) 0%, transparent 60%);
    pointer-events: none;
}
.kpi-label { font-size: 11px; letter-spacing: .08em; text-transform: uppercase; opacity: .45; margin-bottom: 8px; }
.kpi-value { font-family: 'DM Mono', monospace; font-size: 28px; font-weight: 500; line-height: 1; letter-spacing: -1px; margin-bottom: 6px; }
.kpi-delta { font-size: 12px; opacity: .55; }
.kpi-card.accent-green .kpi-value { color: #4ECDA4; }
.kpi-card.accent-blue  .kpi-value { color: #60A5FA; }
.kpi-card.accent-amber .kpi-value { color: #FBBF24; }

.section-header {
    font-size: 13px; font-weight: 600; letter-spacing: .07em;
    text-transform: uppercase; opacity: .45; margin: 0 0 14px;
    display: flex; align-items: center; gap: 8px;
}
.section-header::after { content: ''; flex: 1; height: 1px; background: rgba(255,255,255,.07); }

.pt-row {
    display: grid;
    grid-template-columns: 1fr 60px 140px 80px 1fr;
    gap: 8px; align-items: center; padding: 8px 0;
    border-bottom: 1px solid rgba(255,255,255,.06); flex-wrap: nowrap;
}
.pt-row:last-child { border-bottom: none; }
.pt-header { opacity: .35; font-size: 10.5px; letter-spacing: .08em; text-transform: uppercase; }
.pt-name { font-size: 14px; font-weight: 500; }
.pt-matches { font-family: 'DM Mono', monospace; font-size: 13px; opacity: .7; }
.pill { display: inline-block; font-size: 11.5px; font-weight: 500; padding: 3px 9px; border-radius: 20px; }
.pill-w { background: rgba(78,205,164,.15); color: #4ECDA4; }
.pill-l { background: rgba(248,113,113,.15); color: #F87171; }
.bar-wrap { display: flex; align-items: center; gap: 8px; }
.bar-bg { flex:1; height: 5px; background: rgba(255,255,255,.1); border-radius: 3px; overflow: hidden; min-width: 50px; }
.bar-fill { height: 100%; border-radius: 3px; }
.wr-text { font-family: 'DM Mono', monospace; font-size: 13px; min-width: 36px; }

.form-pill {
    display: inline-block; width: 28px; height: 28px; border-radius: 50%;
    text-align: center; line-height: 28px; font-size: 11px; font-weight: 700;
}
.form-w { background: rgba(78,205,164,.18); color: #4ECDA4; }
.form-l { background: rgba(248,113,113,.18); color: #F87171; }
.form-t { background: rgba(251,191,36,.18); color: #FBBF24; }

.stDataFrame { border-radius: 10px; overflow: hidden; }
div[data-testid="stDataFrame"] table { font-family: 'DM Mono', monospace; font-size: 12.5px; }

[data-testid="stSidebar"] { background: rgba(10,10,15,.95); border-right: 1px solid rgba(255,255,255,.06); }
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stNumberInput input,
[data-testid="stSidebar"] .stSelectbox select { font-family: 'DM Mono', monospace; font-size: 13px; background: rgba(255,255,255,.06) !important; border-color: rgba(255,255,255,.1) !important; }
[data-testid="stSidebar"] label { font-size: 12px !important; opacity: .7; letter-spacing: .03em; }

.streamlit-expanderHeader { font-size: 13px; font-weight: 500; opacity: .7; }
hr { border-color: rgba(255,255,255,.07) !important; margin: 1.6rem 0 !important; }

/* Mobile responsiveness */
@media (max-width: 768px) {
    .kpi-row {
        grid-template-columns: repeat(2, 1fr);
    }
    .pt-row {
        grid-template-columns: 1fr 50px 120px 70px 1fr;
        padding: 6px 0;
    }
}

@media (max-width: 480px) {
    .kpi-row {
        grid-template-columns: 1fr;
    }
    .dash-title {
        font-size: 20px;
    }
    .pt-row {
        grid-template-columns: 1fr 45px;
        gap: 4px;
    }
}
</style>
""", unsafe_allow_html=True)

# ── player selector ────────────────────────────────────────────────────────────────
st.markdown('<h1 class="dash-title">🎾 Tennis Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="dash-subtitle">Doubles rating progression & match analytics</p>', unsafe_allow_html=True)
col_select, col_spacer = st.columns([2, 5])
with col_select:
    selected = st.selectbox(
        "Select player",
        ["Wouter", "Gerard", "Joris"],
        index=0,
        key="player_selector"
    )
    st.session_state.selected_player = selected

st.divider()

# ── constants ──────────────────────────────────────────────────────────────────
DATA_FILE = f"tennis_data_{st.session_state.selected_player}.csv"  # download filename only

# Elo-like rating model configuration
KNLTB_CONFIG = {
    "Q": 2.012,      # Scaling factor
    "K": 0.275,      # Sensitivity
}
_Q = KNLTB_CONFIG["Q"]
_K = KNLTB_CONFIG["K"]

# ── chart theme ────────────────────────────────────────────────────────────────
CHART_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="rgba(255,255,255,0.55)", size=11),
    hoverlabel=dict(bgcolor="#1a1a2e", font_size=12, font_family="DM Sans"),
)
_MARGIN_DEFAULT = dict(l=0, r=0, t=12, b=0)  # Standardized across all charts
_XAXIS_DEFAULTS = dict(showgrid=False, zeroline=False, showline=False, tickcolor="rgba(255,255,255,.2)")
_YAXIS_DEFAULTS = dict(gridcolor="rgba(255,255,255,.05)", zeroline=False, showline=False, tickcolor="rgba(255,255,255,.2)")
C_WIN  = "#4ECDA4"
C_LOSS = "#F87171"
C_LINE = "#60A5FA"
C_PROB = "#A78BFA"
C_AMBER = "#FBBF24"  # Accent color for amber elements


# ── helpers ────────────────────────────────────────────────────────────────────
def calc_diff(start_rating, partner_rating, opp1_rating, opp2_rating, result):
    R1 = (start_rating + partner_rating) / 2
    R2 = (opp1_rating + opp2_rating) / 2
    prob = 1 - (1 / (1 + math.e ** (-_Q * (R1 - R2))))
    res_val = {"win": 1.0, "loss": 0.0, "tie": 0.5}[result]
    return round(_K * (prob - res_val), 4)


def calc_win_prob(start_rating, partner_rating, opp1_rating, opp2_rating):
    """Return model win probability (0–1) for a given match."""
    R1 = (start_rating + partner_rating) / 2
    R2 = (opp1_rating + opp2_rating) / 2
    return 1 / (1 + math.e ** (_Q * (R1 - R2)))


def get_ellipse_points(x, y, std_mult=2, num_points=100):
    """
    Calculate ellipse points from 2D data.
    Uses covariance matrix to determine ellipse dimensions.
    std_mult: number of standard deviations (default 2 ≈ 95% confidence)
    """
    from numpy.linalg import eigh

    if len(x) < 3:
        return None, None

    # Center of ellipse
    center_x, center_y = np.mean(x), np.mean(y)

    # Covariance matrix (symmetric, so eigh is both faster and numerically
    # stable — unlike eig it's guaranteed to return real eigenvalues/vectors
    # and never fails with a ComplexWarning on noisy/near-collinear data)
    cov = np.cov(x, y)

    # Eigenvalues and eigenvectors, sorted ascending by eigh
    eigenvalues, eigenvectors = eigh(cov)
    eigenvalues = np.clip(eigenvalues, a_min=0, a_max=None)  # guard tiny negative noise

    # Take the larger eigenvalue/vector as the semi-major axis
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Semi-axes lengths (std dev * multiplier)
    a = std_mult * np.sqrt(eigenvalues[0])
    b = std_mult * np.sqrt(eigenvalues[1])

    # Angle of rotation
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    
    # Parametric ellipse equation
    t = np.linspace(0, 2 * np.pi, num_points)
    ellipse_x = center_x + a * np.cos(t) * np.cos(angle) - b * np.sin(t) * np.sin(angle)
    ellipse_y = center_y + a * np.cos(t) * np.sin(angle) + b * np.sin(t) * np.cos(angle)
    
    return ellipse_x, ellipse_y


def _data_path(player):
    return APP_DIR / f"tennis_data_{player}.csv"


# ── storage ────────────────────────────────────────────────────────────────────
# Two modes:
#   * Local (default): reads/writes the CSVs next to this script.
#   * GitHub: when Streamlit secrets contain a [github] section, the CSVs live in
#     a separate (private) GitHub repo and every save is committed there. This is
#     what makes data entered through the app survive restarts on Streamlit
#     Cloud, whose own disk is temporary.
def _github_cfg():
    try:
        cfg = st.secrets["github"]
        return {
            "token": cfg["token"],
            "repo": cfg["repo"],                      # e.g. "yourname/tennis-data"
            "branch": cfg.get("branch", "main"),
            "folder": cfg.get("folder", "").strip("/"),
        }
    except Exception:
        return None


def _gh_url(cfg, player):
    name = f"tennis_data_{player}.csv"
    path = f"{cfg['folder']}/{name}" if cfg["folder"] else name
    return f"https://api.github.com/repos/{cfg['repo']}/contents/{path}"


def _gh_headers(cfg):
    return {
        "Authorization": f"Bearer {cfg['token']}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _gh_get(cfg, player):
    """Return (csv_text, sha) or (None, None) if the file doesn't exist yet."""
    r = requests.get(_gh_url(cfg, player), headers=_gh_headers(cfg),
                     params={"ref": cfg["branch"]}, timeout=20)
    if r.status_code == 404:
        return None, None
    r.raise_for_status()
    j = r.json()
    if j.get("content"):
        text = base64.b64decode(j["content"]).decode("utf-8")
    else:  # files > 1 MB come back without inline content
        raw = requests.get(j["download_url"], timeout=20)
        raw.raise_for_status()
        text = raw.text
    return text, j["sha"]


def _empty_frame():
    # Bug fix: Date must be a real datetime dtype even for a brand-new
    # player with no matches yet — every chart calls .dt.year/.dt.strftime
    # on this column, which raises AttributeError on a plain object dtype
    # (this used to crash the whole page for any player without a CSV).
    df = pd.DataFrame(columns=[
        "Match", "Date", "Start_rating", "Diff", "New_rating",
        "Partner", "Partner_rating", "Opp1_rating", "Opp2_rating",
        "Result", "Set1", "Set2", "Set3", "Surface", "Type", "Phase",
    ])
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data(ttl=300)
def load_data(player):
    cfg = _github_cfg()
    if cfg:
        text, _ = _gh_get(cfg, player)
        if text is None:
            return _empty_frame()
        df = pd.read_csv(io.StringIO(text))
    else:
        data_file = _data_path(player)
        if not data_file.exists():
            return _empty_frame()
        df = pd.read_csv(data_file)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    return df


def save_data(df, player):
    out = df.copy()
    out["Date"] = out["Date"].dt.strftime("%d/%m/%Y")
    cfg = _github_cfg()
    if not cfg:
        out.to_csv(_data_path(player), index=False)
        return

    csv_text = out.to_csv(index=False)
    _, sha = _gh_get(cfg, player)          # fresh sha right before writing
    body = {
        "message": f"Update {player} ({datetime.now():%Y-%m-%d %H:%M})",
        "content": base64.b64encode(csv_text.encode("utf-8")).decode("ascii"),
        "branch": cfg["branch"],
    }
    if sha:
        body["sha"] = sha
    r = requests.put(_gh_url(cfg, player), headers=_gh_headers(cfg),
                     json=body, timeout=30)
    if not r.ok:
        raise RuntimeError(
            f"Could not save to GitHub ({r.status_code}): {r.json().get('message', r.text)}"
        )


def result_label(result):
    """Format the Result column value with a check/cross mark."""
    if result is None or (isinstance(result, float) and math.isnan(result)):
        return "—"
    result_str = str(result).strip().lower()
    if result_str == "win":
        return "Win ✓"
    elif result_str == "loss":
        return "Loss ✗"
    elif result_str == "tie":
        return "Tie"
    return "—"


def _wl(df):
    out = df[df["Result"].notna()].copy()
    out["Result_clean"] = out["Result"].str.lower().str.strip()
    out = out[out["Result_clean"].isin(["win", "loss"])].copy()
    out["Avg_Team_Rating"] = (out["Start_rating"] + out["Partner_rating"]) / 2
    out["Avg_Opp_Rating"]  = (out["Opp1_rating"]  + out["Opp2_rating"])  / 2
    return out


def add_year_lines(fig, plot_df):
    """
    Add vertical lines at year boundaries to a Plotly figure.
    
    Args:
        fig: Plotly Figure object
        plot_df: DataFrame with Date and Match_Index columns
    """
    if len(plot_df) == 0:
        return
    
    # Get year boundaries
    plot_df_sorted = plot_df.sort_values("Date")
    years = plot_df_sorted["Date"].dt.year.unique()
    
    for year in sorted(years)[:-1]:  # Exclude the last year to avoid drawing at the edge
        # Find matches in this year and the next
        current_year_matches = plot_df_sorted[plot_df_sorted["Date"].dt.year == year]
        next_year_matches = plot_df_sorted[plot_df_sorted["Date"].dt.year == year + 1]
        
        if len(current_year_matches) > 0 and len(next_year_matches) > 0:
            # Position the line between the last match of current year and first of next year
            last_idx = current_year_matches["Match_Index"].max()
            first_idx_next = next_year_matches["Match_Index"].min()
            line_x = (last_idx + first_idx_next) / 2
            
            fig.add_vline(
                x=line_x,
                line_width=1.5,
                line_dash="dot",
                line_color="rgba(255,255,255,.2)",
                annotation_text=str(year + 1),
                annotation_position="top right",
                annotation_font_size=9,
                annotation_font_color="rgba(255,255,255,.3)",
                layer="below"
            )


# ── load ───────────────────────────────────────────────────────────────────────
df = load_data(st.session_state.selected_player)

# ── sidebar: add match ─────────────────────────────────────────────────────────
st.sidebar.markdown("### ➕ Add match")
last_rating = float(df["New_rating"].dropna().iloc[-1]) if df["New_rating"].notna().any() else 8.0

# Use session state to track form inputs
if "new_date" not in st.session_state:
    st.session_state.new_date = datetime.today()
if "new_start" not in st.session_state:
    # After a successful add, we stash the resulting rating here so the next
    # entry starts pre-filled with it (see submit handler below) instead of
    # writing to the `new_start` widget key directly, which Streamlit forbids
    # once the widget has been instantiated for the run.
    st.session_state.new_start = st.session_state.pop("_carry_start_rating", last_rating)
if "new_partner" not in st.session_state:
    st.session_state.new_partner = ""
if "new_part_r" not in st.session_state:
    st.session_state.new_part_r = 8.0
if "new_opp1" not in st.session_state:
    st.session_state.new_opp1 = 8.0
if "new_opp2" not in st.session_state:
    st.session_state.new_opp2 = 8.0
if "new_result" not in st.session_state:
    st.session_state.new_result = "win"
if "new_set1" not in st.session_state:
    st.session_state.new_set1 = ""
if "new_set2" not in st.session_state:
    st.session_state.new_set2 = ""
if "new_set3" not in st.session_state:
    st.session_state.new_set3 = ""
if "new_surface" not in st.session_state:
    st.session_state.new_surface = "Kunstgras"
if "new_type" not in st.session_state:
    st.session_state.new_type = "Competition"
if "new_phase" not in st.session_state:
    st.session_state.new_phase = ""

# Input fields (outside form for live updates)
new_date    = st.sidebar.date_input("Date", key="new_date")
new_start   = st.sidebar.number_input("Your start rating", step=0.0001, format="%.4f", key="new_start")
new_partner = st.sidebar.text_input("Partner name", key="new_partner")
new_part_r  = st.sidebar.number_input("Partner rating", step=0.0001, format="%.4f", key="new_part_r")
st.sidebar.markdown("<div style='margin-top:4px'></div>", unsafe_allow_html=True)
st.sidebar.markdown("<span style='font-size:11px;opacity:.5;text-transform:uppercase;letter-spacing:.06em'>Opponents</span>", unsafe_allow_html=True)
new_opp1    = st.sidebar.number_input("Opponent 1 rating", step=0.0001, format="%.4f", key="new_opp1")
new_opp2    = st.sidebar.number_input("Opponent 2 rating", step=0.0001, format="%.4f", key="new_opp2")
st.sidebar.markdown("<span style='font-size:11px;opacity:.5;text-transform:uppercase;letter-spacing:.06em'>Score</span>", unsafe_allow_html=True)
new_result  = st.sidebar.selectbox("Result", ["win", "loss", "tie"], key="new_result")
new_set1    = st.sidebar.text_input("Set 1  (e.g. 6-3)", key="new_set1")
new_set2    = st.sidebar.text_input("Set 2", key="new_set2")
new_set3    = st.sidebar.text_input("Set 3 (opt.)", key="new_set3")
st.sidebar.markdown("<span style='font-size:11px;opacity:.5;text-transform:uppercase;letter-spacing:.06em'>Context</span>", unsafe_allow_html=True)
new_surface = st.sidebar.selectbox("Surface", ["Kunstgras", "Gravel", "Tapijt", "Indoor", "Other"], key="new_surface")
new_type    = st.sidebar.selectbox("Type", ["Competition", "Tournament"], key="new_type")
new_phase   = st.sidebar.text_input("Phase (e.g. Group, Final)", key="new_phase")

# Calculate and display expected diff (live)
expected_diff = calc_diff(new_start, new_part_r, new_opp1, new_opp2, new_result)
expected_rating = round(new_start + expected_diff, 4)
st.sidebar.markdown(f"<div style='background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.1);border-radius:8px;padding:12px;margin:12px 0;'><span style='font-size:11px;opacity:.6;text-transform:uppercase;letter-spacing:.06em'>Expected diff</span><div style='font-family:DM Mono;font-size:18px;font-weight:500;color:#60A5FA;margin-top:4px'>{expected_diff:+.4f}</div><span style='font-size:11px;opacity:.5;margin-top:6px;display:block'>New rating: {expected_rating:.4f}</span></div>", unsafe_allow_html=True)

# Submit button
if st.sidebar.button("Add match", key="submit_match", width='stretch', type="primary"):
    # ── basic validation before writing anything ──────────────────────────────
    errors = []
    if not new_partner.strip():
        errors.append("Partner name is required.")
    for label, val in [("Your start rating", new_start), ("Partner rating", new_part_r),
                        ("Opponent 1 rating", new_opp1), ("Opponent 2 rating", new_opp2)]:
        if val is None or val <= 0:
            errors.append(f"{label} must be a positive number.")
    if new_date > datetime.today().date():
        errors.append("Date can't be in the future.")

    if errors:
        for e in errors:
            st.sidebar.error(e)
    else:
        diff       = calc_diff(new_start, new_part_r, new_opp1, new_opp2, new_result)
        new_rating = round(new_start + diff, 4)
        next_match = int(df["Match"].max()) + 1 if len(df) else 1
        new_row = pd.DataFrame([{
            "Match": next_match, "Date": pd.to_datetime(new_date),
            "Start_rating": new_start, "Diff": diff, "New_rating": new_rating,
            "Partner": new_partner.strip(), "Partner_rating": new_part_r,
            "Opp1_rating": new_opp1, "Opp2_rating": new_opp2,
            "Result": new_result.capitalize(),
            "Set1": new_set1 or "-", "Set2": new_set2 or "-", "Set3": new_set3 or "-",
            "Surface": new_surface, "Type": new_type, "Phase": new_phase or None,
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        save_data(df, st.session_state.selected_player)
        load_data.clear()

        # Reset the per-match fields but carry the new rating forward, so the
        # next entry starts pre-filled and today's still-fresh values (e.g.
        # partner/sets) don't leak into the next match by accident.
        for key in ("new_partner", "new_part_r", "new_opp1", "new_opp2",
                    "new_set1", "new_set2", "new_set3", "new_phase", "new_start"):
            st.session_state.pop(key, None)
        st.session_state["_carry_start_rating"] = new_rating

        st.sidebar.success(f"Diff {diff:+.4f} → {new_rating:.4f}")
        st.rerun()

# ── delete match ───────────────────────────────────────────────────────────────
st.sidebar.markdown("### ✕ Delete match")
if len(df) > 0:
    with st.sidebar.form("delete_match"):
        delete_match_num = st.selectbox(
            "Select match",
            options=df["Match"].tolist(),
            format_func=lambda m: f"Match {m} ({df[df['Match']==m]['Date'].iloc[0].strftime('%d %b %Y')})"
        )
        delete_submit = st.form_submit_button("✓ Delete")
        if delete_submit:
            df = df[df["Match"] != delete_match_num].reset_index(drop=True)
            df["Match"] = range(1, len(df) + 1)
            save_data(df, st.session_state.selected_player)
            load_data.clear()  # bug fix: without this the cached (pre-delete) data kept reappearing
            st.success(f"✓ Match {delete_match_num} deleted!")
            st.rerun()

# ── main content ───────────────────────────────────────────────────────────────
real = df[df["Diff"].notna()].copy()

# ── last 5 matches form ────────────────────────────────────────────────────────
def _wlt(result):
    """Map a Result cell to 'W' / 'L' / 'T' using the actual recorded result
    (rather than the sign of Diff, which mislabels ties and any match where
    the rating happened not to move)."""
    r = str(result).strip().lower()
    return {"win": "W", "loss": "L", "tie": "T"}.get(r, "?")

recent_results = real.tail(5)["Result"].apply(_wlt).tolist()
_form_class = {"W": "w", "L": "l", "T": "t"}
form_html = "".join(
    f'<span class="form-pill form-{_form_class.get(r, "l")}">{r}</span>'
    for r in recent_results
)

# ── KPIs ───────────────────────────────────────────────────────────────────────
if len(real) > 0:
    matches_count = len(real)
    win_count = (real["Result"].str.lower().str.strip() == "win").sum()
    loss_count = (real["Result"].str.lower().str.strip() == "loss").sum()
    tie_count = (real["Result"].str.lower().str.strip() == "tie").sum()
    win_pct = round(win_count / matches_count * 100, 1) if matches_count > 0 else 0
    
    current = real["New_rating"].iloc[-1]
    best = real["New_rating"].min()  # Lower is better in tennis ratings
    worst = real["New_rating"].max()  # Higher is worse
    diff_from_best = round(current - best, 4)  # How far we are from our best rating
    
    # Calculate rating delta from last match
    rating_delta = ""
    if len(real) >= 2:
        prev = real["New_rating"].iloc[-2]
        delta = current - prev
        sign = "▼" if delta < 0 else "▲"
        rating_delta = f'<div class="kpi-delta">{sign} {abs(delta):.4f} last match</div>'

    # Current streak: how many matches in a row (most recent first) share the
    # same result. Ties break the streak (neither a win nor a loss run).
    streak_results = real["Result"].str.lower().str.strip().tolist()
    streak_len, streak_kind = 0, None
    for r in reversed(streak_results):
        if r not in ("win", "loss"):
            break
        if streak_kind is None:
            streak_kind = r
        if r != streak_kind:
            break
        streak_len += 1
    if streak_kind == "win":
        streak_text, streak_color = f"{streak_len}W", C_WIN
    elif streak_kind == "loss":
        streak_text, streak_color = f"{streak_len}L", C_LOSS
    else:
        streak_text, streak_color = "—", "inherit"

    st.markdown(f"""
<div class="kpi-row">
  <div class="kpi-card">
    <div class="kpi-label">Matches played</div>
    <div class="kpi-value">{matches_count}</div>
    <div class="kpi-delta">Form: {form_html}</div>
  </div>
  <div class="kpi-card accent-green">
    <div class="kpi-label">Win rate</div>
    <div class="kpi-value">{win_pct:.0f}<span style="font-size:16px;opacity:.6">%</span></div>
    <div class="kpi-delta">{win_count}W &nbsp;·&nbsp; {loss_count}L{f" &nbsp;·&nbsp; {tie_count}T" if tie_count > 0 else ""}</div>
  </div>
  <div class="kpi-card accent-blue">
    <div class="kpi-label">Current rating</div>
    <div class="kpi-value" style="font-size:24px">{current:.4f}</div>
    {rating_delta}
  </div>
  <div class="kpi-card accent-amber">
    <div class="kpi-label">Best rating</div>
    <div class="kpi-value" style="font-size:24px">{best:.4f}</div>
    <div class="kpi-delta">+{diff_from_best:.4f} from current</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Current streak</div>
    <div class="kpi-value" style="font-size:24px;color:{streak_color}">{streak_text}</div>
    <div class="kpi-delta">consecutive same result</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── rating progression with YEAR LINES ─────────────────────────────────────────
st.markdown('<div class="section-header">Rating progression</div>', unsafe_allow_html=True)

plot_df = (
    df[df["New_rating"].notna()]
    .sort_values("Match")
    .reset_index(drop=True)
    .assign(Match_Index=lambda d: d.index)
)
def _result_label(result):
    r = str(result).strip().lower()
    if r == "win":
        return "Win"
    if r == "loss":
        return "Loss"
    if r == "tie":
        return "Tie"
    return "No result"

plot_df["Result_label"] = plot_df["Result"].apply(_result_label)

# Calculate unified x-axis ticks (used by both rating progression and rolling win rate)
# This ensures both charts have aligned x-axis labels
n = len(plot_df)
step = max(1, n // 8)
tick_idx = list(range(0, n, step))
if n > 0 and (n - 1) not in tick_idx:
    tick_idx.append(n - 1)

fig_line = go.Figure()
fig_line.add_trace(go.Scatter(
    x=plot_df["Match_Index"], y=plot_df["New_rating"],
    mode="lines", name="Rating",
    line=dict(color=C_LINE, width=2),
    fill="tozeroy",
    fillcolor="rgba(96,165,250,0.05)",
    hoverinfo="skip",
))
for label, color in [("Win", C_WIN), ("Loss", C_LOSS), ("Tie", C_AMBER)]:
    sub = plot_df[plot_df["Result_label"] == label]
    if len(sub) == 0:
        continue  # e.g. a player with zero recorded losses/ties — nothing to plot
    fig_line.add_trace(go.Scatter(
        x=sub["Match_Index"], y=sub["New_rating"],
        mode="markers", name=label,
        marker=dict(color=color, size=7, symbol="circle",
                    line=dict(width=1.5, color="rgba(0,0,0,.4)")),
        customdata=sub[["Partner", "Set1", "Set2", "Set3", "Date"]].values,
        hovertemplate=(
            "<b>%{customdata[4]}</b><br>"
            "Rating: <b>%{y:.4f}</b><br>"
            "Partner: %{customdata[0]}<br>"
            "%{customdata[1]} %{customdata[2]} %{customdata[3]}"
            "<extra></extra>"
        ),
    ))

fig_line.update_layout(
    **CHART_DEFAULTS,
    height=280,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                bgcolor="rgba(0,0,0,0)", borderwidth=0),
    margin=dict(l=0, r=0, t=40, b=0),
    hovermode="closest",
)
# Bug fix: this used to derive the range from `sub`, a leftover loop variable
# holding only the *last* result-label subset (e.g. just Ties) — that gave a
# wrong/too-narrow range, and crashed outright for any player with zero
# losses (min()/max() on an empty series). Use the full plotted series instead.
if len(plot_df):
    y_min = plot_df["New_rating"].min() - 0.2
    y_max = plot_df["New_rating"].max() + 0.2
else:
    y_min, y_max = 0, 1
fig_line.update_yaxes(**_YAXIS_DEFAULTS, range=[y_min, y_max], fixedrange=True)
fig_line.update_xaxes(
    **_XAXIS_DEFAULTS,
    tickmode="array",
    tickvals=plot_df.loc[tick_idx, "Match_Index"].tolist() if n else [],
    ticktext=plot_df.loc[tick_idx, "Date"].dt.strftime("%d %b '%y").tolist() if n else [],
)

# ADD YEAR LINES to the rating progression chart
add_year_lines(fig_line, plot_df)

st.plotly_chart(fig_line, width='stretch')

# ── rolling win rate ───────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Rolling win rate · 5-match window</div>', unsafe_allow_html=True)

ROLL = 5

if len(real) >= 2:
    roll_df = real.sort_values("Match").copy().reset_index(drop=True)
    roll_df["Is_Win"]      = (roll_df["Result"].str.lower().str.strip() == "win").astype(float)
    roll_df["Rolling_WR"]  = roll_df["Is_Win"].rolling(ROLL, min_periods=1).mean() * 100
    roll_df["Match_Index"] = roll_df.index

    # Use the same tick indices as rating progression for alignment
    # But filter to only indices that exist in roll_df (which may be smaller)
    rtick_idx = [idx for idx in tick_idx if idx < len(roll_df)]

    fig_roll = go.Figure()

    # Shaded above/below 50%
    fig_roll.add_hrect(y0=50, y1=102, fillcolor="rgba(78,205,164,0.04)", line_width=0, layer="below")
    fig_roll.add_hrect(y0=-2, y1=50,  fillcolor="rgba(248,113,113,0.04)", line_width=0, layer="below")
    fig_roll.add_hline(y=50, line_width=1, line_dash="dash", line_color="rgba(255,255,255,.18)")

    # Rolling line
    fig_roll.add_trace(go.Scatter(
        x=roll_df["Match_Index"],
        y=roll_df["Rolling_WR"],
        mode="lines+markers",
        name=f"{ROLL}-match rolling WR",
        line=dict(color=C_PROB, width=2.5, shape="spline", smoothing=0.6),
        marker=dict(
            color=roll_df["Rolling_WR"].apply(lambda v: C_WIN if v >= 50 else C_LOSS),
            size=7,
            line=dict(width=1.5, color="rgba(0,0,0,.4)"),
        ),
        customdata=roll_df[["Partner", "Rolling_WR"]].assign(
            Date=roll_df["Date"].dt.strftime("%d %b %Y")
        )[["Date", "Partner", "Rolling_WR"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Partner: %{customdata[1]}<br>"
            "Rolling WR: <b>%{y:.0f}%</b>"
            "<extra></extra>"
        ),
    ))

    # Individual match result dots along the bottom strip
    for label, color in [("Win", C_WIN), ("Loss", C_LOSS)]:
        sub = roll_df[roll_df["Is_Win"] == (1.0 if label == "Win" else 0.0)]
        fig_roll.add_trace(go.Scatter(
            x=sub["Match_Index"], y=[4] * len(sub),
            mode="markers", name=label,
            marker=dict(color=color, size=6, symbol="circle",
                        line=dict(width=1, color="rgba(0,0,0,.3)")),
            hoverinfo="skip",
        ))

    fig_roll.update_layout(
        **CHART_DEFAULTS,
        height=220,
        margin=_MARGIN_DEFAULT,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", borderwidth=0),
        yaxis=dict(**_YAXIS_DEFAULTS, range=[-2, 105],
                   tickvals=[0, 25, 50, 75, 100],
                   ticktext=["0%", "25%", "50%", "75%", "100%"]),
        xaxis=dict(
            **_XAXIS_DEFAULTS,
            tickmode="array",
            tickvals=roll_df.loc[rtick_idx, "Match_Index"].tolist() if len(rtick_idx) else [],
            ticktext=roll_df.iloc[rtick_idx]["Date"].dt.strftime("%d %b '%y").tolist() if len(rtick_idx) else [],
        ),
    )
    
    # ADD YEAR LINES to the rolling win rate chart
    add_year_lines(fig_roll, roll_df)
    
    st.plotly_chart(fig_roll, width='stretch')
else:
    st.caption("Need at least 2 matches to show rolling win rate.")

# ── row 2: win-by-year + set distribution ─────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.markdown('<div class="section-header">Wins &amp; losses by year</div>', unsafe_allow_html=True)
    _res_clean = real["Result"].str.lower().str.strip()
    yr = (
        real.assign(Year=real["Date"].dt.year, Is_Win=(_res_clean == "win"), Is_Loss=(_res_clean == "loss"))
        .groupby("Year")
        .agg(Wins=("Is_Win", "sum"), Losses=("Is_Loss", "sum"))
        .reset_index()
    )
    fig_yr = go.Figure()
    fig_yr.add_bar(x=yr["Year"].astype(str), y=yr["Wins"],   name="Wins",
                   marker_color=C_WIN, marker_cornerradius=4,
                   hovertemplate="%{y} wins<extra></extra>")
    fig_yr.add_bar(x=yr["Year"].astype(str), y=yr["Losses"], name="Losses",
                   marker_color=C_LOSS, marker_cornerradius=4,
                   hovertemplate="%{y} losses<extra></extra>")
    fig_yr.update_layout(
        **CHART_DEFAULTS,
        margin=_MARGIN_DEFAULT,
        barmode="group", height=240,
        bargap=0.25, bargroupgap=0.08,
        legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)", borderwidth=0),
        yaxis_title="Matches",
    )
    fig_yr.update_yaxes(**_YAXIS_DEFAULTS)
    st.plotly_chart(fig_yr, width='stretch')

with col2:
    st.markdown('<div class="section-header">Set score distribution</div>', unsafe_allow_html=True)

    def parse_scoreline(row):
        sets = [row["Set1"], row["Set2"], row["Set3"]]
        won, lost = 0, 0
        for s in sets:
            if pd.isna(s) or str(s).strip() == "":
                continue
            try:
                a, b = map(int, str(s).strip().split("-"))
                if a > b: won += 1
                else:     lost += 1
            except ValueError:
                continue
        return f"{won}-{lost}" if (won + lost) > 0 else None

    scored = real.copy()
    scored["Scoreline"] = scored.apply(parse_scoreline, axis=1)
    dist = (
        scored["Scoreline"]
        .dropna()
        .value_counts()
        .reset_index()
        .rename(columns={"index": "Scoreline", "count": "Count", "Scoreline": "Scoreline"})
    )
    SCORE_ORDER = ["2-0", "2-1", "1-1", "1-2", "0-2"]
    dist = dist[dist["Scoreline"].isin(SCORE_ORDER)].copy()  # drop any unexpected/malformed scorelines first
    dist["Scoreline"] = pd.Categorical(dist["Scoreline"], categories=SCORE_ORDER, ordered=True)
    dist = dist.sort_values("Scoreline")

    color_map = {"2-0": C_WIN, "2-1": C_WIN, "1-2": C_LOSS, "0-2": C_LOSS, "1-1": C_AMBER}
    bar_colors = [color_map.get(s, "#94A3B8") for s in dist["Scoreline"]]

    fig_sets = go.Figure(go.Bar(
        x=dist["Scoreline"], y=dist["Count"],
        marker_color=bar_colors, marker_cornerradius=4,
        hovertemplate="%{x}: <b>%{y}</b> matches<extra></extra>",
    ))
    fig_sets.update_layout(
        **CHART_DEFAULTS, margin=_MARGIN_DEFAULT,
        height=240, showlegend=False, yaxis_title="Matches",
    )
    fig_sets.update_xaxes(**_XAXIS_DEFAULTS)
    fig_sets.update_yaxes(**_YAXIS_DEFAULTS)
    st.plotly_chart(fig_sets, width='stretch')

# ── row 3: histogram + scatter ─────────────────────────────────────────────────
col3, col4 = st.columns([1, 1], gap="medium")

with col3:
    st.markdown('<div class="section-header">Rating diff vs. opponent</div>', unsafe_allow_html=True)
    wl_df = _wl(real)
    wl_df["Rating_Diff"] = wl_df["Avg_Team_Rating"] - wl_df["Avg_Opp_Rating"]

    # symmetric range around 0
    max_abs = wl_df["Rating_Diff"].abs().max()
    pad = max_abs * 0.05  # small breathing room, optional
    x_range = [-(max_abs + pad), max_abs + pad]

    fig_hist = go.Figure()
    for label, color in [("win", C_WIN), ("loss", C_LOSS)]:
        fig_hist.add_trace(go.Histogram(
            x=wl_df[wl_df["Result_clean"] == label]["Rating_Diff"],
            name=f"{label.capitalize()}s", xbins=dict(size=0.1),
            marker_color=color, marker_cornerradius=2, opacity=0.85,
        ))
    fig_hist.add_vline(x=0, line_width=1, line_dash="dash", line_color="rgba(255,255,255,.25)")
    fig_hist.update_layout(
        **CHART_DEFAULTS, margin=_MARGIN_DEFAULT,
        barmode="stack", bargap=0.08, height=240,
        xaxis_title="Team rating − Opp rating", yaxis_title="Matches",
        legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)", borderwidth=0),
    )
    fig_hist.update_xaxes(**_XAXIS_DEFAULTS, range=x_range)
    fig_hist.update_yaxes(**_YAXIS_DEFAULTS)
    st.plotly_chart(fig_hist, width='stretch')

with col4:
    st.markdown('<div class="section-header">Team vs. opponent rating</div>', unsafe_allow_html=True)
    wl_df = _wl(real)
    
    # Partner dropdown
    partners = sorted(wl_df["Partner"].unique().tolist())
    selected_partner = st.selectbox("Highlight partner:", ["All partners"] + partners, key="partner_ellipse")
    
    fig_scatter = go.Figure()
    
    # Determine which partner to highlight for ellipse
    partner_for_ellipse = selected_partner if selected_partner != "All partners" else None
    
    for label, color in [("win", C_WIN), ("loss", C_LOSS)]:
        sub = wl_df[wl_df["Result_clean"] == label]
        
        # Determine opacity based on selection
        if partner_for_ellipse and len(sub) > 0:
            is_selected = sub["Partner"] == partner_for_ellipse
            # Add non-selected points with lower opacity
            non_selected = sub[~is_selected]
            if len(non_selected) > 0:
                fig_scatter.add_trace(go.Scatter(
                    x=non_selected["Avg_Opp_Rating"], y=non_selected["Avg_Team_Rating"],
                    mode="markers", name=f"{label.capitalize()}s (other)",
                    marker=dict(color=color, size=7, opacity=0.3,
                                line=dict(width=1, color="rgba(0,0,0,.2)")),
                    customdata=list(zip(non_selected["Date"].dt.strftime("%d %b %Y"), non_selected["Partner"])),
                    hovertemplate=(
                        f"<b>{label.capitalize()}</b><br>%{{customdata[0]}}<br>"
                        "Team: <b>%{y:.2f}</b> · Opp: <b>%{x:.2f}</b><br>"
                        "Partner: %{customdata[1]}<extra></extra>"
                    ),
                    showlegend=False,
                ))
            # Add selected points with full opacity
            selected = sub[is_selected]
            if len(selected) > 0:
                fig_scatter.add_trace(go.Scatter(
                    x=selected["Avg_Opp_Rating"], y=selected["Avg_Team_Rating"],
                    mode="markers", name=f"{label.capitalize()}s ({partner_for_ellipse})",
                    marker=dict(color=color, size=10, opacity=0.9,
                                line=dict(width=2, color="rgba(255,255,255,.6)")),
                    customdata=list(zip(selected["Date"].dt.strftime("%d %b %Y"), selected["Partner"])),
                    hovertemplate=(
                        f"<b>{label.capitalize()}</b><br>%{{customdata[0]}}<br>"
                        "Team: <b>%{y:.2f}</b> · Opp: <b>%{x:.2f}</b><br>"
                        "Partner: %{customdata[1]}<extra></extra>"
                    ),
                ))
        else:
            # No selection or partner_for_ellipse is None - show all normally
            fig_scatter.add_trace(go.Scatter(
                x=sub["Avg_Opp_Rating"], y=sub["Avg_Team_Rating"],
                mode="markers", name=f"{label.capitalize()}s",
                marker=dict(color=color, size=9, opacity=0.8,
                            line=dict(width=1.5, color="rgba(0,0,0,.3)")),
                customdata=list(zip(sub["Date"].dt.strftime("%d %b %Y"), sub["Partner"])),
                hovertemplate=(
                    f"<b>{label.capitalize()}</b><br>%{{customdata[0]}}<br>"
                    "Team: <b>%{y:.2f}</b> · Opp: <b>%{x:.2f}</b><br>"
                    "Partner: %{customdata[1]}<extra></extra>"
                ),
            ))
    
    # Add ellipse if partner is selected
    if partner_for_ellipse and len(wl_df[wl_df["Partner"] == partner_for_ellipse]) >= 3:
        partner_data = wl_df[wl_df["Partner"] == partner_for_ellipse]
        ellipse_x, ellipse_y = get_ellipse_points(
            partner_data["Avg_Opp_Rating"].values,
            partner_data["Avg_Team_Rating"].values,
            std_mult=1.5
        )
        if ellipse_x is not None:
            fig_scatter.add_trace(go.Scatter(
                x=ellipse_x, y=ellipse_y,
                mode="lines", name=f"{partner_for_ellipse} envelope",
                line=dict(color="rgba(255,255,255,.4)", width=2, dash="dash"),
                fill="toself", fillcolor="rgba(255,255,255,.02)",
                hoverinfo="skip",
            ))
    
    if len(wl_df):
        r_min = wl_df[["Avg_Team_Rating", "Avg_Opp_Rating"]].min().min() - 0.15
        r_max = wl_df[["Avg_Team_Rating", "Avg_Opp_Rating"]].max().max() + 0.15
        fig_scatter.add_trace(go.Scatter(
            x=[r_min, r_max], y=[r_min, r_max],
            mode="lines", name="Even",
            line=dict(color="rgba(255,255,255,.2)", dash="dash", width=1),
            hoverinfo="skip",
        ))
    fig_scatter.update_layout(
        **CHART_DEFAULTS, margin=_MARGIN_DEFAULT, height=240,
        xaxis_title="Avg opponent rating", yaxis_title="Avg team rating",
        legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)", borderwidth=0),
    )
    fig_scatter.update_xaxes(**_XAXIS_DEFAULTS)
    fig_scatter.update_yaxes(**_YAXIS_DEFAULTS)
    st.plotly_chart(fig_scatter, width='stretch')

# ── win probability analysis ───────────────────────────────────────────────────
st.markdown('<div class="section-header">Win probability · model vs. reality</div>', unsafe_allow_html=True)

wl_prob = _wl(real).copy()
if len(wl_prob) >= 3:
    wl_prob["Win_prob"] = wl_prob.apply(
        lambda r: calc_win_prob(r["Start_rating"], r["Partner_rating"],
                                r["Opp1_rating"], r["Opp2_rating"]) * 100,
        axis=1,
    )
    wl_prob["Is_Win"]      = (wl_prob["Result_clean"] == "win").astype(int)
    wl_prob["Match_Index"] = range(len(wl_prob))

    col5, col6 = st.columns([1, 1], gap="medium")

    # ── left: per-match probability scatter ───────────────────────────────────
    with col5:
        fig_prob = go.Figure()

        fig_prob.add_hrect(y0=50, y1=102, fillcolor="rgba(78,205,164,0.04)", line_width=0)
        fig_prob.add_hrect(y0=-2, y1=50,  fillcolor="rgba(248,113,113,0.04)", line_width=0)
        fig_prob.add_hline(y=50, line_width=1, line_dash="dash", line_color="rgba(255,255,255,.18)")

        for label, color, sym in [("win", C_WIN, "circle"), ("loss", C_LOSS, "circle-open")]:
            sub = wl_prob[wl_prob["Result_clean"] == label]
            fig_prob.add_trace(go.Scatter(
                x=sub["Match_Index"], y=sub["Win_prob"],
                mode="markers", name=label.capitalize(),
                marker=dict(color=color, size=9, symbol=sym,
                            line=dict(width=1.8, color=color)),
                customdata=sub.assign(
                    Date_str=sub["Date"].dt.strftime("%d %b %Y")
                )[["Date_str", "Partner", "Win_prob"]].values,
                hovertemplate=(
                    f"<b>{label.capitalize()}</b> · %{{customdata[0]}}<br>"
                    "Partner: %{customdata[1]}<br>"
                    "Model win prob: <b>%{y:.1f}%</b>"
                    "<extra></extra>"
                ),
            ))

        # 5-match trend line for probability
        if len(wl_prob) >= 4:
            prob_sorted = wl_prob.sort_values("Match_Index")
            fig_prob.add_trace(go.Scatter(
                x=prob_sorted["Match_Index"],
                y=prob_sorted["Win_prob"].rolling(5, min_periods=2).mean(),
                mode="lines", name="Trend (5-match avg)",
                line=dict(color=C_PROB, width=2, dash="dot"),
                hoverinfo="skip",
            ))
        
        # Annotate upsets and shock losses
        upsets = wl_prob[
            ((wl_prob["Result_clean"] == "loss") & (wl_prob["Win_prob"] > 80)) |
            ((wl_prob["Result_clean"] == "win")  & (wl_prob["Win_prob"] < 20))
        ]
        for _, row in upsets.iterrows():
            ann_label = "Upset!" if row["Result_clean"] == "win" else "Shock loss"
            ann_color = C_WIN   if row["Result_clean"] == "win" else C_LOSS
            fig_prob.add_annotation(
                x=row["Match_Index"], y=row["Win_prob"],
                text=ann_label, showarrow=True,
                arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor=ann_color,
                ax=0, ay=-25,
                font=dict(size=10, color=ann_color),
                bgcolor="rgba(0,0,0,0.3)", bordercolor=ann_color, borderwidth=1,
            )

        # Calculate x-axis ticks
        prob_n    = len(wl_prob)
        pstep     = max(1, prob_n // 8)
        ptick_idx = list(range(0, prob_n, pstep))
        if prob_n > 0 and (prob_n - 1) not in ptick_idx:
            ptick_idx.append(prob_n - 1)

        fig_prob.update_layout(
            **CHART_DEFAULTS, height=270,
            margin=_MARGIN_DEFAULT,
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        bgcolor="rgba(0,0,0,0)", borderwidth=0),
            yaxis=dict(**_YAXIS_DEFAULTS, range=[0, 102],
                       tickvals=[0, 25, 50, 75, 100],
                       ticktext=["0%", "25%", "50%", "75%", "100%"],
                       title="Model win probability"),
            xaxis=dict(
                **_XAXIS_DEFAULTS,
                tickmode="array",
                tickvals=wl_prob.iloc[ptick_idx]["Match_Index"].tolist() if prob_n else [],
                ticktext=wl_prob.iloc[ptick_idx]["Date"].dt.strftime("%d %b '%y").tolist() if prob_n else [],
            ),
        )
        st.plotly_chart(fig_prob, width='stretch')

    # ── right: calibration plot ────────────────────────────────────────────────
    with col6:
        fig_cal = go.Figure()

        # Probability bins
        wl_prob_sorted = wl_prob.sort_values("Win_prob").reset_index(drop=True)
        bin_count = 5
        wl_prob_sorted["Prob_bin"] = pd.cut(wl_prob_sorted["Win_prob"],
                                            bins=bin_count, labels=[f"{i*20}-{(i+1)*20}%" for i in range(bin_count)])
        calib = (
            wl_prob_sorted.groupby("Prob_bin", observed=True)
            .agg(Mid_prob=("Win_prob", "mean"), Wins=("Is_Win", "sum"), Count=("Is_Win", "size"))
            .reset_index()
        )
        calib = calib[calib["Count"] > 0].copy()
        calib["Actual_WR_pct"] = calib["Wins"] / calib["Count"] * 100

        # Perfect calibration reference
        fig_cal.add_trace(go.Scatter(
            x=[0, 100], y=[0, 100],
            mode="lines", name="Perfect calibration",
            line=dict(color="rgba(255,255,255,.2)", dash="dash", width=1.5),
            hoverinfo="skip",
        ))

        cal_colors = [
            C_WIN if a >= m else C_LOSS
            for a, m in zip(calib["Actual_WR_pct"], calib["Mid_prob"])
        ]
        fig_cal.add_trace(go.Bar(
            x=calib["Mid_prob"], y=calib["Actual_WR_pct"],
            name="Actual WR",
            marker_color=cal_colors, marker_cornerradius=4,
            opacity=0.75, width=10,
            customdata=calib[["Prob_bin", "Count"]].values,
            hovertemplate=(
                "Bucket: <b>%{customdata[0]}</b><br>"
                "Predicted: ~%{x:.0f}%<br>"
                "Actual WR: <b>%{y:.0f}%</b><br>"
                "Matches: %{customdata[1]}"
                "<extra></extra>"
            ),
        ))

        # Dots with sample-size labels
        fig_cal.add_trace(go.Scatter(
            x=calib["Mid_prob"], y=calib["Actual_WR_pct"],
            mode="markers+text", name="",
            marker=dict(color=cal_colors, size=9,
                        line=dict(width=1.5, color="rgba(0,0,0,.4)")),
            text=calib["Count"].apply(lambda c: f"n={c}"),
            textposition="top center",
            textfont=dict(size=9, color="rgba(255,255,255,.4)"),
            hoverinfo="skip", showlegend=False,
        ))

        fig_cal.update_layout(
            **CHART_DEFAULTS, height=270,
            margin=_MARGIN_DEFAULT,
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        bgcolor="rgba(0,0,0,0)", borderwidth=0),
            xaxis=dict(**_XAXIS_DEFAULTS,
                       title="Predicted win probability (%)", range=[0, 105]),
            yaxis=dict(**_YAXIS_DEFAULTS,
                       title="Actual win rate (%)", range=[0, 105],
                       tickvals=[0, 25, 50, 75, 100],
                       ticktext=["0%", "25%", "50%", "75%", "100%"]),
        )
        st.plotly_chart(fig_cal, width='stretch')

    # ── summary metrics ───────────────────────────────────────────────────────
    avg_pred   = wl_prob["Win_prob"].mean()
    avg_actual = wl_prob["Is_Win"].mean() * 100
    upsets_cnt = len(wl_prob[
        ((wl_prob["Result_clean"] == "loss") & (wl_prob["Win_prob"] > 80)) |
        ((wl_prob["Result_clean"] == "win")  & (wl_prob["Win_prob"] < 20))
    ])
    overperform    = avg_actual - avg_pred
    overperf_arrow = "▲" if overperform >= 0 else "▼"
    overperf_word  = "over" if overperform >= 0 else "under"
    overperf_color = C_WIN if overperform >= 0 else C_LOSS

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:4px;margin-bottom:1.8rem;">
      <div class="kpi-card">
        <div class="kpi-label">Avg predicted win prob</div>
        <div class="kpi-value" style="font-size:22px;color:{C_PROB}">{avg_pred:.1f}<span style="font-size:14px;opacity:.6">%</span></div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Avg actual win rate</div>
        <div class="kpi-value" style="font-size:22px;color:{overperf_color}">{avg_actual:.1f}<span style="font-size:14px;opacity:.6">%</span></div>
        <div class="kpi-delta">{overperf_arrow} {abs(overperform):.1f}pp {overperf_word}performing model</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Upset matches</div>
        <div class="kpi-value" style="font-size:22px;color:{C_AMBER}">{upsets_cnt}</div>
        <div class="kpi-delta">result vs. model &gt;20pp off</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.caption("Need at least 3 W/L matches to show probability analysis.")

# ── year over year ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Year over year</div>', unsafe_allow_html=True)

YEAR_PALETTE = ["#60A5FA", "#4ECDA4", "#FBBF24", "#A78BFA", "#F87171", "#F472B6", "#94A3B8", "#34D399"]
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def build_year_summary(real_df):
    """One row per calendar year: volume, results, rating movement and model stats."""
    d = real_df.sort_values("Match").copy()
    d["Year"] = d["Date"].dt.year
    d["Res"] = d["Result"].str.lower().str.strip()
    rows = []
    for year, g in d.groupby("Year"):
        wins   = int((g["Res"] == "win").sum())
        losses = int((g["Res"] == "loss").sum())
        ties   = int((g["Res"] == "tie").sum())
        total  = len(g)
        start  = g["Start_rating"].dropna().iloc[0] if g["Start_rating"].notna().any() else np.nan
        end    = g["New_rating"].dropna().iloc[-1] if g["New_rating"].notna().any() else np.nan

        wl_g = _wl(g)
        if len(wl_g):
            exp_wr = wl_g.apply(
                lambda r: calc_win_prob(r["Start_rating"], r["Partner_rating"],
                                        r["Opp1_rating"], r["Opp2_rating"]),
                axis=1,
            ).mean() * 100
            act_wr   = (wl_g["Result_clean"] == "win").mean() * 100
            avg_opp  = wl_g["Avg_Opp_Rating"].mean()
            avg_team = wl_g["Avg_Team_Rating"].mean()
        else:
            exp_wr = act_wr = avg_opp = avg_team = np.nan

        top = g["Partner"].value_counts()
        rows.append(dict(
            Year=int(year), Matches=total, Wins=wins, Losses=losses, Ties=ties,
            WinRate=wins / total * 100,
            StartRating=start, EndRating=end, RatingChange=end - start,
            BestRating=g["New_rating"].min(), WorstRating=g["New_rating"].max(),
            AvgDiff=g["Diff"].mean(),
            ExpWR=exp_wr, ActWR=act_wr, ModelDelta=act_wr - exp_wr,
            AvgOpp=avg_opp, AvgTeam=avg_team,
            Partners=int(g["Partner"].nunique()),
            TopPartner=top.index[0] if len(top) else "—",
        ))
    return pd.DataFrame(rows)


def _fmt_delta(value, fmt):
    """st.metric delta string, or None when there is nothing sensible to show."""
    return None if pd.isna(value) else format(value, fmt)


year_df = build_year_summary(real) if len(real) else pd.DataFrame()

if len(year_df) == 0:
    st.caption("No matches yet — year-over-year stats appear once you add some.")
else:
    year_df["dMatches"] = year_df["Matches"].diff()
    year_df["dWinRate"] = year_df["WinRate"].diff()
    year_df["dEnd"]     = year_df["EndRating"].diff()
    x_years    = year_df["Year"].astype(str)
    years_list = year_df["Year"].tolist()

    # ── season vs season ──────────────────────────────────────────────────────
    if len(years_list) >= 2:
        sel_a, sel_b, _sp = st.columns([1, 1, 4])
        year_a = sel_a.selectbox("Season", years_list, index=len(years_list) - 1, key="yoy_a")
        year_b = sel_b.selectbox("Compare with", years_list, index=len(years_list) - 2, key="yoy_b")
        ra = year_df[year_df["Year"] == year_a].iloc[0]
        rb = year_df[year_df["Year"] == year_b].iloc[0]

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Matches", int(ra["Matches"]),
                  _fmt_delta(ra["Matches"] - rb["Matches"], "+.0f"))
        m2.metric("Win rate", f"{ra['WinRate']:.0f}%",
                  f"{ra['WinRate'] - rb['WinRate']:+.1f} pp")
        m3.metric("Rating change in season", f"{ra['RatingChange']:+.4f}",
                  _fmt_delta(ra["RatingChange"] - rb["RatingChange"], "+.4f"),
                  delta_color="inverse", help="Negative = rating improved (lower is better)")
        m4.metric("End-of-season rating", f"{ra['EndRating']:.4f}",
                  _fmt_delta(ra["EndRating"] - rb["EndRating"], "+.4f"),
                  delta_color="inverse")
        m5.metric("Vs. model", f"{ra['ModelDelta']:+.1f} pp" if pd.notna(ra["ModelDelta"]) else "—",
                  (f"{ra['ModelDelta'] - rb['ModelDelta']:+.1f} pp"
                   if pd.notna(ra["ModelDelta"]) and pd.notna(rb["ModelDelta"]) else None),
                  help="Actual win rate minus the win rate the rating model expected")

    # ── row 1: win rate + volume | rating change ──────────────────────────────
    yc1, yc2 = st.columns([1, 1], gap="medium")

    with yc1:
        st.markdown('<div class="section-header">Win rate &amp; matches played</div>', unsafe_allow_html=True)
        fig_ywr = go.Figure()
        fig_ywr.add_bar(
            x=x_years, y=year_df["Matches"], name="Matches",
            marker_color="rgba(96,165,250,.35)", marker_cornerradius=4,
            hovertemplate="%{y} matches<extra></extra>",
        )
        fig_ywr.add_trace(go.Scatter(
            x=x_years, y=year_df["WinRate"], name="Win rate", yaxis="y2",
            mode="lines+markers+text", line=dict(color=C_PROB, width=2.5),
            marker=dict(size=9, color=[C_WIN if v >= 50 else C_LOSS for v in year_df["WinRate"]],
                        line=dict(width=1.5, color="rgba(0,0,0,.4)")),
            text=[f"{v:.0f}%" for v in year_df["WinRate"]], textposition="top center",
            textfont=dict(size=10, color="rgba(255,255,255,.7)"),
            customdata=year_df[["Wins", "Losses", "Ties"]].values,
            hovertemplate="Win rate <b>%{y:.1f}%</b><br>%{customdata[0]}W · %{customdata[1]}L · %{customdata[2]}T<extra></extra>",
        ))
        fig_ywr.add_shape(type="line", xref="paper", x0=0, x1=1, yref="y2", y0=50, y1=50,
                          line=dict(width=1, dash="dash", color="rgba(255,255,255,.18)"))
        fig_ywr.update_layout(
            **CHART_DEFAULTS, height=260, margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        bgcolor="rgba(0,0,0,0)", borderwidth=0),
            yaxis=dict(**_YAXIS_DEFAULTS, title="Matches"),
            yaxis2=dict(overlaying="y", side="right", range=[0, 110], showgrid=False,
                        zeroline=False, ticksuffix="%", automargin=True),
            xaxis=dict(**_XAXIS_DEFAULTS, type="category"),
        )
        st.plotly_chart(fig_ywr, width='stretch')

    with yc2:
        st.markdown('<div class="section-header">Rating change per season · lower is better</div>', unsafe_allow_html=True)
        fig_yrc = go.Figure(go.Bar(
            x=x_years, y=year_df["RatingChange"],
            marker_color=[C_WIN if v <= 0 else C_LOSS for v in year_df["RatingChange"]],
            marker_cornerradius=4,
            text=[f"{v:+.2f}" for v in year_df["RatingChange"]], textposition="outside",
            textfont=dict(size=10, color="rgba(255,255,255,.7)"),
            customdata=year_df[["StartRating", "EndRating", "BestRating", "WorstRating"]].values,
            hovertemplate=(
                "<b>%{x}</b><br>Change: <b>%{y:+.4f}</b><br>"
                "Start → end: %{customdata[0]:.4f} → %{customdata[1]:.4f}<br>"
                "Best / worst: %{customdata[2]:.4f} / %{customdata[3]:.4f}<extra></extra>"
            ),
        ))
        fig_yrc.add_hline(y=0, line_width=1, line_color="rgba(255,255,255,.25)")
        fig_yrc.update_layout(
            **CHART_DEFAULTS, height=260, margin=dict(l=0, r=0, t=30, b=0), showlegend=False,
            yaxis=dict(**_YAXIS_DEFAULTS, title="Rating Δ"),
            xaxis=dict(**_XAXIS_DEFAULTS, type="category"),
        )
        st.plotly_chart(fig_yrc, width='stretch')

    # ── row 2: actual vs expected | breakdown heatmap ─────────────────────────
    yc3, yc4 = st.columns([1, 1], gap="medium")

    with yc3:
        st.markdown('<div class="section-header">Actual vs. model-expected win rate</div>', unsafe_allow_html=True)
        model_df = year_df[year_df["ExpWR"].notna()]
        if len(model_df):
            fig_ymod = go.Figure()
            fig_ymod.add_bar(
                x=model_df["Year"].astype(str), y=model_df["ExpWR"], name="Expected (model)",
                marker_color="rgba(167,139,250,.55)", marker_cornerradius=4,
                hovertemplate="Expected <b>%{y:.1f}%</b><extra></extra>",
            )
            fig_ymod.add_bar(
                x=model_df["Year"].astype(str), y=model_df["ActWR"], name="Actual",
                marker_color=[C_WIN if a >= e else C_LOSS for a, e in zip(model_df["ActWR"], model_df["ExpWR"])],
                marker_cornerradius=4,
                text=[f"{d:+.0f}pp" for d in model_df["ModelDelta"]], textposition="outside",
                textfont=dict(size=10, color="rgba(255,255,255,.7)"),
                hovertemplate="Actual <b>%{y:.1f}%</b><extra></extra>",
            )
            fig_ymod.update_layout(
                **CHART_DEFAULTS, height=260, margin=dict(l=0, r=0, t=30, b=0),
                barmode="group", bargap=0.25, bargroupgap=0.08,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                            bgcolor="rgba(0,0,0,0)", borderwidth=0),
                yaxis=dict(**_YAXIS_DEFAULTS, range=[0, 110], ticksuffix="%"),
                xaxis=dict(**_XAXIS_DEFAULTS, type="category"),
            )
            st.plotly_chart(fig_ymod, width='stretch')
        else:
            st.caption("Needs at least one decided (W/L) match.")

    with yc4:
        st.markdown('<div class="section-header">Win rate by season &amp; match type</div>', unsafe_allow_html=True)
        dim = "Type"
        ctx = _wl(real)
        ctx["Year"] = ctx["Date"].dt.year.astype(str)
        ctx[dim] = ctx[dim].fillna("Unknown")
        ctx["Is_Win"] = (ctx["Result_clean"] == "win").astype(int)
        grp = ctx.groupby(["Year", dim]).agg(N=("Is_Win", "size"), Wins=("Is_Win", "sum")).reset_index()
        if len(grp):
            grp["WR"] = grp["Wins"] / grp["N"] * 100
            piv_wr = grp.pivot(index=dim, columns="Year", values="WR")
            piv_n  = grp.pivot(index=dim, columns="Year", values="N")
            text = [["" if pd.isna(v) else f"{v:.0f}%<br>n={int(piv_n.loc[r, c])}"
                     for c, v in zip(piv_wr.columns, piv_wr.loc[r])] for r in piv_wr.index]
            fig_ctx = go.Figure(go.Heatmap(
                z=piv_wr.values, x=piv_wr.columns.tolist(), y=piv_wr.index.tolist(),
                text=text, texttemplate="%{text}", textfont=dict(size=11, color="#fff"),
                zmin=0, zmax=100, showscale=False, hoverongaps=False, xgap=3, ygap=3,
                colorscale=[[0, "#7f2d2d"], [0.5, "#3a3f4b"], [1, "#1f7a63"]],
                hovertemplate="%{y} · %{x}<br>Win rate <b>%{z:.0f}%</b><extra></extra>",
            ))
            fig_ctx.update_layout(
                **CHART_DEFAULTS, height=max(200, 60 + 44 * len(piv_wr.index)),
                margin=dict(l=0, r=0, t=6, b=0),
                xaxis=dict(**_XAXIS_DEFAULTS, type="category"),
                yaxis=dict(showgrid=False, zeroline=False, autorange="reversed"),
            )
            st.plotly_chart(fig_ctx, width='stretch')
        else:
            st.caption("Needs at least one decided (W/L) match.")

    # ── row 3: season trajectories | monthly activity ─────────────────────────
    yc5, yc6 = st.columns([1, 1], gap="medium")

    with yc5:
        st.markdown('<div class="section-header">Season trajectories · rating since season start</div>', unsafe_allow_html=True)
        traj = real.sort_values("Match").copy()
        traj["Year"] = traj["Date"].dt.year
        fig_traj = go.Figure()
        for i, (year, g) in enumerate(traj.groupby("Year")):
            g = g[g["New_rating"].notna() & g["Start_rating"].notna()]
            if len(g) == 0:
                continue
            base = g["Start_rating"].iloc[0]
            fig_traj.add_trace(go.Scatter(
                x=list(range(1, len(g) + 1)), y=(g["New_rating"] - base).round(4),
                mode="lines+markers", name=str(year),
                line=dict(color=YEAR_PALETTE[i % len(YEAR_PALETTE)], width=2),
                marker=dict(size=5),
                customdata=g["Date"].dt.strftime("%d %b %Y"),
                hovertemplate=f"<b>{year}</b> · match %{{x}}<br>%{{customdata}}<br>Δ since start: <b>%{{y:+.4f}}</b><extra></extra>",
            ))
        fig_traj.add_hline(y=0, line_width=1, line_dash="dash", line_color="rgba(255,255,255,.2)")
        fig_traj.update_layout(
            **CHART_DEFAULTS, height=260, margin=dict(l=0, r=0, t=30, b=0),
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        bgcolor="rgba(0,0,0,0)", borderwidth=0),
            xaxis=dict(**_XAXIS_DEFAULTS, title="Match # in season"),
            yaxis=dict(**_YAXIS_DEFAULTS, title="Rating Δ (lower = better)"),
        )
        st.plotly_chart(fig_traj, width='stretch')

    with yc6:
        st.markdown('<div class="section-header">Matches per month</div>', unsafe_allow_html=True)
        mm = real.assign(Year=real["Date"].dt.year.astype(str), Month=real["Date"].dt.month)
        piv_m = (mm.pivot_table(index="Year", columns="Month", values="Match", aggfunc="count")
                   .reindex(columns=range(1, 13)))
        text_m = [["" if pd.isna(v) else str(int(v)) for v in row] for row in piv_m.values]
        fig_mon = go.Figure(go.Heatmap(
            z=piv_m.values, x=MONTH_LABELS, y=piv_m.index.tolist(),
            text=text_m, texttemplate="%{text}", textfont=dict(size=11, color="#fff"),
            showscale=False, hoverongaps=False, xgap=3, ygap=3,
            colorscale=[[0, "#1b2a41"], [1, "#3B82F6"]],
            hovertemplate="%{y} %{x}: <b>%{z}</b> matches<extra></extra>",
        ))
        fig_mon.update_layout(
            **CHART_DEFAULTS, height=max(200, 60 + 44 * len(piv_m.index)),
            margin=dict(l=0, r=0, t=6, b=0),
            xaxis=dict(**_XAXIS_DEFAULTS),
            yaxis=dict(showgrid=False, zeroline=False, type="category", autorange="reversed"),
        )
        st.plotly_chart(fig_mon, width='stretch')

    # ── season table ──────────────────────────────────────────────────────────
    season_tbl = year_df.rename(columns={
        "dMatches": "Δ Matches", "dWinRate": "Δ Win rate", "WinRate": "Win rate",
        "StartRating": "Start", "EndRating": "End", "dEnd": "Δ End rating",
        "RatingChange": "Rating Δ", "BestRating": "Best", "AvgDiff": "Avg diff",
        "ExpWR": "Model WR", "ModelDelta": "Vs. model", "AvgOpp": "Avg opp",
        "AvgTeam": "Avg team", "TopPartner": "Top partner",
        "Wins": "W", "Losses": "L", "Ties": "T",
    })[["Year", "Matches", "Δ Matches", "W", "L", "T", "Win rate", "Δ Win rate",
        "Start", "End", "Δ End rating", "Rating Δ", "Best", "Avg diff",
        "Model WR", "Vs. model", "Avg opp", "Avg team", "Partners", "Top partner"]]
    st.dataframe(
        season_tbl.sort_values("Year", ascending=False),
        width='stretch', hide_index=True,
        column_config={
            "Year":         st.column_config.NumberColumn("Year", format="%d"),
            "Δ Matches":    st.column_config.NumberColumn("Δ Matches", format="%+d"),
            "Win rate":     st.column_config.NumberColumn("Win rate", format="%.0f%%"),
            "Δ Win rate":   st.column_config.NumberColumn("Δ WR (pp)", format="%+.1f"),
            "Start":        st.column_config.NumberColumn("Start", format="%.4f"),
            "End":          st.column_config.NumberColumn("End", format="%.4f"),
            "Δ End rating": st.column_config.NumberColumn("Δ End", format="%+.4f",
                                help="Change in end-of-season rating vs. previous season (negative = better)"),
            "Rating Δ":     st.column_config.NumberColumn("Rating Δ", format="%+.4f",
                                help="Movement within the season (negative = better)"),
            "Best":         st.column_config.NumberColumn("Best", format="%.4f"),
            "Avg diff":     st.column_config.NumberColumn("Avg diff", format="%+.4f"),
            "Model WR":     st.column_config.NumberColumn("Model WR", format="%.1f%%"),
            "Vs. model":    st.column_config.NumberColumn("Vs. model (pp)", format="%+.1f"),
            "Avg opp":      st.column_config.NumberColumn("Avg opp", format="%.2f"),
            "Avg team":     st.column_config.NumberColumn("Avg team", format="%.2f"),
        },
    )
    st.caption("Δ columns compare each season with the previous season played. "
               "Rating: lower is better, so negative rating changes are improvements.")


# ── partners ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Partner performance · 3+ matches</div>', unsafe_allow_html=True)
wl_df = _wl(real)
wl_df["Is_Win"] = wl_df["Result_clean"] == "win"
wl_df["Is_Loss"] = wl_df["Result_clean"] == "loss"
partner_stats = (
    wl_df.groupby("Partner")
    .agg(Matches=("Partner", "size"), Wins=("Is_Win", "sum"),
         Losses=("Is_Loss", "sum"), MeanDiff=("Diff", "mean"))
    .reset_index()
    .pipe(lambda d: d[d["Matches"] >= 3])
    .assign(
        WinRate=lambda d: (d["Wins"] / d["Matches"] * 100).round(0).astype(int),
        MeanDiff=lambda d: d["MeanDiff"].round(4),
        Matches=lambda d: d["Matches"].astype(int),
        Wins=lambda d: d["Wins"].astype(int),
        Losses=lambda d: d["Losses"].astype(int),
    )
    .sort_values(["Matches", "WinRate"], ascending=[False, False])
)

rows_html = """
<div class="pt-row">
  <span class="pt-header">Partner</span>
  <span class="pt-header">Matches</span>
  <span class="pt-header">W / L</span>
  <span class="pt-header">Avg diff</span>
  <span class="pt-header">Win rate</span>
</div>"""
for r in partner_stats.itertuples():
    bar_color  = C_WIN if r.WinRate >= 50 else C_LOSS
    diff_color = C_WIN if r.MeanDiff <= 0 else C_LOSS
    diff_sign  = "+" if r.MeanDiff >= 0 else ""
    rows_html += f"""
<div class="pt-row">
  <span class="pt-name">{r.Partner}</span>
  <span class="pt-matches">{r.Matches}</span>
  <span>
    <span class="pill pill-w">{r.Wins}W</span>&nbsp;
    <span class="pill pill-l">{r.Losses}L</span>
  </span>
  <span style="font-weight:600;color:{diff_color}">{diff_sign}{r.MeanDiff}</span>
  <div class="bar-wrap">
    <span class="wr-text">{r.WinRate}%</span>
    <div class="bar-bg"><div class="bar-fill" style="width:{r.WinRate}%;background:{bar_color}"></div></div>
  </div>
</div>"""
st.markdown(rows_html, unsafe_allow_html=True)

# ── full log + download ────────────────────────────────────────────────────────
with st.expander("📋 Full match log"):
    show = df.copy()
    show["Result"]   = show["Result"].apply(result_label)
    show["Date_str"] = show["Date"].dt.strftime("%d %b %Y")

    csv_bytes = (
        df.copy()
        .assign(Date=df["Date"].dt.strftime("%d/%m/%Y"))
        .to_csv(index=False)
        .encode("utf-8")
    )
    st.download_button(
        label="⬇️ Download CSV",
        data=csv_bytes,
        file_name=DATA_FILE,
        mime="text/csv",
    )

    st.dataframe(
        show[["Match", "Date_str", "Start_rating", "Diff", "New_rating", "Result",
              "Partner", "Partner_rating", "Opp1_rating", "Opp2_rating",
              "Set1", "Set2", "Set3", "Surface", "Type", "Phase"]]
        .rename(columns={"Date_str": "Date"}),
        width='stretch',
        hide_index=True,
        column_config={"Diff": st.column_config.NumberColumn("Δ", format="%.4f")},
    )

# ── calculator ────────────────────────────────────────────────────────────────
with st.expander("🧮 KNLTB diff calculator"):
    cc1, cc2, cc3 = st.columns(3)
    cr1  = cc1.number_input("Your rating",    value=last_rating, step=0.0001, format="%.4f", key="cr1")
    cr2  = cc1.number_input("Partner rating", value=7.50, step=0.0001, format="%.4f", key="cr2")
    cr3  = cc2.number_input("Opp 1 rating",   value=8.00, step=0.0001, format="%.4f", key="cr3")
    cr4  = cc2.number_input("Opp 2 rating",   value=8.00, step=0.0001, format="%.4f", key="cr4")
    cres = cc3.selectbox("Result", ["win", "loss", "tie"], key="cres")
    d    = calc_diff(cr1, cr2, cr3, cr4, cres)
    wp   = calc_win_prob(cr1, cr2, cr3, cr4) * 100
    cc3.metric("Model win probability", f"{wp:.1f}%",
               help="Probability the KNLTB model assigns to winning this match")
    cc3.metric("Expected diff",  f"{d:+.4f}", help="Negative = rating drop (better)")
    cc3.metric("New rating",     f"{cr1 + d:.4f}")
