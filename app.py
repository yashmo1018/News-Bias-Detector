"""
Layer 4 — Streamlit Dashboard (app.py)
News Bias Detector · Portfolio Edition

Run:
    pip install streamlit plotly pandas
    streamlit run app.py
"""

import json
import math
import os
import subprocess
import sys
import threading
import time as time_module
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from collections import defaultdict

# ──────────────────────────────────────────────
#  Cloud detection
# ──────────────────────────────────────────────

def is_cloud_env() -> bool:
    """
    Returns True when running on Streamlit Cloud (ephemeral filesystem).
    Detection order:
      1. STREAMLIT_CLOUD env var set to any non-empty value  ← set this in
         Streamlit Cloud > App settings > Secrets / Environment variables
      2. Presence of the /.streamlit/secrets.toml mount path that Cloud injects
      3. HOME=/home/appuser  (Streamlit Cloud's default user)
    Any one match is sufficient.
    """
    if os.environ.get("STREAMLIT_CLOUD"):
        return True
    if Path("/.streamlit/secrets.toml").exists():
        return True
    if os.environ.get("HOME", "") == "/home/appuser":
        return True
    return False

CLOUD = is_cloud_env()

# ──────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────
DATA_FILE        = "clustered.json"
FALLBACK         = "analyzed.json"
ARTICLES_FILE    = "articles.json"
REFRESH_INTERVAL = 6        # hours — refresh if data is older than this
ROLLING_DAYS     = 7        # days to keep in rolling window

BIAS_ORDER  = ["Left", "Left-Center", "Center", "Right-Center", "Right", "Unknown"]
BIAS_COLORS = {
    "Left":         "#4e79a7",
    "Left-Center":  "#76b7b2",
    "Center":       "#59a14f",
    "Right-Center": "#f28e2b",
    "Right":        "#e15759",
    "Unknown":      "#9e9e9e",
}
FRAMING_COLORS = px.colors.qualitative.Pastel

st.set_page_config(
    page_title="News Bias Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
#  Styling
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 { font-family: 'Playfair Display', serif !important; }

.bias-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px;
}
.card {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    border-left: 4px solid #ccc;
}
.metric-num {
    font-size: 2rem;
    font-weight: 700;
    font-family: 'Playfair Display', serif;
}
.refresh-box {
    background: #1e2a3a;
    border: 1px solid #4a90d9;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.85rem;
    margin-bottom: 12px;
    color: #e0e0e0;
}
.refresh-running {
    background: #2a2410;
    border-color: #f0c040;
    color: #f0e0a0;
}
.refresh-done {
    background: #0f2a18;
    border-color: #4caf50;
    color: #a5d6a7;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
#  Auto-refresh engine
# ──────────────────────────────────────────────

def get_data_age_hours() -> float | None:
    """
    Returns how many hours ago articles.json was last modified.
    Returns None if the file doesn't exist.
    """
    p = Path(ARTICLES_FILE)
    if not p.exists():
        return None
    mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
    delta = datetime.now(timezone.utc) - mtime
    return delta.total_seconds() / 3600


def is_data_stale(threshold_hours: float = REFRESH_INTERVAL) -> bool:
    age = get_data_age_hours()
    if age is None:
        return True   # no file at all → definitely stale
    return age >= threshold_hours


def run_pipeline(status_key: str = "pipeline_status") -> None:
    """
    Runs scraper → analyzer → clusterer in sequence.
    Updates st.session_state[status_key] with progress strings.
    Designed to be called in a background thread.
    """
    steps = [
        ("scraper",   ["python", "scraper.py"]),
        ("analyzer",  ["python", "analyzer.py"]),
        ("clusterer", ["python", "clusterer.py", "--eps", "0.40"]),
    ]
    for step_name, cmd in steps:
        st.session_state[status_key] = f"running:{step_name}"
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,   # 5-min timeout per step
            )
            if result.returncode != 0:
                st.session_state[status_key] = f"error:{step_name}:{result.stderr[:200]}"
                return
        except subprocess.TimeoutExpired:
            st.session_state[status_key] = f"error:{step_name}:timeout"
            return
        except FileNotFoundError:
            st.session_state[status_key] = f"error:{step_name}:not found"
            return

    st.session_state[status_key] = "done"


def trigger_pipeline_background() -> None:
    """Kick off the pipeline in a background thread (non-blocking)."""
    t = threading.Thread(target=run_pipeline, daemon=True)
    t.start()


# ──────────────────────────────────────────────
#  Refresh state management
# ──────────────────────────────────────────────

if "pipeline_status" not in st.session_state:
    st.session_state["pipeline_status"] = "idle"

if "auto_refresh_triggered" not in st.session_state:
    st.session_state["auto_refresh_triggered"] = False

# Auto-trigger on load if stale (only once per session, never on cloud)
if (
    not CLOUD
    and not st.session_state["auto_refresh_triggered"]
    and st.session_state["pipeline_status"] == "idle"
    and is_data_stale()
):
    st.session_state["auto_refresh_triggered"] = True
    st.session_state["pipeline_status"] = "running:scraper"
    trigger_pipeline_background()


# ──────────────────────────────────────────────
#  Data loading
# ──────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    path = DATA_FILE if Path(DATA_FILE).exists() else FALLBACK
    if not Path(path).exists():
        st.error("No data file found. Run scraper → analyzer → clusterer first.")
        st.stop()

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for a in raw:
        analysis = a.get("analysis", {})
        def get(key, fallback=None):
            return a.get(key) or analysis.get(key) or fallback
        rows.append({
            "source":        a.get("source", "Unknown"),
            "title":         a.get("title", ""),
            "url":           a.get("url", ""),
            "scraped_at":    a.get("scraped_at", ""),
            "bias_label":    get("bias_label", "Unknown"),
            "sentiment":     get("sentiment_score", 0.0),
            "framing":       get("framing", "Unknown"),
            "reasoning":     get("reasoning", ""),
            "key_entities":  get("key_entities") or [],
            "group_id":      a.get("group_id", "noise_0"),
            "cluster_label": a.get("cluster_label", "Unclustered"),
            "is_noise":      a.get("is_noise", True),
        })

    df = pd.DataFrame(rows)
    df["bias_label"] = df["bias_label"].apply(
        lambda x: x if x in BIAS_ORDER else "Unknown"
    )
    return df


df = load_data()

# ──────────────────────────────────────────────
#  Sidebar
# ──────────────────────────────────────────────
st.sidebar.title("📰 News Bias Detector")
st.sidebar.caption("Powered by Groq · LLaMA 3.3 70B")
st.sidebar.markdown("---")

# ── Auto-refresh controls ────────────────────
st.sidebar.subheader("🔄 Data Refresh")

age = get_data_age_hours()
if age is None:
    age_str = "No data yet"
elif age < 1:
    age_str = f"{int(age * 60)} min ago"
else:
    age_str = f"{age:.1f} hrs ago"

status = st.session_state["pipeline_status"]

if CLOUD:
    # On Streamlit Cloud the filesystem is ephemeral — pipeline can't run here.
    # Just show data freshness and explain the workflow.
    st.sidebar.markdown(
        f"<div class='refresh-box'>📡 Live dashboard · {age_str}<br>"
        f"<small>To update: run pipeline locally → commit clustered.json → push</small></div>",
        unsafe_allow_html=True,
    )
elif status.startswith("running:"):
    step = status.split(":")[1]
    step_labels = {"scraper": "1/3 Scraping feeds…", "analyzer": "2/3 Analyzing with LLM…", "clusterer": "3/3 Clustering events…"}
    label = step_labels.get(step, "Running…")
    st.sidebar.markdown(f"<div class='refresh-box refresh-running'>⏳ {label}<br><small>Last data: {age_str}</small></div>", unsafe_allow_html=True)
    # Rerun every 3s while pipeline is running to pick up status changes
    time_module.sleep(3)
    st.rerun()

elif status == "done":
    st.sidebar.markdown(f"<div class='refresh-box refresh-done'>✅ Refresh complete!<br><small>Data age: just now</small></div>", unsafe_allow_html=True)
    st.session_state["pipeline_status"] = "idle"
    # Clear cache so new data is loaded
    st.cache_data.clear()
    st.rerun()

elif status.startswith("error:"):
    parts = status.split(":", 2)
    failed_step = parts[1] if len(parts) > 1 else "?"
    err_msg = parts[2] if len(parts) > 2 else "unknown error"
    st.sidebar.markdown(f"<div class='refresh-box' style='background:#2a1010;border-color:#ef9a9a;color:#ffcdd2'>❌ Failed at {failed_step}<br><small>{err_msg[:80]}</small></div>", unsafe_allow_html=True)
    st.sidebar.caption("Check that GROQ_API_KEY is set and pipeline scripts are present.")

else:  # idle
    stale_indicator = "⚠️ Stale" if is_data_stale() else "✅ Fresh"
    st.sidebar.markdown(f"<div class='refresh-box'>{stale_indicator} · Last updated: {age_str}<br><small>Auto-refreshes every {REFRESH_INTERVAL}h</small></div>", unsafe_allow_html=True)

if not CLOUD:
    refresh_interval = st.sidebar.slider("Auto-refresh threshold (hrs)", 1, 24, REFRESH_INTERVAL)

    if st.sidebar.button("🔄 Refresh now", disabled=status.startswith("running:")):
        st.session_state["pipeline_status"] = "running:scraper"
        trigger_pipeline_background()
        st.rerun()

st.sidebar.markdown("---")

# ── Filters ──────────────────────────────────
all_sources = sorted(df["source"].unique())
sel_sources = st.sidebar.multiselect("Filter by outlet", all_sources, default=all_sources)

all_bias = [b for b in BIAS_ORDER if b in df["bias_label"].unique()]
sel_bias = st.sidebar.multiselect("Filter by bias", all_bias, default=all_bias)

show_noise = st.sidebar.checkbox("Include unclustered articles", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Cluster EPS tip:** Re-run `clusterer.py --eps 0.3` for tighter grouping, `--eps 0.45` for looser.")

# Apply filters
mask = (
    df["source"].isin(sel_sources) &
    df["bias_label"].isin(sel_bias) &
    (show_noise | ~df["is_noise"])
)
fdf = df[mask].copy()

# ──────────────────────────────────────────────
#  Header metrics
# ──────────────────────────────────────────────
st.title("News Bias Detector")
st.caption(f"Analyzing {len(fdf)} articles across {fdf['source'].nunique()} outlets · Rolling {ROLLING_DAYS}-day window")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"<div class='metric-num'>{len(fdf)}</div><div>Articles</div>", unsafe_allow_html=True)
with col2:
    clusters = fdf[~fdf["is_noise"]]["group_id"].nunique()
    st.markdown(f"<div class='metric-num'>{clusters}</div><div>Event Clusters</div>", unsafe_allow_html=True)
with col3:
    avg_sent = fdf["sentiment"].mean()
    color = "#e15759" if avg_sent < 0 else "#59a14f"
    st.markdown(f"<div class='metric-num' style='color:{color}'>{avg_sent:+.2f}</div><div>Avg Sentiment</div>", unsafe_allow_html=True)
with col4:
    dom_bias = fdf["bias_label"].value_counts().idxmax() if len(fdf) else "—"
    st.markdown(f"<div class='metric-num' style='font-size:1.4rem'>{dom_bias}</div><div>Dominant Bias</div>", unsafe_allow_html=True)

st.markdown("---")

# ──────────────────────────────────────────────
#  Tab layout
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🗂️ Coverage Gaps", "🌡️ Entity Heatmap", "📡 Bias Radar", "📋 Article Browser"
])


# ══════════════════════════════════════════════
#  TAB 1 — Coverage Gaps
# ══════════════════════════════════════════════
with tab1:
    st.subheader("Coverage Gap Analysis")
    st.caption("Each row = a news cluster (same real-world event). Columns = outlets. Color = bias label.")

    clustered = fdf[~fdf["is_noise"]].copy()
    if clustered.empty:
        st.info("No clusters found. Run `clusterer.py` to group articles by event, or lower `--eps`.")
    else:
        pivot_data = clustered.groupby(["cluster_label", "source"])["bias_label"].first().reset_index()
        pivot = pivot_data.pivot(index="cluster_label", columns="source", values="bias_label").fillna("—")

        sent_pivot = clustered.groupby(["cluster_label", "source"])["sentiment"].mean()

        pol = clustered.groupby("cluster_label")["sentiment"].std().fillna(0).rename("polarization")
        pivot = pivot.join(pol)
        pivot = pivot.sort_values("polarization", ascending=False)
        pol_col = pivot.pop("polarization")

        def bias_cell(val):
            color = BIAS_COLORS.get(val, "#ccc")
            if val == "—":
                return f"<td style='color:#aaa;text-align:center'>—</td>"
            return f"<td style='background:{color}22;color:{color};font-weight:600;text-align:center;padding:6px 10px;border-radius:6px'>{val}</td>"

        html = "<table style='width:100%;border-collapse:separate;border-spacing:4px'>"
        html += "<tr><th style='text-align:left;padding:4px 8px'>Event Cluster</th>"
        for src in pivot.columns:
            html += f"<th style='text-align:center;padding:4px 8px'>{src}</th>"
        html += "<th style='text-align:center'>Polarization ↑</th></tr>"
        for cluster, row in pivot.iterrows():
            pol_score = pol_col[cluster]
            bar = "█" * min(int(pol_score * 10), 10)
            short_label = cluster[:50] + ("…" if len(cluster) > 50 else "")
            html += f"<tr><td style='padding:6px 8px;font-size:0.85rem'>{short_label}</td>"
            for src in pivot.columns:
                html += bias_cell(row[src])
            html += f"<td style='text-align:center;font-size:0.8rem;color:#e15759'>{bar} {pol_score:.2f}</td></tr>"
        html += "</table>"
        st.markdown(html, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Sentiment Distribution by Outlet")
        fig = px.box(
            fdf, x="source", y="sentiment", color="source",
            points="all", color_discrete_sequence=px.colors.qualitative.G10,
            labels={"sentiment": "Sentiment Score (−1 negative → +1 positive)", "source": "Outlet"}
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 2 — Entity Heatmap
# ══════════════════════════════════════════════
with tab2:
    st.subheader("Entity × Outlet Portrayal Heatmap")
    st.caption("How each outlet portrays the same entities. Red = Negative, Green = Positive, Grey = Neutral.")

    entity_rows = []
    for _, row in fdf.iterrows():
        for ent in row["key_entities"]:
            if isinstance(ent, dict):
                portrayal = ent.get("portrayal", "Neutral").lower()
                score = 1 if "positive" in portrayal else (-1 if "negative" in portrayal else 0)
                entity_rows.append({
                    "source": row["source"],
                    "entity": ent.get("name", "Unknown"),
                    "score":  score,
                })

    if not entity_rows:
        st.info("No entity data found. Check that `key_entities` is populated in analyzed.json.")
    else:
        edf = pd.DataFrame(entity_rows)
        top_n = st.slider("Show top N entities", 5, 30, 15)
        top_entities = edf["entity"].value_counts().head(top_n).index.tolist()
        edf = edf[edf["entity"].isin(top_entities)]

        heat = edf.groupby(["entity", "source"])["score"].mean().reset_index()
        heat_pivot = heat.pivot(index="entity", columns="source", values="score")

        text_vals = []
        for row_vals in heat_pivot.values:
            text_row = []
            for v in row_vals:
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    text_row.append("")
                elif v > 0:
                    text_row.append("Positive")
                elif v < 0:
                    text_row.append("Negative")
                else:
                    text_row.append("Neutral")
            text_vals.append(text_row)

        fig = go.Figure(data=go.Heatmap(
            z=heat_pivot.values,
            x=heat_pivot.columns.tolist(),
            y=heat_pivot.index.tolist(),
            colorscale=[
                [0.0, "#e15759"],
                [0.5, "#d0d0d0"],
                [1.0, "#59a14f"],
            ],
            zmin=-1, zmax=1,
            text=text_vals,
            texttemplate="%{text}",
            hoverongaps=False,
            colorbar=dict(title="Portrayal", tickvals=[-1, 0, 1],
                          ticktext=["Negative", "Neutral", "Positive"]),
        ))
        fig.update_layout(
            height=max(300, 30 * len(top_entities)),
            xaxis_title="Outlet",
            yaxis_title="Entity",
            font=dict(size=12),
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 3 — Bias Radar
# ══════════════════════════════════════════════
with tab3:
    st.subheader("Outlet Bias Radar Chart")
    st.caption("Per-outlet 'personality' across bias dimensions. Larger area = more extreme bias distribution.")

    radar_data = fdf.groupby(["source", "bias_label"]).size().reset_index(name="count")
    totals = radar_data.groupby("source")["count"].sum()
    radar_data["pct"] = radar_data.apply(lambda r: r["count"] / totals[r["source"]], axis=1)

    bias_dims = [b for b in BIAS_ORDER if b != "Unknown"]
    sources   = sorted(fdf["source"].unique())

    fig = go.Figure()
    for src in sources:
        src_data = radar_data[radar_data["source"] == src]
        r_vals = []
        for b in bias_dims:
            row = src_data[src_data["bias_label"] == b]
            r_vals.append(float(row["pct"].values[0]) if not row.empty else 0.0)
        r_vals.append(r_vals[0])

        fig.add_trace(go.Scatterpolar(
            r=r_vals,
            theta=bias_dims + [bias_dims[0]],
            fill="toself",
            name=src,
            opacity=0.65,
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(255,255,255,0.05)",
            radialaxis=dict(visible=True, range=[0, 1], color="#aaa", gridcolor="#444"),
            angularaxis=dict(color="#ddd", gridcolor="#444"),
        ),
        showlegend=True,
        height=500,
        font=dict(color="#ddd"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Framing Type Distribution")
    frame_counts = fdf.groupby(["source", "framing"]).size().reset_index(name="count")
    fig2 = px.bar(
        frame_counts, x="source", y="count", color="framing",
        barmode="stack", color_discrete_sequence=FRAMING_COLORS,
        labels={"source": "Outlet", "count": "# Articles", "framing": "Framing"},
    )
    fig2.update_layout(height=350)
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 4 — Article Browser
# ══════════════════════════════════════════════
with tab4:
    st.subheader("Article Browser")

    search = st.text_input("🔍 Search titles / reasoning", "")
    sort_by = st.selectbox("Sort by", ["sentiment ↑", "sentiment ↓", "source", "cluster_label", "newest first"])

    bdf = fdf.copy()
    if search:
        mask2 = (
            bdf["title"].str.contains(search, case=False, na=False) |
            bdf["reasoning"].str.contains(search, case=False, na=False)
        )
        bdf = bdf[mask2]

    sort_map = {
        "sentiment ↑":    ("sentiment", True),
        "sentiment ↓":    ("sentiment", False),
        "source":          ("source", True),
        "cluster_label":   ("cluster_label", True),
        "newest first":    ("scraped_at", False),
    }
    col, asc = sort_map[sort_by]
    bdf = bdf.sort_values(col, ascending=asc)

    st.caption(f"Showing {len(bdf)} articles")

    for _, row in bdf.iterrows():
        bias_color = BIAS_COLORS.get(row["bias_label"], "#999")
        sent_emoji = "😊" if row["sentiment"] > 0.2 else ("😡" if row["sentiment"] < -0.2 else "😐")

        # Format scraped_at nicely
        scraped_label = ""
        if row.get("scraped_at"):
            try:
                dt = datetime.fromisoformat(row["scraped_at"])
                scraped_label = f" · {dt.strftime('%d %b %H:%M UTC')}"
            except ValueError:
                pass

        with st.expander(f"[{row['source']}] {row['title'][:80]}{scraped_label}"):
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                st.markdown(f"**Bias:** <span style='color:{bias_color};font-weight:700'>{row['bias_label']}</span>", unsafe_allow_html=True)
                st.markdown(f"**Framing:** {row['framing']}")
            with c2:
                st.markdown(f"**Sentiment:** {sent_emoji} `{row['sentiment']:+.2f}`")
                st.markdown(f"**Cluster:** {row['cluster_label'][:40]}")
            with c3:
                st.markdown(f"**Reasoning:** {row['reasoning']}")

            if row["key_entities"]:
                ent_html = " ".join(
                    f"<span class='bias-pill' style='background:{BIAS_COLORS.get('Left','#eee')}22;color:#333'>{e.get('name','')} ({e.get('portrayal','')})</span>"
                    for e in row["key_entities"] if isinstance(e, dict)
                )
                st.markdown(f"**Entities:** {ent_html}", unsafe_allow_html=True)

            st.markdown(f"[🔗 Read original]({row['url']})")


