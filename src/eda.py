"""
eda.py
Reusable chart and summary-table functions for EDA and the Streamlit app.
All functions return Plotly figures so they render nicely in Streamlit.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Distribution charts
# ---------------------------------------------------------------------------

def plot_attack_family_distribution(df):
    """Bar chart of attack family counts."""
    counts = df["attack_family"].value_counts().reset_index()
    counts.columns = ["Attack Family", "Count"]
    fig = px.bar(
        counts, x="Attack Family", y="Count",
        color="Attack Family",
        title="Attack Family Distribution",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(showlegend=False)
    return fig


def plot_protocol_distribution(df):
    """Pie chart of protocol_type."""
    counts = df["protocol_type"].value_counts().reset_index()
    counts.columns = ["Protocol", "Count"]
    fig = px.pie(
        counts, names="Protocol", values="Count",
        title="Protocol Type Distribution",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    return fig


def plot_service_top_n(df, n=15):
    """Horizontal bar of top-N services."""
    counts = df["service"].value_counts().head(n).reset_index()
    counts.columns = ["Service", "Count"]
    fig = px.bar(
        counts, y="Service", x="Count", orientation="h",
        title=f"Top {n} Services",
        color="Count", color_continuous_scale="Blues",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    return fig


def plot_flag_distribution(df):
    """Bar chart of flag values."""
    counts = df["flag"].value_counts().reset_index()
    counts.columns = ["Flag", "Count"]
    fig = px.bar(
        counts, x="Flag", y="Count",
        title="Connection Flag Distribution",
        color="Flag",
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )
    fig.update_layout(showlegend=False)
    return fig


# ---------------------------------------------------------------------------
# Correlation and feature analysis
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(df, top_n=15):
    """Heatmap of top correlated numeric features."""
    numeric_df = df.select_dtypes(include=[np.number])
    # Drop near-zero variance columns
    numeric_df = numeric_df.loc[:, numeric_df.std() > 0.01]
    corr = numeric_df.corr().abs()

    # Pick features most correlated with is_suspicious
    if "is_suspicious" in corr.columns:
        top_feats = corr["is_suspicious"].sort_values(ascending=False).head(top_n).index.tolist()
    else:
        top_feats = corr.mean().sort_values(ascending=False).head(top_n).index.tolist()

    sub_corr = numeric_df[top_feats].corr()
    fig = px.imshow(
        sub_corr, text_auto=".2f",
        title=f"Correlation Heatmap (Top {top_n} Features)",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        aspect="auto",
    )
    return fig


def plot_boxplots_suspicious(df, features=None):
    """Box plots comparing normal vs suspicious for key numeric features."""
    if features is None:
        features = ["duration", "src_bytes", "dst_bytes", "count",
                     "srv_count", "serror_rate", "dst_host_srv_count"]
    features = [f for f in features if f in df.columns]

    fig = make_subplots(rows=2, cols=4,
                        subplot_titles=features[:8])
    for i, feat in enumerate(features[:8]):
        row, col = i // 4 + 1, i % 4 + 1
        for label, color in [("normal", "#2ecc71"), ("suspicious", "#e74c3c")]:
            subset = df[df["is_suspicious"] == (0 if label == "normal" else 1)][feat]
            fig.add_trace(
                go.Box(y=subset, name=label, marker_color=color, showlegend=(i == 0)),
                row=row, col=col,
            )
    fig.update_layout(title="Normal vs Suspicious — Key Features", height=500)
    return fig


# ---------------------------------------------------------------------------
# PCA 2D visualization
# ---------------------------------------------------------------------------

def plot_pca_2d(df, max_points=5000):
    """PCA 2D scatter of normal vs attack families."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_use = [c for c in numeric_cols if c not in ("is_suspicious",)]

    sample = df.sample(n=min(max_points, len(df)), random_state=42)
    X = sample[cols_to_use].fillna(0).values
    X = StandardScaler().fit_transform(X)
    X_pca = PCA(n_components=2, random_state=42).fit_transform(X)

    plot_df = pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1],
                            "Family": sample["attack_family"].values})
    fig = px.scatter(
        plot_df, x="PC1", y="PC2", color="Family",
        title="PCA 2D — Attack Families",
        opacity=0.6,
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_traces(marker_size=4)
    return fig


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def get_summary_stats(df):
    """Return a summary DataFrame for display."""
    summary = {
        "Total Records": len(df),
        "Features": len(df.columns) - 3,  # minus label, attack_family, is_suspicious
        "Protocols": df["protocol_type"].nunique(),
        "Services": df["service"].nunique(),
        "Flags": df["flag"].nunique(),
        "Attack Families": df["attack_family"].nunique(),
        "Suspicious %": f"{df['is_suspicious'].mean() * 100:.1f}%",
        "Duplicate Rows": 0,
        "Null Values": df.isnull().sum().sum(),
    }
    return pd.DataFrame(list(summary.items()), columns=["Metric", "Value"])


def get_family_breakdown(df):
    """Counts and percentages per attack family."""
    counts = df["attack_family"].value_counts()
    pcts = df["attack_family"].value_counts(normalize=True) * 100
    breakdown = pd.DataFrame({"Count": counts, "Percentage": pcts.round(2)})
    breakdown.index.name = "Attack Family"
    return breakdown.reset_index()


# ---------------------------------------------------------------------------
# Suspicion timeline (used by app and storyline)
# ---------------------------------------------------------------------------

def plot_suspicion_timeline(df, score_col="anomaly_score"):
    """Line chart of anomaly scores across pseudo-time (row index)."""
    if score_col not in df.columns:
        return None
    plot_df = df.reset_index()
    plot_df["pseudo_time"] = plot_df.index
    fig = px.line(
        plot_df, x="pseudo_time", y=score_col,
        title="Suspicion Score Over Pseudo-Time",
        labels={"pseudo_time": "Connection #", score_col: "Anomaly Score"},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red",
                  annotation_text="Threshold")
    return fig


def plot_episode_sizes(episodes):
    """Bar chart of episode sizes."""
    sizes = [ep["size"] for ep in episodes]
    ids = [f"Ep {ep['episode_id']}" for ep in episodes]
    fig = px.bar(
        x=ids, y=sizes,
        title="Episode Sizes",
        labels={"x": "Episode", "y": "# Suspicious Connections"},
        color=sizes, color_continuous_scale="Reds",
    )
    fig.update_layout(coloraxis_showscale=False)
    return fig
