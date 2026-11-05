"""
app.py
Intrusion Storyline Builder — Streamlit Application
6 tabs: Overview, Data Explorer, Suspicion Scanner, Storyline Builder,
        Model Performance, Explainability
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from eda import (
    plot_attack_family_distribution,
    plot_protocol_distribution,
    plot_service_top_n,
    plot_flag_distribution,
    plot_correlation_heatmap,
    plot_boxplots_suspicious,
    plot_pca_2d,
    get_summary_stats,
    get_family_breakdown,
    plot_suspicion_timeline,
    plot_episode_sizes,
)
from storyline import (
    score_suspicion,
    predict_families,
    build_episodes,
    generate_narrative,
    generate_full_report,
)
from data_prep import CATEGORICAL_COLS, ATTACK_MAP

# ---------------------------------------------------------------------------
# Config and constants
# ---------------------------------------------------------------------------

MODELS_DIR = os.path.join(BASE_DIR, "models")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

FAMILY_COLORS = {
    "normal": "#2ecc71",
    "dos": "#e74c3c",
    "probe": "#f39c12",
    "r2l": "#9b59b6",
    "u2r": "#e67e22",
    "unknown": "#95a5a6",
}

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_resource
def load_models():
    """Load all pickled model artifacts."""
    preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.pkl"))
    anomaly_model = joblib.load(os.path.join(MODELS_DIR, "anomaly_model.pkl"))
    classifier = joblib.load(os.path.join(MODELS_DIR, "family_classifier.pkl"))
    label_mapper = joblib.load(os.path.join(MODELS_DIR, "label_mapper.pkl"))
    eval_results = joblib.load(os.path.join(MODELS_DIR, "eval_results.pkl"))
    return preprocessor, anomaly_model, classifier, label_mapper, eval_results


@st.cache_data
def load_data():
    """Load processed train and test CSVs."""
    train = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    return train, test


def get_feature_cols(df):
    """Return feature columns (exclude label/target columns)."""
    exclude = {"label", "attack_family", "is_suspicious",
               "anomaly_score", "is_flagged", "predicted_family", "family_confidence"}
    return [c for c in df.columns if c not in exclude]


# ---------------------------------------------------------------------------
# Demo sample records
# ---------------------------------------------------------------------------

def get_demo_samples(df):
    """Return a dict of demo sample records for quick testing."""
    demos = {}
    for family in ["normal", "dos", "probe", "r2l", "u2r"]:
        subset = df[df["attack_family"] == family]
        if len(subset) > 0:
            demos[f"{family.upper()} Example"] = subset.iloc[0]
    return demos


# ===================================================================
# PAGE CONFIG
# ===================================================================

st.set_page_config(
    page_title="Intrusion Storyline Builder",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===================================================================
# LOAD DATA AND MODELS
# ===================================================================

preprocessor, anomaly_model, classifier, label_mapper, eval_results = load_models()
train_df, test_df = load_data()
full_df = pd.concat([train_df, test_df], ignore_index=True)
feature_cols = get_feature_cols(train_df)

# ===================================================================
# SIDEBAR
# ===================================================================

st.sidebar.title("🛡️ Intrusion Storyline Builder")
st.sidebar.markdown("---")

# Anomaly threshold slider
anomaly_threshold = st.sidebar.slider(
    "Anomaly Threshold", min_value=-0.5, max_value=0.5, value=0.0, step=0.01,
    help="Higher = stricter (fewer alerts). Lower = more sensitive."
)

# Alert mode toggle
alert_mode = st.sidebar.radio(
    "Alert Mode", ["Relaxed", "Strict"],
    help="Strict mode raises threshold by 0.1"
)
if alert_mode == "Strict":
    anomaly_threshold += 0.1

# Dataset filter
st.sidebar.markdown("### Data Filters")
protocol_filter = st.sidebar.multiselect(
    "Protocol", options=sorted(full_df["protocol_type"].unique()),
    default=sorted(full_df["protocol_type"].unique()),
)
service_filter = st.sidebar.multiselect(
    "Service (top 20)", options=sorted(full_df["service"].value_counts().head(20).index),
    default=sorted(full_df["service"].value_counts().head(5).index),
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** KDD Cup 1999 (10%)")
st.sidebar.markdown("**Models:** Isolation Forest + Logistic Regression")

# ===================================================================
# TABS
# ===================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Overview",
    "🔍 Data Explorer",
    "⚠️ Suspicion Scanner",
    "📖 Storyline Builder",
    "📊 Model Performance",
    "🧠 Explainability",
])


# ===================================================================
# TAB 1: OVERVIEW
# ===================================================================

with tab1:
    st.header("Project Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(full_df):,}")
    col2.metric("Features", len(feature_cols))
    col3.metric("Attack Families", full_df["attack_family"].nunique())
    col4.metric("Suspicious %", f"{full_df['is_suspicious'].mean() * 100:.1f}%")

    st.markdown("---")

    st.subheader("Problem Statement")
    st.markdown("""
    This project builds an **Intrusion Detection & Storyline System** that:

    1. **Detects suspicious network connections** using Isolation Forest (anomaly detection)
    2. **Classifies attack families** (DoS, Probe, R2L, U2R) using Logistic Regression
    3. **Generates incident storylines** — grouping suspicious events into episodes
       with auto-written narratives

    The dataset is the classic **KDD Cup 1999** network intrusion dataset with 41 features
    describing network connection attributes.
    """)

    st.subheader("Model Flow")
    flow_col1, flow_col2, flow_col3 = st.columns(3)
    with flow_col1:
        st.markdown("""
        **Stage A: Anomaly Detection**
        - Isolation Forest
        - Scores each connection
        - Flags suspicious traffic
        """)
    with flow_col2:
        st.markdown("""
        **Stage B: Family Classification**
        - Logistic Regression
        - 5-class prediction
        - Confidence scoring
        """)
    with flow_col3:
        st.markdown("""
        **Stage C: Storyline Engine**
        - Episode grouping
        - Feature deviation analysis
        - Auto-generated narratives
        """)

    st.subheader("Dataset Summary")
    st.dataframe(get_summary_stats(full_df), use_container_width=True, hide_index=True)

    st.subheader("Attack Family Breakdown")
    st.dataframe(get_family_breakdown(full_df), use_container_width=True, hide_index=True)


# ===================================================================
# TAB 2: DATA EXPLORER
# ===================================================================

with tab2:
    st.header("Data Explorer")

    # Apply filters
    filtered_df = full_df[
        (full_df["protocol_type"].isin(protocol_filter)) &
        (full_df["service"].isin(service_filter))
    ]

    st.info(f"Showing **{len(filtered_df):,}** records after filtering "
            f"(from {len(full_df):,} total)")

    # Data sample
    with st.expander("📄 Dataset Sample (first 100 rows)", expanded=True):
        st.dataframe(filtered_df.head(100), use_container_width=True, height=300)

    # Charts
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(plot_attack_family_distribution(filtered_df),
                        use_container_width=True)
    with chart_col2:
        st.plotly_chart(plot_protocol_distribution(filtered_df),
                        use_container_width=True)

    chart_col3, chart_col4 = st.columns(2)
    with chart_col3:
        st.plotly_chart(plot_service_top_n(filtered_df, n=15),
                        use_container_width=True)
    with chart_col4:
        st.plotly_chart(plot_flag_distribution(filtered_df),
                        use_container_width=True)

    with st.expander("🔗 Correlation Heatmap"):
        st.plotly_chart(plot_correlation_heatmap(filtered_df),
                        use_container_width=True)

    with st.expander("📦 Normal vs Suspicious — Box Plots"):
        st.plotly_chart(plot_boxplots_suspicious(filtered_df),
                        use_container_width=True)

    with st.expander("🎯 PCA 2D Visualization"):
        st.plotly_chart(plot_pca_2d(filtered_df), use_container_width=True)


# ===================================================================
# TAB 3: SUSPICION SCANNER
# ===================================================================

with tab3:
    st.header("Suspicion Scanner")
    st.markdown("Analyze individual connections or upload a batch CSV.")

    scan_mode = st.radio("Input Mode", ["Demo Samples", "Manual Entry", "CSV Upload"],
                         horizontal=True)

    if scan_mode == "Demo Samples":
        demos = get_demo_samples(full_df)
        selected_demo = st.selectbox("Choose a sample", list(demos.keys()))
        sample_row = demos[selected_demo]
        input_df = pd.DataFrame([sample_row[feature_cols]])
        st.markdown(f"**Selected:** {selected_demo}")
        with st.expander("View raw values"):
            st.dataframe(input_df, use_container_width=True)

    elif scan_mode == "Manual Entry":
        st.markdown("Enter values for key features (others default to 0):")
        manual_col1, manual_col2, manual_col3 = st.columns(3)
        with manual_col1:
            m_protocol = st.selectbox("protocol_type",
                                      sorted(full_df["protocol_type"].unique()))
            m_duration = st.number_input("duration", value=0, min_value=0)
            m_src_bytes = st.number_input("src_bytes", value=0, min_value=0)
        with manual_col2:
            m_service = st.selectbox("service",
                                     sorted(full_df["service"].unique()))
            m_dst_bytes = st.number_input("dst_bytes", value=0, min_value=0)
            m_count = st.number_input("count", value=1, min_value=0)
        with manual_col3:
            m_flag = st.selectbox("flag", sorted(full_df["flag"].unique()))
            m_serror_rate = st.slider("serror_rate", 0.0, 1.0, 0.0)
            m_srv_count = st.number_input("srv_count", value=1, min_value=0)

        # Build a row with defaults from the dataset median
        row_data = {}
        for col in feature_cols:
            if col == "protocol_type":
                row_data[col] = m_protocol
            elif col == "service":
                row_data[col] = m_service
            elif col == "flag":
                row_data[col] = m_flag
            elif col == "duration":
                row_data[col] = m_duration
            elif col == "src_bytes":
                row_data[col] = m_src_bytes
            elif col == "dst_bytes":
                row_data[col] = m_dst_bytes
            elif col == "count":
                row_data[col] = m_count
            elif col == "srv_count":
                row_data[col] = m_srv_count
            elif col == "serror_rate":
                row_data[col] = m_serror_rate
            else:
                row_data[col] = 0
        input_df = pd.DataFrame([row_data])

    else:  # CSV Upload
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            input_df = pd.read_csv(uploaded)
            # Ensure required columns exist
            missing = set(feature_cols) - set(input_df.columns)
            if missing:
                st.error(f"Missing columns: {missing}")
                st.stop()
            input_df = input_df[feature_cols]
            st.success(f"Uploaded {len(input_df)} rows")
        else:
            st.info("Please upload a CSV file with the same columns as the dataset.")
            st.stop()

    # --- RUN PREDICTION ---
    if st.button("🔍 Scan for Intrusions", type="primary", use_container_width=True):
        with st.spinner("Analyzing connections..."):
            X_proc = preprocessor.transform(input_df)

            # Anomaly scoring
            raw_scores = anomaly_model.decision_function(X_proc)
            anomaly_preds = anomaly_model.predict(X_proc)
            suspicion_scores = -raw_scores

            # Family classification
            family_preds = classifier.predict(X_proc)
            family_proba = classifier.predict_proba(X_proc)
            family_conf = family_proba.max(axis=1)

        results_df = input_df.copy()
        results_df["anomaly_score"] = suspicion_scores
        results_df["is_flagged"] = (anomaly_preds == -1).astype(int)
        results_df["predicted_family"] = family_preds
        results_df["family_confidence"] = family_conf

        # Display results
        n_flagged = results_df["is_flagged"].sum()
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Records Scanned", len(results_df))
        res_col2.metric("Flagged Suspicious", n_flagged)
        res_col3.metric("Flag Rate", f"{n_flagged / len(results_df) * 100:.1f}%")

        if len(results_df) == 1:
            # Single record display
            row = results_df.iloc[0]
            st.markdown("---")
            if row["is_flagged"]:
                st.error(f"⚠️ **SUSPICIOUS** — Predicted: **{row['predicted_family'].upper()}** "
                         f"(Confidence: {row['family_confidence']:.1%})")
            else:
                st.success(f"✅ **NORMAL** — Predicted: **{row['predicted_family'].upper()}** "
                           f"(Confidence: {row['family_confidence']:.1%})")

            st.markdown(f"**Anomaly Score:** {row['anomaly_score']:.4f}")

            # Confidence breakdown
            st.markdown("**Family Probabilities:**")
            prob_df = pd.DataFrame({
                "Family": classifier.classes_,
                "Probability": family_proba[0],
            }).sort_values("Probability", ascending=False)
            fig = px.bar(prob_df, x="Family", y="Probability",
                         color="Family", color_discrete_map=FAMILY_COLORS)
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)

        else:
            # Batch display
            st.markdown("---")
            st.subheader("Scan Results")
            display_cols = ["predicted_family", "family_confidence",
                            "anomaly_score", "is_flagged"]
            st.dataframe(
                results_df[display_cols + ["protocol_type", "service", "flag"]],
                use_container_width=True, height=400,
            )

            # Download button
            csv_out = results_df.to_csv(index=False)
            st.download_button("📥 Download Results CSV", csv_out,
                               "scan_results.csv", "text/csv")


# ===================================================================
# TAB 4: STORYLINE BUILDER
# ===================================================================

with tab4:
    st.header("Storyline Builder")
    st.markdown("Generate incident storylines from suspicious network activity.")

    # Data source
    story_source = st.radio(
        "Data Source",
        ["Test Dataset Sample", "Upload CSV"],
        horizontal=True,
    )

    if story_source == "Test Dataset Sample":
        sample_size = st.slider("Sample size", 100, 5000, 500, step=100)
        story_df = test_df.sample(n=min(sample_size, len(test_df)), random_state=42).copy()
    else:
        uploaded_story = st.file_uploader("Upload CSV for storyline", type=["csv"],
                                          key="storyline_upload")
        if uploaded_story is not None:
            story_df = pd.read_csv(uploaded_story)
            missing = set(feature_cols) - set(story_df.columns)
            if missing:
                st.error(f"Missing columns: {missing}")
                st.stop()
        else:
            st.info("Upload a CSV or use the test dataset sample.")
            st.stop()

    # Episode gap threshold
    gap_threshold = st.slider("Episode Gap Threshold",
                              min_value=1, max_value=50, value=5,
                              help="Max row gap between connections in the same episode")

    if st.button("📖 Build Storyline", type="primary", use_container_width=True):
        with st.spinner("Scoring and building episodes..."):
            # Score and predict
            story_df = score_suspicion(story_df, anomaly_model, preprocessor, feature_cols)
            story_df = predict_families(story_df, classifier, preprocessor, feature_cols)

            # Build episodes
            episodes = build_episodes(story_df, gap_threshold=gap_threshold)

        # Metrics
        n_sus = story_df["is_flagged"].sum()
        st_col1, st_col2, st_col3, st_col4 = st.columns(4)
        st_col1.metric("Records Analyzed", len(story_df))
        st_col2.metric("Suspicious", n_sus)
        st_col3.metric("Episodes Found", len(episodes))
        st_col4.metric("Alert Rate", f"{n_sus / len(story_df) * 100:.1f}%")

        st.markdown("---")

        # Suspicion timeline
        st.subheader("Suspicion Timeline")
        timeline_fig = plot_suspicion_timeline(story_df)
        if timeline_fig:
            st.plotly_chart(timeline_fig, use_container_width=True)

        # Episode sizes
        if episodes:
            st.subheader("Episode Overview")
            st.plotly_chart(plot_episode_sizes(episodes), use_container_width=True)

            # Episode summary table
            ep_table = pd.DataFrame([
                {
                    "Episode": ep["episode_id"],
                    "Size": ep["size"],
                    "Family": ep["dominant_family"].upper(),
                    "Confidence": f"{ep['avg_confidence']:.1%}",
                    "Protocol": ep["top_protocol"],
                    "Service": ep["top_service"],
                    "Anomaly Score": f"{ep['avg_anomaly_score']:.4f}",
                }
                for ep in episodes
            ])
            st.dataframe(ep_table, use_container_width=True, hide_index=True)

            # Individual episode narratives
            st.markdown("---")
            st.subheader("Episode Narratives")

            for ep in episodes:
                with st.expander(
                    f"Episode {ep['episode_id']} — {ep['dominant_family'].upper()} "
                    f"({ep['size']} connections)", expanded=(ep == episodes[0])
                ):
                    st.markdown(generate_narrative(ep))

            # Full report
            st.markdown("---")
            st.subheader("Full Incident Report")
            report = generate_full_report(episodes)
            st.markdown(report)

            # Download report
            st.download_button("📥 Download Report", report,
                               "incident_report.md", "text/markdown")
        else:
            st.info("No suspicious episodes found with current settings. "
                    "Try lowering the anomaly threshold in the sidebar.")


# ===================================================================
# TAB 5: MODEL PERFORMANCE
# ===================================================================

with tab5:
    st.header("Model Performance")

    perf_tab1, perf_tab2 = st.tabs(["Anomaly Detector", "Family Classifier"])

    with perf_tab1:
        st.subheader("Isolation Forest — Anomaly Detection")

        anomaly_eval = eval_results["anomaly"]
        ae_col1, ae_col2, ae_col3 = st.columns(3)
        ae_col1.metric("Precision", f"{anomaly_eval['precision']:.4f}")
        ae_col2.metric("Recall", f"{anomaly_eval['recall']:.4f}")
        ae_col3.metric("ROC-AUC", f"{anomaly_eval['roc_auc']:.4f}")

        # Score distribution
        st.subheader("Anomaly Score Distribution")
        scores = anomaly_eval["scores"]
        score_df = pd.DataFrame({
            "Anomaly Score": -scores,
            "True Label": ["Suspicious" if y == 1 else "Normal"
                           for y in eval_results["anomaly"]["predictions"]],
        })

        # Use actual test labels for coloring
        test_labels = test_df["is_suspicious"].values
        score_df["Actual"] = ["Suspicious" if y == 1 else "Normal" for y in test_labels]

        fig = px.histogram(
            score_df, x="Anomaly Score", color="Actual",
            nbins=80, barmode="overlay", opacity=0.7,
            title="Anomaly Score Distribution (Test Set)",
            color_discrete_map={"Normal": "#2ecc71", "Suspicious": "#e74c3c"},
        )
        fig.add_vline(x=0, line_dash="dash", line_color="black",
                      annotation_text="Default Threshold")
        st.plotly_chart(fig, use_container_width=True)

        # ROC Curve
        st.subheader("ROC Curve")
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(test_labels, -scores)
        roc_fig = px.area(
            x=fpr, y=tpr,
            title=f"ROC Curve (AUC = {anomaly_eval['roc_auc']:.4f})",
            labels={"x": "False Positive Rate", "y": "True Positive Rate"},
        )
        roc_fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                          line=dict(dash="dash", color="gray"))
        st.plotly_chart(roc_fig, use_container_width=True)

    with perf_tab2:
        st.subheader("Logistic Regression — Family Classifier")

        cls_eval = eval_results["classifier"]
        ce_col1, ce_col2 = st.columns(2)
        ce_col1.metric("Macro F1", f"{cls_eval['macro_f1']:.4f}")
        ce_col2.metric("Accuracy",
                       f"{cls_eval['report']['accuracy']:.4f}")

        # Classification report
        st.subheader("Classification Report")
        report_df = pd.DataFrame(cls_eval["report"]).T
        report_df = report_df.drop(["accuracy", "macro avg", "weighted avg"],
                                    errors="ignore")
        report_df = report_df.round(3)
        st.dataframe(report_df, use_container_width=True)

        # Confusion matrix
        st.subheader("Confusion Matrix")
        cm = cls_eval["confusion_matrix"]
        class_names = cls_eval["class_names"]
        cm_fig = px.imshow(
            cm, text_auto=True,
            x=class_names, y=class_names,
            labels={"x": "Predicted", "y": "Actual", "color": "Count"},
            title="Confusion Matrix",
            color_continuous_scale="Blues",
            aspect="auto",
        )
        st.plotly_chart(cm_fig, use_container_width=True)

        # Per-class recall bar
        st.subheader("Per-Class Recall")
        recalls = {cls: cls_eval["report"][cls]["recall"]
                   for cls in class_names if cls in cls_eval["report"]}
        recall_fig = px.bar(
            x=list(recalls.keys()), y=list(recalls.values()),
            title="Recall by Attack Family",
            labels={"x": "Family", "y": "Recall"},
            color=list(recalls.keys()),
            color_discrete_map=FAMILY_COLORS,
        )
        recall_fig.update_layout(showlegend=False)
        st.plotly_chart(recall_fig, use_container_width=True)

        # 10 sample predictions
        st.subheader("Sample Predictions (10 records)")
        sample_test = test_df.sample(10, random_state=42)
        X_sample = preprocessor.transform(sample_test[feature_cols])
        sample_preds = classifier.predict(X_sample)
        sample_proba = classifier.predict_proba(X_sample)

        pred_table = pd.DataFrame({
            "Actual": sample_test["attack_family"].values,
            "Predicted": sample_preds,
            "Confidence": sample_proba.max(axis=1).round(3),
            "Correct": sample_test["attack_family"].values == sample_preds,
            "Protocol": sample_test["protocol_type"].values,
            "Service": sample_test["service"].values,
        })
        st.dataframe(pred_table, use_container_width=True, hide_index=True)


# ===================================================================
# TAB 6: EXPLAINABILITY
# ===================================================================

with tab6:
    st.header("Explainability")

    # Feature importance from Logistic Regression coefficients
    st.subheader("Top Features by Classifier Coefficients")

    # Get feature names from preprocessor
    try:
        num_features = preprocessor.transformers_[0][2]  # numeric col names
        cat_encoder = preprocessor.transformers_[1][1]  # OneHotEncoder
        cat_features = cat_encoder.get_feature_names_out(CATEGORICAL_COLS).tolist()
        all_feature_names = list(num_features) + cat_features
    except Exception:
        all_feature_names = [f"feature_{i}" for i in range(classifier.coef_.shape[1])]

    # Average absolute coefficient across classes
    avg_importance = np.abs(classifier.coef_).mean(axis=0)
    importance_df = pd.DataFrame({
        "Feature": all_feature_names[:len(avg_importance)],
        "Importance": avg_importance,
    }).sort_values("Importance", ascending=False).head(20)

    fig_imp = px.bar(
        importance_df, x="Importance", y="Feature", orientation="h",
        title="Top 20 Features (Avg Absolute Coefficient)",
        color="Importance", color_continuous_scale="Reds",
    )
    fig_imp.update_layout(yaxis={"categoryorder": "total ascending"},
                          coloraxis_showscale=False, height=600)
    st.plotly_chart(fig_imp, use_container_width=True)

    # Per-class coefficients
    st.subheader("Per-Class Feature Importance")
    selected_class = st.selectbox("Select attack family",
                                  classifier.classes_.tolist())
    class_idx = list(classifier.classes_).index(selected_class)
    class_coefs = classifier.coef_[class_idx]

    class_imp_df = pd.DataFrame({
        "Feature": all_feature_names[:len(class_coefs)],
        "Coefficient": class_coefs,
    })
    # Top positive and negative
    top_pos = class_imp_df.nlargest(10, "Coefficient")
    top_neg = class_imp_df.nsmallest(10, "Coefficient")
    combined = pd.concat([top_pos, top_neg]).sort_values("Coefficient")

    fig_class = px.bar(
        combined, x="Coefficient", y="Feature", orientation="h",
        title=f"Top Features for {selected_class.upper()}",
        color="Coefficient", color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
    )
    fig_class.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
    st.plotly_chart(fig_class, use_container_width=True)

    # Normal vs Suspicious comparison
    st.subheader("Normal vs Suspicious — Feature Comparison")
    compare_features = st.multiselect(
        "Select features to compare",
        options=[c for c in feature_cols if c not in CATEGORICAL_COLS],
        default=["src_bytes", "dst_bytes", "count", "serror_rate",
                 "srv_count", "dst_host_srv_count"],
    )

    if compare_features:
        normal_means = full_df[full_df["is_suspicious"] == 0][compare_features].mean()
        suspicious_means = full_df[full_df["is_suspicious"] == 1][compare_features].mean()

        compare_df = pd.DataFrame({
            "Feature": compare_features,
            "Normal (mean)": normal_means.values,
            "Suspicious (mean)": suspicious_means.values,
        })

        fig_compare = go.Figure()
        fig_compare.add_trace(go.Bar(
            name="Normal", x=compare_df["Feature"],
            y=compare_df["Normal (mean)"], marker_color="#2ecc71",
        ))
        fig_compare.add_trace(go.Bar(
            name="Suspicious", x=compare_df["Feature"],
            y=compare_df["Suspicious (mean)"], marker_color="#e74c3c",
        ))
        fig_compare.update_layout(
            barmode="group", title="Mean Feature Values: Normal vs Suspicious",
            height=400,
        )
        st.plotly_chart(fig_compare, use_container_width=True)

        st.dataframe(compare_df, use_container_width=True, hide_index=True)
