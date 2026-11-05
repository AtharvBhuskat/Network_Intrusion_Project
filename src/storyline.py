"""
storyline.py
Episode builder + narrative generation for the Intrusion Storyline Builder.
Converts suspicious network records into grouped "incidents" with auto-written summaries.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_suspicion(df, anomaly_model, preprocessor, feature_cols):
    """Add anomaly_score and is_flagged columns to df."""
    df = df.copy()
    X = preprocessor.transform(df[feature_cols])
    raw = anomaly_model.decision_function(X)
    preds = anomaly_model.predict(X)

    df["anomaly_score"] = -raw  # higher = more suspicious
    df["is_flagged"] = (preds == -1).astype(int)
    return df


def predict_families(df, classifier, preprocessor, feature_cols):
    """Add predicted_family and family confidence columns."""
    df = df.copy()
    X = preprocessor.transform(df[feature_cols])
    df["predicted_family"] = classifier.predict(X)
    proba = classifier.predict_proba(X)
    df["family_confidence"] = proba.max(axis=1)
    return df


# ---------------------------------------------------------------------------
# Episode grouping
# ---------------------------------------------------------------------------

def build_episodes(df, gap_threshold=5):
    """
    Group consecutive suspicious (flagged) rows into episodes.
    A new episode starts when the row gap exceeds `gap_threshold`.
    Uses row index as pseudo-time since KDD has no timestamps.
    """
    flagged = df[df["is_flagged"] == 1].copy()
    if len(flagged) == 0:
        return []

    flagged = flagged.sort_index()
    flagged["pseudo_time"] = range(len(flagged))

    episodes = []
    ep_id = 1
    current_ep = [flagged.index[0]]

    for i in range(1, len(flagged)):
        curr_idx = flagged.index[i]
        prev_idx = flagged.index[i - 1]
        if curr_idx - prev_idx <= gap_threshold:
            current_ep.append(curr_idx)
        else:
            episodes.append(_summarize_episode(df, current_ep, ep_id))
            ep_id += 1
            current_ep = [curr_idx]

    # Last episode
    if current_ep:
        episodes.append(_summarize_episode(df, current_ep, ep_id))

    return episodes


def _summarize_episode(df, indices, ep_id):
    """Build summary dict for a single episode."""
    ep_df = df.loc[indices]

    # Top deviating numeric features
    top_features = _get_top_deviating_features(ep_df, df)

    # Dominant predicted family
    if "predicted_family" in ep_df.columns:
        dominant_family = ep_df["predicted_family"].mode().iloc[0]
        family_counts = ep_df["predicted_family"].value_counts().to_dict()
    else:
        dominant_family = "unknown"
        family_counts = {}

    # Average confidence
    avg_conf = ep_df["family_confidence"].mean() if "family_confidence" in ep_df.columns else 0.0

    return {
        "episode_id": ep_id,
        "start_idx": indices[0],
        "end_idx": indices[-1],
        "size": len(indices),
        "dominant_family": dominant_family,
        "family_counts": family_counts,
        "avg_confidence": round(avg_conf, 3),
        "avg_anomaly_score": round(ep_df["anomaly_score"].mean(), 4),
        "top_protocol": ep_df["protocol_type"].mode().iloc[0] if "protocol_type" in ep_df.columns else "N/A",
        "top_service": ep_df["service"].mode().iloc[0] if "service" in ep_df.columns else "N/A",
        "top_flag": ep_df["flag"].mode().iloc[0] if "flag" in ep_df.columns else "N/A",
        "top_features": top_features,
        "indices": indices,
    }


def _get_top_deviating_features(ep_df, full_df, n=5):
    """Find features that deviate most from the overall mean in this episode."""
    numeric_cols = ep_df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {"anomaly_score", "is_flagged", "is_suspicious", "family_confidence"}
    numeric_cols = [c for c in numeric_cols if c not in exclude]

    if not numeric_cols:
        return []

    full_mean = full_df[numeric_cols].mean()
    full_std = full_df[numeric_cols].std().replace(0, 1)
    ep_mean = ep_df[numeric_cols].mean()

    z_scores = ((ep_mean - full_mean) / full_std).abs()
    top = z_scores.nlargest(n)
    return [{"feature": feat, "z_score": round(val, 2), "ep_mean": round(ep_mean[feat], 2)}
            for feat, val in top.items()]


# ---------------------------------------------------------------------------
# Narrative generation
# ---------------------------------------------------------------------------

SEVERITY_MAP = {
    (0, 0.3): "low",
    (0.3, 0.6): "moderate",
    (0.6, 0.85): "high",
    (0.85, 1.0): "critical",
}

FAMILY_DESCRIPTIONS = {
    "dos": "Denial-of-Service flooding, attempting to overwhelm network resources",
    "probe": "network scanning and reconnaissance, mapping hosts and open ports",
    "r2l": "Remote-to-Local intrusion, attempting unauthorized access from a remote machine",
    "u2r": "User-to-Root privilege escalation, trying to gain root access",
    "normal": "normal traffic patterns with no malicious indicators",
    "unknown": "unrecognized traffic patterns",
}


def _get_severity(confidence):
    for (lo, hi), label in SEVERITY_MAP.items():
        if lo <= confidence < hi:
            return label
    return "critical"


def generate_narrative(episode):
    """Generate a human-readable narrative for a single episode."""
    ep = episode
    severity = _get_severity(ep["avg_confidence"])
    family = ep["dominant_family"]
    desc = FAMILY_DESCRIPTIONS.get(family, "unknown activity")

    # Feature description
    feat_parts = []
    for f in ep["top_features"][:3]:
        feat_parts.append(f"{f['feature']} (z={f['z_score']})")
    feat_str = ", ".join(feat_parts) if feat_parts else "no standout features"

    narrative = (
        f"**Episode {ep['episode_id']}** | Severity: **{severity.upper()}** | "
        f"Confidence: {ep['avg_confidence']:.1%}\n\n"
        f"A cluster of **{ep['size']}** suspicious connections was detected "
        f"(connections #{ep['start_idx']} to #{ep['end_idx']}). "
        f"The dominant traffic pattern suggests **{family.upper()}** activity — {desc}.\n\n"
        f"**Protocol:** {ep['top_protocol'].upper()} | "
        f"**Service:** {ep['top_service']} | "
        f"**Flag:** {ep['top_flag']}\n\n"
        f"**Key deviating features:** {feat_str}\n\n"
        f"**Average anomaly score:** {ep['avg_anomaly_score']:.4f}"
    )

    return narrative


def generate_full_report(episodes):
    """Generate a combined incident report from all episodes."""
    if not episodes:
        return "No suspicious episodes detected. All traffic appears normal."

    total_suspicious = sum(ep["size"] for ep in episodes)
    families_seen = set(ep["dominant_family"] for ep in episodes)

    header = (
        f"## Intrusion Storyline Report\n\n"
        f"**Total Episodes:** {len(episodes)} | "
        f"**Total Suspicious Connections:** {total_suspicious} | "
        f"**Attack Families Detected:** {', '.join(f.upper() for f in families_seen)}\n\n"
        f"---\n\n"
    )

    body = ""
    for ep in episodes:
        body += generate_narrative(ep) + "\n\n---\n\n"

    # Summary
    severity_counts = {}
    for ep in episodes:
        sev = _get_severity(ep["avg_confidence"])
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    summary = (
        f"## Summary\n\n"
        f"| Severity | Count |\n"
        f"|----------|-------|\n"
    )
    for sev in ["critical", "high", "moderate", "low"]:
        if sev in severity_counts:
            summary += f"| {sev.upper()} | {severity_counts[sev]} |\n"

    return header + body + summary
