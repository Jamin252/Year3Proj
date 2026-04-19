import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_clip_id(clip_id: str) -> Dict[str, Optional[float]]:
    """Parse clip metadata from clip_id pattern mix_xxxxxx_ovr_maxspk_snr_noise."""
    parts = clip_id.split("_")
    # Expected pattern: mix, id, overlap, max_speakers, snr, noise_type
    overlap = None
    snr = None
    max_speakers = None
    noise_type = None

    if len(parts) >= 6:
        overlap_raw = parts[2]
        max_spk_raw = parts[3]
        snr_raw = parts[4]
        noise_type = parts[5]

        overlap = float(overlap_raw) if overlap_raw not in ("None", "") else None
        max_speakers = int(max_spk_raw) if max_spk_raw not in ("None", "") else None
        snr = float(snr_raw) if snr_raw not in ("None", "") else None

    return {
        "clip_id": clip_id,
        "overlap_ratio": overlap,
        "max_speakers": max_speakers,
        "snr_db": snr,
        "noise_type": noise_type,
    }


def _extract_error_value(result: Dict) -> Optional[float]:
    metrics = result.get("metrics", {}) or {}
    if "cpwer" in metrics and metrics["cpwer"] is not None:
        return float(metrics["cpwer"])
    if "wer" in metrics and metrics["wer"] is not None:
        return float(metrics["wer"])
    return None


def _extract_metric_name(result: Dict) -> str:
    metrics = result.get("metrics", {}) or {}
    if "cpwer" in metrics and metrics["cpwer"] is not None:
        return "cpwer"
    if "wer" in metrics and metrics["wer"] is not None:
        return "wer"
    return "unknown"


def load_wer_json(json_path: str) -> Dict:
    with open(json_path, "r") as f:
        return json.load(f)


def build_analysis_df(wer_json: Dict, include_failed: bool = True) -> pd.DataFrame:
    """Build an analysis dataframe from WER result json."""
    rows: List[Dict] = []
    for result in wer_json.get("results", []):
        parsed = parse_clip_id(result.get("clip_id", ""))
        row = {
            "clip_id": result.get("clip_id"),
            "status": result.get("status"),
            "model_name": result.get("model_name"),
            "metric_type": _extract_metric_name(result),
            "error_value": _extract_error_value(result),
            "ref_segments": result.get("ref_segments"),
            "hyp_segments": result.get("hyp_segments"),
            "is_segmented": result.get("is_segmented"),
            "error_message": result.get("error"),
            **parsed,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if not include_failed:
        df = df[df["status"] == "success"].copy()
    return df


def get_interpretation_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Return interpretation-ready tables for reporting."""
    success_df = df[df["status"] == "success"].copy()
    ordered_snr = _ordered_snr_levels(df["snr_db"])
    ordered_snr_labels = [_snr_label(v) for v in ordered_snr]

    # Success coverage by condition
    coverage = (
        df.groupby(["snr_db", "overlap_ratio"], dropna=False)["status"]
        .agg(total="count", successful=lambda s: (s == "success").sum())
        .reset_index()
    )
    coverage["success_rate"] = coverage["successful"] / coverage["total"]
    coverage["snr_label"] = coverage["snr_db"].apply(_snr_label)
    coverage["snr_order"] = pd.Categorical(
        coverage["snr_label"], categories=ordered_snr_labels, ordered=True
    )
    coverage = coverage.sort_values(["snr_order", "overlap_ratio"]).drop(columns=["snr_order"])

    # Mean error by SNR
    snr_stats = (
        success_df.groupby("snr_db", dropna=False)["error_value"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    snr_stats["snr_label"] = snr_stats["snr_db"].apply(_snr_label)
    snr_stats["snr_order"] = pd.Categorical(
        snr_stats["snr_label"], categories=ordered_snr_labels, ordered=True
    )
    snr_stats = snr_stats.sort_values("snr_order").drop(columns=["snr_order"])

    # Mean error by overlap
    overlap_stats = (
        success_df.groupby("overlap_ratio", dropna=False)["error_value"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
        .sort_values("overlap_ratio")
    )

    # Cross table SNR x overlap
    success_cross = success_df.copy()
    success_cross["snr_label"] = success_cross["snr_db"].apply(_snr_label)

    cross_mean = success_cross.pivot_table(
        index="snr_label",
        columns="overlap_ratio",
        values="error_value",
        aggfunc="mean",
        dropna=False,
    )
    cross_mean = cross_mean.reindex(ordered_snr_labels)

    cross_count = success_cross.pivot_table(
        index="snr_label",
        columns="overlap_ratio",
        values="error_value",
        aggfunc="count",
        dropna=False,
    )
    cross_count = cross_count.reindex(ordered_snr_labels)

    return {
        "coverage": coverage,
        "snr_stats": snr_stats,
        "overlap_stats": overlap_stats,
        "cross_mean": cross_mean,
        "cross_count": cross_count,
    }


def _ordered_snr_levels(series: pd.Series) -> List[Optional[float]]:
    """Order SNR levels as None first, then descending dB (low noise to high noise)."""
    levels = list(pd.unique(series))
    has_none = any(pd.isna(v) for v in levels)
    numeric = sorted([float(v) for v in levels if not pd.isna(v)], reverse=True)
    return ([None] if has_none else []) + numeric


def _snr_label(value: Optional[float]) -> str:
    return "No noise" if pd.isna(value) else f"{float(value):g} dB"


def plot_snr_vs_error(success_df: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
    plot_df = success_df.copy()
    ordered_snr = _ordered_snr_levels(plot_df["snr_db"])
    plot_df["snr_label"] = plot_df["snr_db"].apply(
        lambda v: "No noise" if pd.isna(v) else f"{v:g}"
    )
    x_order = ["No noise" if v is None else f"{v:g}" for v in ordered_snr]

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=plot_df, x="snr_label", y="error_value", order=x_order, ax=ax)
    sns.stripplot(
        data=plot_df,
        x="snr_label",
        y="error_value",
        order=x_order,
        color="black",
        alpha=0.2,
        size=2,
        ax=ax,
    )
    ax.set_title("wav2vec2 Error by SNR")
    ax.set_xlabel("SNR (dB, No noise = clean)")
    ax.set_ylabel("Error (WER or cpWER)")
    return fig, ax


def plot_overlap_vs_error_by_snr(success_df: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
    grouped = (
        success_df.groupby(["snr_db", "overlap_ratio"], dropna=False)["error_value"]
        .mean()
        .reset_index()
        .sort_values(["snr_db", "overlap_ratio"])
    )

    grouped["snr_label"] = grouped["snr_db"].apply(
        lambda v: "No noise" if pd.isna(v) else f"{v:g} dB"
    )

    ordered_snr = _ordered_snr_levels(grouped["snr_db"])
    hue_order = [
        "No noise" if v is None else f"{v:g} dB" for v in ordered_snr
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=grouped,
        x="overlap_ratio",
        y="error_value",
        hue="snr_label",
        hue_order=hue_order,
        marker="o",
        ax=ax,
    )

    overall = (
        success_df.groupby("overlap_ratio", dropna=False)["error_value"]
        .mean()
        .reset_index()
        .sort_values("overlap_ratio")
    )
    sns.lineplot(
        data=overall,
        x="overlap_ratio",
        y="error_value",
        color="black",
        marker="o",
        linestyle="--",
        linewidth=2.5,
        label="Overall",
        ax=ax,
    )

    ax.set_title("Overlap vs Error (per SNR + overall)")
    ax.set_xlabel("Overlap Ratio")
    ax.set_ylabel("Mean Error")
    ax.legend(title="Series", bbox_to_anchor=(1.02, 1), loc="upper left")
    return fig, ax


def plot_snr_vs_error_by_overlap(success_df: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
    grouped = (
        success_df.groupby(["overlap_ratio", "snr_db"], dropna=False)["error_value"]
        .mean()
        .reset_index()
    )

    ordered_snr = _ordered_snr_levels(grouped["snr_db"])
    snr_label_order = ["No noise" if v is None else f"{v:g}" for v in ordered_snr]
    grouped["snr_label"] = grouped["snr_db"].apply(
        lambda v: "No noise" if pd.isna(v) else f"{v:g}"
    )
    grouped["snr_label"] = pd.Categorical(
        grouped["snr_label"], categories=snr_label_order, ordered=True
    )
    grouped = grouped.sort_values(["overlap_ratio", "snr_label"])
    grouped["overlap_label"] = grouped["overlap_ratio"].apply(lambda v: f"OVR {v:g}")

    overlap_levels = sorted(grouped["overlap_ratio"].dropna().unique().tolist())
    hue_order = [f"OVR {v:g}" for v in overlap_levels]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=grouped,
        x="snr_label",
        y="error_value",
        hue="overlap_label",
        hue_order=hue_order,
        marker="o",
        ax=ax,
    )

    overall = (
        success_df.groupby("snr_db", dropna=False)["error_value"]
        .mean()
        .reset_index()
    )
    overall["snr_label"] = overall["snr_db"].apply(
        lambda v: "No noise" if pd.isna(v) else f"{v:g}"
    )
    overall["snr_label"] = pd.Categorical(
        overall["snr_label"], categories=snr_label_order, ordered=True
    )
    overall = overall.sort_values("snr_label")

    sns.lineplot(
        data=overall,
        x="snr_label",
        y="error_value",
        color="black",
        marker="o",
        linestyle="--",
        linewidth=2.5,
        label="Overall",
        ax=ax,
    )

    ax.set_title("SNR vs Error (per overlap + overall)")
    ax.set_xlabel("SNR (dB, No noise = clean)")
    ax.set_ylabel("Mean Error")
    ax.legend(title="Series", bbox_to_anchor=(1.02, 1), loc="upper left")
    return fig, ax


def plot_overall_overlap_vs_error(success_df: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
    overall = (
        success_df.groupby("overlap_ratio", dropna=False)["error_value"]
        .mean()
        .reset_index()
        .sort_values("overlap_ratio")
    )

    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.lineplot(
        data=overall,
        x="overlap_ratio",
        y="error_value",
        color="black",
        marker="o",
        linewidth=2.5,
        ax=ax,
    )
    ax.set_title("Overall Error vs Overlap Ratio")
    ax.set_xlabel("Overlap Ratio")
    ax.set_ylabel("Mean Error")
    return fig, ax


def plot_cross_heatmaps(success_df: pd.DataFrame) -> Tuple[plt.Figure, List[plt.Axes]]:
    ordered_snr = _ordered_snr_levels(success_df["snr_db"])
    ordered_snr_labels = [_snr_label(v) for v in ordered_snr]
    plot_df = success_df.copy()
    plot_df["snr_label"] = plot_df["snr_db"].apply(_snr_label)

    mean_pivot = plot_df.pivot_table(
        index="snr_label",
        columns="overlap_ratio",
        values="error_value",
        aggfunc="mean",
        dropna=False,
    )
    mean_pivot = mean_pivot.reindex(ordered_snr_labels)

    count_pivot = plot_df.pivot_table(
        index="snr_label",
        columns="overlap_ratio",
        values="error_value",
        aggfunc="count",
        dropna=False,
    )
    count_pivot = count_pivot.reindex(ordered_snr_labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    sns.heatmap(mean_pivot, annot=True, fmt=".3f", cmap="mako", ax=axes[0])
    axes[0].set_title("Mean Error (SNR x Overlap)")
    axes[0].set_xlabel("Overlap Ratio")
    axes[0].set_ylabel("SNR (dB)")

    sns.heatmap(count_pivot, annot=True, fmt=".0f", cmap="crest", ax=axes[1])
    axes[1].set_title("Sample Count (SNR x Overlap)")
    axes[1].set_xlabel("Overlap Ratio")
    axes[1].set_ylabel("SNR (dB)")

    return fig, [axes[0], axes[1]]


def plot_noise_type_effect(success_df: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
    ranked = (
        success_df.groupby("noise_type")["error_value"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=ranked, x="noise_type", y="error_value", ax=ax)
    ax.set_title("Mean Error by Noise Type")
    ax.set_xlabel("Noise Type")
    ax.set_ylabel("Mean Error")
    return fig, ax


def plot_error_cdf_by_snr(success_df: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
    plot_df = success_df.dropna(subset=["error_value"]).copy()
    ordered_snr = _ordered_snr_levels(plot_df["snr_db"])
    ordered_labels = [_snr_label(v) for v in ordered_snr]
    plot_df["snr_label"] = plot_df["snr_db"].apply(_snr_label)

    fig, ax = plt.subplots(figsize=(10, 6))
    for label in ordered_labels:
        vals = np.sort(plot_df.loc[plot_df["snr_label"] == label, "error_value"].to_numpy())
        if vals.size == 0:
            continue
        cdf = np.arange(1, vals.size + 1) / vals.size
        ax.step(vals, cdf, where="post", label=label)

    all_vals = np.sort(plot_df["error_value"].to_numpy())
    if all_vals.size > 0:
        all_cdf = np.arange(1, all_vals.size + 1) / all_vals.size
        ax.step(all_vals, all_cdf, where="post", color="black", linestyle="--", linewidth=2.2, label="Overall")

    ax.set_title("Error CDF by SNR")
    ax.set_xlabel("Error")
    ax.set_ylabel("Cumulative fraction <= x")
    ax.legend(title="Series", bbox_to_anchor=(1.02, 1), loc="upper left")
    return fig, ax


def plot_degradation_delta_vs_snr(success_df: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
    grouped = (
        success_df.groupby(["overlap_ratio", "snr_db"], dropna=False)["error_value"]
        .mean()
        .reset_index()
    )
    grouped["snr_label"] = grouped["snr_db"].apply(_snr_label)

    baseline = grouped[grouped["snr_label"] == "No noise"][["overlap_ratio", "error_value"]].rename(
        columns={"error_value": "baseline_error"}
    )
    delta_df = grouped.merge(baseline, on="overlap_ratio", how="left")
    delta_df["delta_error"] = delta_df["error_value"] - delta_df["baseline_error"]

    ordered_snr = _ordered_snr_levels(delta_df["snr_db"])
    ordered_labels = [_snr_label(v) for v in ordered_snr]
    delta_df["snr_label"] = pd.Categorical(delta_df["snr_label"], categories=ordered_labels, ordered=True)
    delta_df = delta_df.sort_values(["overlap_ratio", "snr_label"])
    delta_df["overlap_label"] = delta_df["overlap_ratio"].apply(lambda v: f"OVR {v:g}")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=delta_df,
        x="snr_label",
        y="delta_error",
        hue="overlap_label",
        marker="o",
        ax=ax,
    )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.set_title("Degradation vs No-noise Baseline")
    ax.set_xlabel("SNR (dB, No noise = clean)")
    ax.set_ylabel("Delta Error")
    ax.legend(title="Overlap", bbox_to_anchor=(1.02, 1), loc="upper left")
    return fig, ax


def plot_error_interaction_contours(success_df: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
    plot_df = success_df.copy()
    ordered_snr = _ordered_snr_levels(plot_df["snr_db"])
    ordered_labels = [_snr_label(v) for v in ordered_snr]
    plot_df["snr_label"] = plot_df["snr_db"].apply(_snr_label)

    z_df = plot_df.pivot_table(
        index="snr_label",
        columns="overlap_ratio",
        values="error_value",
        aggfunc="mean",
        dropna=False,
    ).reindex(ordered_labels)

    x_vals = np.array(sorted([v for v in z_df.columns.tolist() if pd.notna(v)]), dtype=float)
    z_df = z_df[x_vals]
    y_vals = np.arange(len(z_df.index), dtype=float)
    z_vals = z_df.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    contour = ax.contourf(x_vals, y_vals, z_vals, levels=12, cmap="mako")
    ax.contour(x_vals, y_vals, z_vals, levels=6, colors="white", linewidths=0.7, alpha=0.7)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("Mean Error")
    ax.set_yticks(y_vals)
    ax.set_yticklabels(z_df.index.tolist())
    ax.set_title("Interaction Surface: Error over SNR x Overlap")
    ax.set_xlabel("Overlap Ratio")
    ax.set_ylabel("SNR")
    return fig, ax


def plot_error_variability_by_condition(success_df: pd.DataFrame) -> Tuple[plt.Figure, List[plt.Axes]]:
    plot_df = success_df.dropna(subset=["error_value"]).copy()

    snr_var = (
        plot_df.groupby("snr_db", dropna=False)["error_value"]
        .agg(std="std", iqr=lambda s: s.quantile(0.75) - s.quantile(0.25))
        .reset_index()
    )
    ordered_snr = _ordered_snr_levels(snr_var["snr_db"])
    ordered_labels = [_snr_label(v) for v in ordered_snr]
    snr_var["snr_label"] = snr_var["snr_db"].apply(_snr_label)
    snr_var["snr_label"] = pd.Categorical(snr_var["snr_label"], categories=ordered_labels, ordered=True)
    snr_var = snr_var.sort_values("snr_label")

    ovr_var = (
        plot_df.groupby("overlap_ratio", dropna=False)["error_value"]
        .agg(std="std", iqr=lambda s: s.quantile(0.75) - s.quantile(0.25))
        .reset_index()
        .sort_values("overlap_ratio")
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].plot(snr_var["snr_label"].astype(str), snr_var["std"], marker="o", label="Std")
    axes[0].plot(snr_var["snr_label"].astype(str), snr_var["iqr"], marker="o", label="IQR")
    axes[0].set_title("Error Variability by SNR")
    axes[0].set_xlabel("SNR")
    axes[0].set_ylabel("Spread")
    axes[0].legend()

    axes[1].plot(ovr_var["overlap_ratio"], ovr_var["std"], marker="o", label="Std")
    axes[1].plot(ovr_var["overlap_ratio"], ovr_var["iqr"], marker="o", label="IQR")
    axes[1].set_title("Error Variability by Overlap")
    axes[1].set_xlabel("Overlap Ratio")
    axes[1].set_ylabel("Spread")
    axes[1].legend()
    return fig, [axes[0], axes[1]]


def plot_error_vs_support(success_df: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
    plot_df = success_df.copy()
    plot_df["snr_label"] = plot_df["snr_db"].apply(_snr_label)

    cond = (
        plot_df.groupby(["snr_label", "overlap_ratio"], dropna=False)["error_value"]
        .agg(mean_error="mean", support="count")
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=cond,
        x="support",
        y="mean_error",
        hue="snr_label",
        style="overlap_ratio",
        s=110,
        ax=ax,
    )
    ax.set_title("Sample Efficiency: Mean Error vs Support")
    ax.set_xlabel("Sample count per condition")
    ax.set_ylabel("Mean Error")
    ax.legend(title="SNR / OVR", bbox_to_anchor=(1.02, 1), loc="upper left")
    return fig, ax


def plot_error_threshold_exceedance_by_snr(
    success_df: pd.DataFrame,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    plot_df = success_df.dropna(subset=["error_value"]).copy()
    plot_df["snr_label"] = plot_df["snr_db"].apply(_snr_label)
    ordered_snr = _ordered_snr_levels(plot_df["snr_db"])
    ordered_labels = [_snr_label(v) for v in ordered_snr]

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 51)

    rows: List[Dict[str, float]] = []
    for label in ordered_labels:
        vals = plot_df.loc[plot_df["snr_label"] == label, "error_value"].to_numpy()
        if vals.size == 0:
            continue
        for t in thresholds:
            rows.append(
                {
                    "snr_label": label,
                    "threshold": float(t),
                    "rate": float((vals > t).mean()),
                }
            )

    all_vals = plot_df["error_value"].to_numpy()
    if all_vals.size > 0:
        for t in thresholds:
            rows.append(
                {
                    "snr_label": "Overall",
                    "threshold": float(t),
                    "rate": float((all_vals > t).mean()),
                }
            )

    curve_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=curve_df[curve_df["snr_label"] != "Overall"],
        x="threshold",
        y="rate",
        hue="snr_label",
        marker=None,
        ax=ax,
    )
    sns.lineplot(
        data=curve_df[curve_df["snr_label"] == "Overall"],
        x="threshold",
        y="rate",
        color="black",
        linestyle="--",
        linewidth=2.2,
        label="Overall",
        ax=ax,
    )

    ax.set_title("Failure Curve: P(error > t) by SNR")
    ax.set_xlabel("Threshold t")
    ax.set_ylabel("Proportion above threshold")
    ax.legend(title="Series", bbox_to_anchor=(1.02, 1), loc="upper left")
    return fig, ax

def plot_snr_ovr_heatmap(cross_mean: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
    plt.figure(figsize=(10, 4.5))
    sns.heatmap(
        cross_mean,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        linewidths=0.5,
        cbar_kws={"label": "Mean error"},
    )
    plt.xlabel("Overlap ratio")
    plt.ylabel("SNR")
    plt.title("Mean Error Heatmap by SNR and Overlap")
    plt.tight_layout()
    return plt.gcf(), plt.gca()