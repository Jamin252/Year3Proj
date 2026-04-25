from __future__ import annotations

import ast
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "Documentation" / "figures"
SUMMARY_PATH = PROJECT_ROOT / "Documentation" / "statistical_analysis_summary.json"
WORST_WER_PATH = PROJECT_ROOT / "Documentation" / "worst_wer_examples.json"
WER_OUTLIER_CONDITIONS_PATH = PROJECT_ROOT / "Documentation" / "wer_outlier_condition_summary.json"
REAL_EVAL_WER_ORC_PATH = PROJECT_ROOT / "Output" / "real_eval_wer_orc_results.json"

CROSS_MODEL_CLIP_REGEX = re.compile(
    r"(^mix_[0-9]+_0\.(00|14|20|40)_2_7\.4_T$)"
    r"|(^mix_[0-9]+_0\.14_2_(None|7\.4|0|-5)_T$)"
)

CROSS_MODEL_CLIP_REGEX_PLAIN = re.compile(".*")

MODELS = ["faster-whisper", "whisperx", "wav2vec2", "parakeet"]
MODEL_LABELS = {
    "faster-whisper": "faster-whisper",
    "whisperx": "WhisperX",
    "wav2vec2": "wav2vec2",
    "parakeet": "Parakeet",
}
SNR_ORDER = ["clean", "7.4", "0", "-5"]
OVR_ORDER = [0.00, 0.14, 0.20, 0.40]
DSS_PERMUTATION_N = 10_000
DSS_RANDOM_SEED = 20260425
DSS_DIFF_TOLERANCE = 1e-12


def _maybe_text(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text if text else None
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    parts.append(text)
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                text = str(item[1]).strip()
                if text:
                    parts.append(text)
            elif isinstance(item, dict):
                text = (
                    _maybe_text(item.get("text"))
                    or _maybe_text(item.get("transcript"))
                    or _maybe_text(item.get("hypothesis"))
                    or _maybe_text(item.get("reference"))
                )
                if text:
                    parts.append(text)
        return " ".join(parts) if parts else None
    if isinstance(value, dict):
        return (
            _maybe_text(value.get("text"))
            or _maybe_text(value.get("transcript"))
            or _maybe_text(value.get("hypothesis"))
            or _maybe_text(value.get("reference"))
        )
    return None


def load_transcriptions() -> dict[str, Any]:
    path = PROJECT_ROOT / "ASR_transcriptions.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def load_reference_transcripts() -> dict[str, str]:
    manifest_path = PROJECT_ROOT / "Output" / "manifest.csv"
    if not manifest_path.exists():
        return {}

    refs: dict[str, str] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip_id = (row.get("clip_id") or "").strip()
            transcript_raw = row.get("transcript")
            if not clip_id or not transcript_raw:
                continue

            text_parts: list[str] = []
            try:
                segments = ast.literal_eval(transcript_raw)
            except (ValueError, SyntaxError):
                segments = None

            if isinstance(segments, (list, tuple)):
                for segment in segments:
                    # Expected segment layout is typically (speaker_id, text, [start], [end]).
                    if isinstance(segment, (list, tuple)) and len(segment) >= 2 and isinstance(segment[1], str):
                        text = segment[1].strip()
                        if text:
                            text_parts.append(text)
                    elif isinstance(segment, dict):
                        text = _maybe_text(segment.get("text")) or _maybe_text(segment.get("transcript"))
                        if text:
                            text_parts.append(text)

            if text_parts:
                refs[clip_id] = " ".join(text_parts)

    return refs


def get_k_worst_wer_examples(k: int = 3) -> dict[str, list[dict[str, Any]]]:
    k = max(1, int(k))
    transcriptions = load_transcriptions()
    references = load_reference_transcripts()
    rows_by_model: dict[str, list[dict[str, Any]]] = {model: [] for model in MODELS}

    for model in MODELS:
        path = PROJECT_ROOT / f"WER_results_{model}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        for result in payload.get("results", []):
            if result.get("status") != "success":
                continue

            metrics = result.get("metrics") or {}
            wer = metrics.get("wer")
            if wer is None:
                continue

            clip_id = result.get("clip_id", "")
            ref_text = (
                _maybe_text(result.get("reference"))
                or _maybe_text(result.get("ref_text"))
                or _maybe_text(result.get("ref"))
            )
            if ref_text is None:
                ref_text = _maybe_text(references.get(clip_id))
            hyp_text = (
                _maybe_text(result.get("hypothesis"))
                or _maybe_text(result.get("hyp_text"))
                or _maybe_text(result.get("hyp"))
            )

            if hyp_text is None:
                clip_payload = transcriptions.get(clip_id, {}) if isinstance(transcriptions, dict) else {}
                transcript_map = clip_payload.get("transcript", {}) if isinstance(clip_payload, dict) else {}
                if isinstance(transcript_map, dict):
                    hyp_text = _maybe_text(transcript_map.get(model))

            rows_by_model[model].append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS.get(model, model),
                    "clip_id": clip_id,
                    "wer": float(wer),
                    "reference": ref_text.lower() if isinstance(ref_text, str) else ref_text,
                    "hypothesis": hyp_text,
                }
            )

    worst_by_model: dict[str, list[dict[str, Any]]] = {}
    for model in MODELS:
        model_rows = rows_by_model.get(model, [])
        model_rows.sort(key=lambda item: item["wer"], reverse=True)
        worst_by_model[model] = model_rows[:k]
    return worst_by_model


def parse_clip_id(clip_id: str) -> dict[str, Any]:
    parts = clip_id.split("_")
    return {
        "clip_id": clip_id,
        "base_id": parts[1] if len(parts) > 1 else None,
        "overlap_ratio": float(parts[2]) if len(parts) > 2 else None,
        "max_speakers": int(parts[3]) if len(parts) > 3 else None,
        "snr_db": None if len(parts) <= 4 or parts[4] == "None" else float(parts[4]),
        "snr_label": "clean" if len(parts) <= 4 or parts[4] == "None" else parts[4],
        "noise_type": parts[5] if len(parts) > 5 else None,
    }


def load_model_results(model: str) -> pd.DataFrame:
    path = PROJECT_ROOT / f"WER_results_{model}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for result in payload["results"]:
        row = {
            "model": model,
            "model_label": MODEL_LABELS[model],
            "status": result.get("status"),
            "wer": None,
            "dss_wer": None,
            "orc_wer": None,
            "cpwer": None,
            "is_cross_model_clip": bool(CROSS_MODEL_CLIP_REGEX.match(result.get("clip_id", ""))),
        }
        row.update(parse_clip_id(result.get("clip_id", "")))
        metrics = result.get("metrics") or {}
        row["wer"] = metrics.get("wer")
        row["dss_wer"] = metrics.get("mrs_wer")
        row["orc_wer"] = metrics.get("orc_wer")
        row["cpwer"] = metrics.get("cpwer")
        rows.append(row)
    return pd.DataFrame(rows)


def load_real_eval_wer_orc_results() -> pd.DataFrame:
    payload = json.loads(REAL_EVAL_WER_ORC_PATH.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for clip_id, by_model in payload.get("by_clip", {}).items():
        if not isinstance(by_model, dict):
            continue
        for model, values in by_model.items():
            if model not in MODEL_LABELS or not isinstance(values, dict):
                continue
            rows.append(
                {
                    "clip_id": clip_id,
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "wer": values.get("wer"),
                    "orc_wer": values.get("orc_wer"),
                    "hyp_segment_count": values.get("hyp_segment_count"),
                    "ref_segment_count": values.get("ref_segment_count"),
                    "is_segmented_hypothesis": bool(values.get("is_segmented_hypothesis", False)),
                }
            )
    return pd.DataFrame(rows)


def build_real_transferability_summary(cross: pd.DataFrame, real_eval: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        synthetic_wer = cross.loc[cross["model"] == model, "wer"].dropna()
        real_model = real_eval[real_eval["model"] == model]
        real_wer = real_model["wer"].dropna()
        real_orc_wer = real_model["orc_wer"].dropna()
        if synthetic_wer.empty or real_wer.empty:
            continue

        rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "synthetic_n": int(synthetic_wer.size),
                "synthetic_mean": float(synthetic_wer.mean()),
                "synthetic_median": float(synthetic_wer.median()),
                "synthetic_sd": float(synthetic_wer.std(ddof=1)),
                "real_n": int(real_wer.size),
                "real_mean": float(real_wer.mean()),
                "real_median": float(real_wer.median()),
                "real_sd": float(real_wer.std(ddof=1)),
                "real_orc_wer_n": int(real_orc_wer.size),
                "real_orc_wer_mean": None if real_orc_wer.empty else float(real_orc_wer.mean()),
                "delta_mean": float(real_wer.mean() - synthetic_wer.mean()),
                "mean_ratio": float(real_wer.mean() / synthetic_wer.mean()),
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    summary["synthetic_rank"] = summary["synthetic_mean"].rank(method="min", ascending=True).astype(int)
    summary["real_rank"] = summary["real_mean"].rank(method="min", ascending=True).astype(int)
    summary["rank_shift"] = summary["real_rank"] - summary["synthetic_rank"]
    return summary


def summarise(values: pd.Series) -> dict[str, float]:
    values = values.dropna()
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "sd": float(values.std(ddof=1)),
        "min": float(values.min()),
        "max": float(values.max()),
        "pct_gt_1": float((values > 1.0).mean() * 100),
    }


def _stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    offset = sum((idx + 1) * ord(char) for idx, char in enumerate(text))
    return int((DSS_RANDOM_SEED + offset) % (2**32 - 1))


def _paired_permutation_p_value(
    differences: np.ndarray,
    alternative: str,
    n_permutations: int = DSS_PERMUTATION_N,
    seed: int = DSS_RANDOM_SEED,
    chunk_size: int = 500,
) -> float:
    if differences.size == 0:
        return float("nan")

    observed = float(np.mean(differences))
    rng = np.random.default_rng(seed)
    extreme_count = 0
    completed = 0

    while completed < n_permutations:
        take = min(chunk_size, n_permutations - completed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(take, differences.size))
        null_means = (signs @ differences) / differences.size

        if alternative == "less":
            extreme_count += int(np.sum(null_means <= observed))
        elif alternative == "greater":
            extreme_count += int(np.sum(null_means >= observed))
        elif alternative == "two-sided":
            extreme_count += int(np.sum(np.abs(null_means) >= abs(observed)))
        else:
            raise ValueError(f"Unsupported alternative: {alternative}")

        completed += take

    return float((1 + extreme_count) / (n_permutations + 1))


def _bootstrap_mean_ci(
    differences: np.ndarray,
    n_resamples: int = DSS_PERMUTATION_N,
    seed: int = DSS_RANDOM_SEED,
    chunk_size: int = 500,
) -> tuple[float, float]:
    if differences.size == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=float)
    completed = 0

    while completed < n_resamples:
        take = min(chunk_size, n_resamples - completed)
        indices = rng.integers(0, differences.size, size=(take, differences.size))
        means[completed : completed + take] = differences[indices].mean(axis=1)
        completed += take

    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def analyse_dss_wer_hypothesis_tests(results: pd.DataFrame) -> pd.DataFrame:
    frame = results.copy()
    if "status" in frame.columns:
        frame = frame[frame["status"] == "success"]
    frame = frame.dropna(subset=["model", "model_label", "wer", "dss_wer", "overlap_ratio"]).copy()
    non_overlap_mask = np.isclose(frame["overlap_ratio"], 0.0)
    frame["overlap_scope"] = np.where(non_overlap_mask, "non_overlap", "overlap")
    frame["overlap_scope_label"] = np.where(frame["overlap_scope"] == "non_overlap", "Non-overlap", "Overlap")
    frame["dss_minus_wer"] = frame["dss_wer"] - frame["wer"]

    rows: list[dict[str, Any]] = []
    scopes = [("non_overlap", "Non-overlap", "two-sided"), ("overlap", "Overlap", "less")]

    for scope, scope_label, alternative in scopes:
        scope_df = frame[frame["overlap_scope"] == scope].copy()
        groups: list[tuple[str, str, pd.DataFrame]] = [
            (model, MODEL_LABELS[model], scope_df[scope_df["model"] == model].copy())
            for model in MODELS
        ]
        groups.append(("all_models", "All models", scope_df))

        for model, model_label, group in groups:
            if group.empty:
                continue

            differences = group["dss_minus_wer"].to_numpy(dtype=float)
            ci_low, ci_high = _bootstrap_mean_ci(
                differences,
                seed=_stable_seed("bootstrap", scope, model),
            )
            p_value = _paired_permutation_p_value(
                differences,
                alternative=alternative,
                seed=_stable_seed("permutation", scope, model),
            )

            equal_mask = np.isclose(differences, 0.0, atol=DSS_DIFF_TOLERANCE, rtol=0.0)
            less_mask = differences < -DSS_DIFF_TOLERANCE
            greater_mask = differences > DSS_DIFF_TOLERANCE
            n = int(differences.size)

            rows.append(
                {
                    "overlap_scope": scope,
                    "overlap_scope_label": scope_label,
                    "model": model,
                    "model_label": model_label,
                    "n": n,
                    "wer_mean": float(group["wer"].mean()),
                    "wer_median": float(group["wer"].median()),
                    "wer_sd": float(group["wer"].std(ddof=1)),
                    "dss_wer_mean": float(group["dss_wer"].mean()),
                    "dss_wer_median": float(group["dss_wer"].median()),
                    "dss_wer_sd": float(group["dss_wer"].std(ddof=1)),
                    "mean_difference": float(np.mean(differences)),
                    "median_difference": float(np.median(differences)),
                    "difference_sd": float(np.std(differences, ddof=1)),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "permutation_p": p_value,
                    "permutation_alternative": alternative,
                    "n_permutations": int(DSS_PERMUTATION_N),
                    "dss_equal_wer_n": int(np.sum(equal_mask)),
                    "dss_less_than_wer_n": int(np.sum(less_mask)),
                    "dss_greater_than_wer_n": int(np.sum(greater_mask)),
                    "dss_equal_wer_pct": float(np.mean(equal_mask) * 100.0),
                    "dss_less_than_wer_pct": float(np.mean(less_mask) * 100.0),
                    "dss_greater_than_wer_pct": float(np.mean(greater_mask) * 100.0),
                }
            )

    return pd.DataFrame(rows)


def summarise_wer_outlier_conditions_by_model(results: pd.DataFrame) -> dict[str, Any]:
    """
    Summarise condition proportions among high-WER outliers per model.

    Outliers are defined per model using the IQR rule:
        WER > Q3 + 1.5 * IQR
    """
    required_cols = {"model", "wer", "overlap_ratio", "snr_label", "noise_type"}
    missing = required_cols.difference(results.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = results.copy()
    if "status" in frame.columns:
        frame = frame[frame["status"] == "success"]
    frame = frame.dropna(subset=["model", "wer"])

    def _ordered_keys(index_values: list[str], preferred_order: list[str]) -> list[str]:
        preferred = [key for key in preferred_order if key in index_values]
        extras = sorted(key for key in index_values if key not in preferred_order)
        return preferred + extras

    def _proportion_dict(series: pd.Series, n_total: int, preferred_order: list[str] | None = None) -> dict[str, float]:
        if n_total == 0:
            return {}
        counts = series.fillna("unknown").astype(str).value_counts()
        keys = list(counts.index)
        if preferred_order is not None:
            keys = _ordered_keys(keys, preferred_order)
        else:
            keys = sorted(keys)
        return {key: float(counts.get(key, 0) / n_total) for key in keys}

    def _side_summary(
        low_share: float,
        high_share: float,
        low_label: str,
        high_label: str,
        tolerance: float = 0.10,
    ) -> dict[str, Any]:
        if (low_share + high_share) == 0.0:
            side_label = "none"
        elif abs(high_share - low_share) <= tolerance:
            side_label = "mixed"
        elif high_share > low_share:
            side_label = high_label
        else:
            side_label = low_label
        return {
            "label": side_label,
            "low_share": float(low_share),
            "high_share": float(high_share),
        }

    summary: dict[str, Any] = {}
    model_order = [model for model in MODELS if model in frame["model"].unique()]
    extra_models = sorted(model for model in frame["model"].unique() if model not in MODELS)

    for model in model_order + extra_models:
        model_df = frame[frame["model"] == model].copy()
        total = int(len(model_df))
        if total == 0:
            continue

        q1 = float(model_df["wer"].quantile(0.25))
        q3 = float(model_df["wer"].quantile(0.75))
        iqr = q3 - q1
        upper_fence = q3 + 1.5 * iqr
        outliers = model_df[model_df["wer"] > upper_fence].copy()
        n_outliers = int(len(outliers))

        overlap_as_text = outliers["overlap_ratio"].map(
            lambda value: "unknown" if pd.isna(value) else f"{float(value):.2f}"
        )
        ovr_proportions = _proportion_dict(
            overlap_as_text,
            n_outliers,
            preferred_order=[f"{value:.2f}" for value in OVR_ORDER] + ["unknown"],
        )
        snr_proportions = _proportion_dict(
            outliers["snr_label"],
            n_outliers,
            preferred_order=SNR_ORDER + ["unknown"],
        )

        ovr_low_share = float(ovr_proportions.get("0.00", 0.0) + ovr_proportions.get("0.14", 0.0))
        ovr_high_share = float(ovr_proportions.get("0.20", 0.0) + ovr_proportions.get("0.40", 0.0))
        snr_high_share = float(snr_proportions.get("clean", 0.0) + snr_proportions.get("7.4", 0.0))
        snr_low_share = float(snr_proportions.get("0", 0.0) + snr_proportions.get("-5", 0.0))

        summary[model] = {
            "n_total": total,
            "n_outliers": n_outliers,
            "outlier_rate": float(n_outliers / total) if total else 0.0,
            "wer_outlier_rule": {
                "q1": q1,
                "q3": q3,
                "iqr": float(iqr),
                "upper_fence": float(upper_fence),
            },
            "outlier_condition_proportions": {
                "overlap_ratio": ovr_proportions,
                "snr_label": snr_proportions,
                "noise_type": _proportion_dict(outliers["noise_type"], n_outliers),
            },
            "outlier_side": {
                "overlap": _side_summary(
                    low_share=ovr_low_share,
                    high_share=ovr_high_share,
                    low_label="lower_overlap_side",
                    high_label="higher_overlap_side",
                ),
                "snr": _side_summary(
                    low_share=snr_low_share,
                    high_share=snr_high_share,
                    low_label="lower_snr_side",
                    high_label="higher_snr_side",
                ),
            },
        }

    return summary


def snr_amplitude_ratio(snr_db: float | None) -> float:
    if pd.isna(snr_db):
        return 0.0
    return 10 ** (-float(snr_db) / 20.0)


def _linear_fit_metrics(y: np.ndarray, y_hat: np.ndarray) -> dict[str, float]:
    residuals = y - y_hat
    rss = float(np.sum(residuals**2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "r2": float(1.0 - rss / tss),
        "mae": float(np.mean(np.abs(residuals))),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
    }


def compute_effect_sizes(wav2: pd.DataFrame) -> dict[str, float]:
    """Compute partial eta-squared for SNR and OVR main effects."""
    fit_df = wav2.dropna(subset=["wer", "overlap_ratio", "snr_label"]).copy()
    
    # Grand mean
    grand_mean = fit_df["wer"].mean()
    ss_total = ((fit_df["wer"] - grand_mean) ** 2).sum()
    
    # SNR effect
    snr_means = fit_df.groupby("snr_label")["wer"].agg(["mean", "size"])
    ss_snr = (snr_means["size"] * (snr_means["mean"] - grand_mean) ** 2).sum()
    
    # OVR effect
    ovr_means = fit_df.groupby("overlap_ratio")["wer"].agg(["mean", "size"])
    ss_ovr = (ovr_means["size"] * (ovr_means["mean"] - grand_mean) ** 2).sum()
    
    # Residual (two-way interaction + error)
    # Note: This is a simplified model; full two-way ANOVA would decompose further
    ss_residual = ss_total - ss_snr - ss_ovr
    
    return {
        "snr_partial_eta_squared": float(ss_snr / (ss_snr + ss_residual)),
        "ovr_partial_eta_squared": float(ss_ovr / (ss_ovr + ss_residual)),
        "ss_snr": float(ss_snr),
        "ss_ovr": float(ss_ovr),
        "ss_total": float(ss_total),
    }


def fit_wav2vec2_wer_surface(wav2: pd.DataFrame) -> dict[str, Any]:
    fit_df = wav2.dropna(subset=["wer", "overlap_ratio"]).copy()
    fit_df["snr_amplitude_ratio"] = fit_df["snr_db"].apply(snr_amplitude_ratio)
    interaction = fit_df["overlap_ratio"] * fit_df["snr_amplitude_ratio"]
    x = np.column_stack(
        [
            np.ones(len(fit_df)),
            fit_df["overlap_ratio"].to_numpy(),
            fit_df["snr_amplitude_ratio"].to_numpy(),
            interaction.to_numpy(),
        ]
    )
    y = fit_df["wer"].to_numpy()
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    y_hat = x @ coefficients

    condition_df = (
        fit_df.groupby(["overlap_ratio", "snr_label", "snr_amplitude_ratio"], as_index=False)["wer"]
        .mean()
        .sort_values(["snr_amplitude_ratio", "overlap_ratio"])
    )
    condition_interaction = condition_df["overlap_ratio"] * condition_df["snr_amplitude_ratio"]
    condition_x = np.column_stack(
        [
            np.ones(len(condition_df)),
            condition_df["overlap_ratio"].to_numpy(),
            condition_df["snr_amplitude_ratio"].to_numpy(),
            condition_interaction.to_numpy(),
        ]
    )
    condition_y = condition_df["wer"].to_numpy()
    condition_y_hat = condition_x @ coefficients

    return {
        "model": "WER_hat = beta0 + beta_ovr*OVR + beta_snr*z + beta_interaction*OVR*z",
        "snr_transform": "z = 0 for clean, otherwise 10^(-SNR_dB/20)",
        "coefficients": {
            "intercept": float(coefficients[0]),
            "overlap_ratio": float(coefficients[1]),
            "snr_amplitude_ratio": float(coefficients[2]),
            "overlap_ratio_x_snr_amplitude_ratio": float(coefficients[3]),
        },
        "clip_level_metrics": {
            "n": int(len(fit_df)),
            **_linear_fit_metrics(y, y_hat),
        },
        "condition_mean_metrics": {
            "n": int(len(condition_df)),
            **_linear_fit_metrics(condition_y, condition_y_hat),
        },
    }


def save(fig: plt.Figure, filename: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_wav2vec2_snr_violin(wav2: pd.DataFrame) -> None:
    plot_df = wav2.copy()
    plot_df["snr_label"] = pd.Categorical(plot_df["snr_label"], categories=SNR_ORDER, ordered=True)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    data = [plot_df.loc[plot_df["snr_label"] == label, "wer"].dropna().to_numpy() for label in SNR_ORDER]
    _draw_violin_plot(ax, data, SNR_ORDER, facecolor="#9ecae1", edgecolor="#4a6f82")
    ax.set_xlabel("SNR condition")
    ax.set_ylabel("Clip-level WER")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "stat_wav2vec2_snr_box.png")


def plot_wav2vec2_heatmap(wav2: pd.DataFrame) -> None:
    plot_df = wav2.copy()
    pivot = plot_df.pivot_table(
        index="snr_label",
        columns="overlap_ratio",
        values="wer",
        aggfunc="mean",
    ).reindex(SNR_ORDER)
    counts = plot_df.pivot_table(
        index="snr_label",
        columns="overlap_ratio",
        values="wer",
        aggfunc="count",
    ).reindex(SNR_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), constrained_layout=True)
    _draw_heatmap(axes[0], pivot, "Mean WER", "YlOrRd", ".3f", colorbar_label="Mean WER")
    _draw_heatmap(axes[1], counts, "Cell count", "Blues", ".0f", colorbar_label=None)
    save(fig, "stat_wav2vec2_snr_overlap_heatmap.png")


def plot_wav2vec2_wer_vs_overlap(wav2: pd.DataFrame) -> None:
    grouped = (
        wav2.groupby(["snr_label", "overlap_ratio"])["wer"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["sem"] = grouped["std"] / np.sqrt(grouped["count"])
    grouped["ci"] = grouped["sem"] * 1.96
    
    overall_grouped = wav2.groupby("overlap_ratio")["wer"].agg(["mean", "std", "count"]).reset_index()
    overall_grouped["sem"] = overall_grouped["std"] / np.sqrt(overall_grouped["count"])
    overall_grouped["ci"] = overall_grouped["sem"] * 1.96
    
    colors = {
        "clean": "#6b7280",
        "7.4": "#1f77b4",
        "0": "#ff7f0e",
        "-5": "#d62728",
    }

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    
    # Plot SD shaded bands for each SNR
    for snr_label in SNR_ORDER:
        line_data = (
            grouped[grouped["snr_label"] == snr_label]
            .set_index("overlap_ratio")[["mean", "std"]]
            .reindex(OVR_ORDER)
        )
        ax.fill_between(
            OVR_ORDER,
            line_data["mean"] - line_data["std"],
            line_data["mean"] + line_data["std"],
            color=colors[snr_label],
            alpha=0.15,
            label=None,
        )
    
    # Plot means without error bars
    for snr_label in SNR_ORDER:
        line = (
            grouped[grouped["snr_label"] == snr_label]
            .set_index("overlap_ratio")["mean"]
            .reindex(OVR_ORDER)
        )
        ax.plot(
            OVR_ORDER,
            line.values,
            marker="o",
            linewidth=1.8,
            color=colors[snr_label],
            label=f"SNR {snr_label}",
        )
    
    # Overall line
    overall_line = overall_grouped.set_index("overlap_ratio").reindex(OVR_ORDER)
    ax.plot(
        OVR_ORDER,
        overall_line["mean"].values,
        marker="o",
        linewidth=2.4,
        linestyle="--",
        color="#111111",
        label="Overall",
    )
    
    ax.set_xlabel("Overlap ratio")
    ax.set_ylabel("Mean clip-level WER")
    ax.set_title("wav2vec2 WER vs overlap ratio")
    ax.set_xticks(OVR_ORDER)
    max_val = (grouped["mean"] + grouped["std"]).max()
    ax.set_ylim(0, max_val * 1.16)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Series", fontsize=7, title_fontsize=8, loc="upper left")
    save(fig, "stat_wav2vec2_wer_vs_ovr.png")


def plot_wav2vec2_wer_vs_snr(wav2: pd.DataFrame) -> None:
    grouped = (
        wav2.groupby(["overlap_ratio", "snr_label"])["wer"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["sem"] = grouped["std"] / np.sqrt(grouped["count"])
    grouped["ci"] = grouped["sem"] * 1.96
    
    overall_grouped = wav2.groupby("snr_label")["wer"].agg(["mean", "std", "count"]).reset_index()
    overall_grouped["sem"] = overall_grouped["std"] / np.sqrt(overall_grouped["count"])
    overall_grouped["ci"] = overall_grouped["sem"] * 1.96
    
    x_positions = list(range(len(SNR_ORDER)))
    colors = {
        0.00: "#1f77b4",
        0.14: "#ff7f0e",
        0.20: "#d62728",
        0.40: "#9467bd",
    }

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    
    # Plot SD shaded bands for each overlap ratio
    for overlap_ratio in OVR_ORDER:
        line_data = (
            grouped[grouped["overlap_ratio"] == overlap_ratio]
            .set_index("snr_label")[["mean", "std"]]
            .reindex(SNR_ORDER)
        )
        x_vals = np.array(x_positions)
        ax.fill_between(
            x_vals,
            line_data["mean"].values - line_data["std"].values,
            line_data["mean"].values + line_data["std"].values,
            color=colors[overlap_ratio],
            alpha=0.15,
            label=None,
        )
    
    # Plot means without error bars
    for overlap_ratio in OVR_ORDER:
        line = (
            grouped[grouped["overlap_ratio"] == overlap_ratio]
            .set_index("snr_label")["mean"]
            .reindex(SNR_ORDER)
        )
        ax.plot(
            x_positions,
            line.values,
            marker="o",
            linewidth=1.8,
            color=colors[overlap_ratio],
            label=f"OVR {overlap_ratio:.2f}",
        )
    
    # Overall line
    overall_line = overall_grouped.set_index("snr_label").reindex(SNR_ORDER)
    ax.plot(
        x_positions,
        overall_line["mean"].values,
        marker="o",
        linewidth=2.4,
        linestyle="--",
        color="#111111",
        label="Overall",
    )
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(SNR_ORDER)
    ax.set_xlabel("SNR condition")
    ax.set_ylabel("Mean clip-level WER")
    ax.set_title("wav2vec2 WER vs SNR")
    max_val = (grouped["mean"] + grouped["std"]).max()
    ax.set_ylim(0, max_val * 1.16)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Series", fontsize=7, title_fontsize=8, loc="upper left")
    save(fig, "stat_wav2vec2_wer_vs_snr.png")


def plot_wav2vec2_delta_vs_overlap(wav2: pd.DataFrame) -> pd.DataFrame:
    mean_pivot = (
        wav2.pivot_table(
            index="overlap_ratio",
            columns="snr_label",
            values="wer",
            aggfunc="mean",
        )
        .reindex(index=OVR_ORDER, columns=SNR_ORDER)
    )
    delta_pivot = mean_pivot.subtract(mean_pivot["clean"], axis=0)
    colors = {
        "clean": "#6b7280",
        "7.4": "#1f77b4",
        "0": "#ff7f0e",
        "-5": "#d62728",
    }

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for snr_label in SNR_ORDER:
        linestyle = "--" if snr_label == "clean" else "-"
        label = "clean baseline" if snr_label == "clean" else f"SNR {snr_label}"
        ax.plot(
            OVR_ORDER,
            delta_pivot[snr_label].values,
            marker="o",
            linewidth=2.0,
            linestyle=linestyle,
            color=colors[snr_label],
            label=label,
        )
    ax.axhline(0.0, color="#333333", linewidth=0.9, alpha=0.6)
    ax.set_xlabel("Overlap ratio")
    ax.set_ylabel(r"$\Delta$ WER relative to clean")
    ax.set_title("wav2vec2 noise-induced delta WER vs overlap")
    ax.set_xticks(OVR_ORDER)
    ax.set_ylim(min(-0.02, delta_pivot.min().min() * 1.12), delta_pivot.max().max() * 1.18)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Series", fontsize=7, title_fontsize=8, loc="upper left")
    save(fig, "stat_wav2vec2_delta_vs_ovr.png")
    return delta_pivot


def plot_wav2vec2_delta_vs_snr(wav2: pd.DataFrame) -> pd.DataFrame:
    mean_pivot = (
        wav2.pivot_table(
            index="overlap_ratio",
            columns="snr_label",
            values="wer",
            aggfunc="mean",
        )
        .reindex(index=OVR_ORDER, columns=SNR_ORDER)
    )
    delta_pivot = mean_pivot.subtract(mean_pivot["clean"], axis=0)
    colors = {
        0.00: "#1f77b4",
        0.14: "#ff7f0e",
        0.20: "#d62728",
        0.40: "#9467bd",
    }
    x_positions = list(range(len(SNR_ORDER)))

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for overlap_ratio in OVR_ORDER:
        ax.plot(
            x_positions,
            delta_pivot.loc[overlap_ratio, SNR_ORDER].values,
            marker="o",
            linewidth=2.0,
            color=colors[overlap_ratio],
            label=f"OVR {overlap_ratio:.2f}",
        )
    ax.axhline(0.0, color="#333333", linewidth=0.9, alpha=0.6)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(SNR_ORDER)
    ax.set_xlabel("SNR condition")
    ax.set_ylabel(r"$\Delta$ WER relative to clean")
    ax.set_title("wav2vec2 noise-induced delta WER vs SNR")
    ax.set_ylim(min(-0.02, delta_pivot.min().min() * 1.12), delta_pivot.max().max() * 1.18)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Series", fontsize=7, title_fontsize=8, loc="upper left")
    save(fig, "stat_wav2vec2_delta_vs_snr.png")
    return delta_pivot


def plot_noise_type(wav2: pd.DataFrame) -> None:
    grouped = wav2.groupby("noise_type")["wer"]
    order = grouped.mean().sort_values(ascending=False).index.tolist()
    means = grouped.mean().reindex(order)
    sem = grouped.sem().reindex(order)
    ci = sem * 1.96
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(order, means, yerr=ci, color="#bdbdbd", edgecolor="#555555", capsize=4)
    ax.set_xlabel("Noise type")
    ax.set_ylabel("Mean clip-level WER")
    ax.grid(axis="y", alpha=0.25)
    save(fig, "stat_wav2vec2_noise_type.png")


def plot_cross_model_violin(cross: pd.DataFrame) -> None:
    order = ["faster-whisper", "WhisperX", "wav2vec2", "Parakeet"]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    data = [cross.loc[cross["model_label"] == label, "wer"].dropna().to_numpy() for label in order]
    _draw_violin_plot(ax, data, order, facecolor="#bcbddc", edgecolor="#5d5277")
    ax.set_xlabel("Model")
    ax.set_ylabel("Clip-level WER")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "stat_cross_model_wer_box.png")


def plot_cross_model_wer_vs_overlap(cross: pd.DataFrame) -> None:
    overlap_slice = cross[(cross["snr_label"] == "7.4") & (cross["noise_type"] == "T")].copy()
    colors = _model_colors()
    
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for label in ["faster-whisper", "WhisperX", "wav2vec2", "Parakeet"]:
        grouped = overlap_slice[overlap_slice["model_label"] == label].groupby("overlap_ratio")["wer"].mean()
        ax.plot(grouped.index, grouped.values, marker="o", linewidth=1.8, label=label, color=colors[label])
    
    ax.set_xlabel("Overlap ratio")
    ax.set_ylabel("Mean WER")
    ax.set_title("Cross-model WER vs overlap (SNR=7.4 dB, transportation noise)")
    ax.set_xticks(OVR_ORDER)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Model", fontsize=7, title_fontsize=8)
    save(fig, "stat_cross_model_wer_vs_ovr.png")


def plot_cross_model_wer_vs_snr(cross: pd.DataFrame) -> None:
    snr_slice = cross[(cross["overlap_ratio"] == 0.14) & (cross["noise_type"] == "T")].copy()
    snr_slice["snr_label"] = pd.Categorical(snr_slice["snr_label"], categories=SNR_ORDER, ordered=True)
    colors = _model_colors()
    
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for label in ["faster-whisper", "WhisperX", "wav2vec2", "Parakeet"]:
        grouped = snr_slice[snr_slice["model_label"] == label].groupby("snr_label", observed=False)["wer"].mean()
        grouped = grouped.reindex(SNR_ORDER)
        x_positions = list(range(len(SNR_ORDER)))
        ax.plot(x_positions, grouped.values, marker="o", linewidth=1.8, label=label, color=colors[label])
    
    ax.set_xticks(list(range(len(SNR_ORDER))))
    ax.set_xticklabels(SNR_ORDER)
    ax.set_xlabel("SNR condition")
    ax.set_ylabel("Mean WER")
    ax.set_title("Cross-model WER vs SNR (OVR=0.14, transportation noise)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Model", fontsize=7, title_fontsize=8)
    save(fig, "stat_cross_model_wer_vs_snr.png")


def plot_cross_model_scope_support(cross: pd.DataFrame) -> pd.DataFrame:
    support = (
        cross.groupby(["snr_label", "overlap_ratio"])["clip_id"]
        .nunique()
        .unstack()
        .reindex(index=SNR_ORDER, columns=OVR_ORDER)
    )

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#eeeeee")
    image = ax.imshow(support.to_numpy(dtype=float), cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(OVR_ORDER)))
    ax.set_xticklabels([f"{value:.2f}" for value in OVR_ORDER])
    ax.set_yticks(range(len(SNR_ORDER)))
    ax.set_yticklabels(SNR_ORDER)
    ax.set_xlabel("Overlap ratio")
    ax.set_ylabel("SNR condition")
    ax.set_title("Cross-model matched-scope support")

    for i, snr_label in enumerate(SNR_ORDER):
        for j, overlap_ratio in enumerate(OVR_ORDER):
            value = support.loc[snr_label, overlap_ratio]
            if pd.isna(value):
                ax.text(j, i, "-", ha="center", va="center", color="#777777", fontsize=9)
            else:
                text_color = "white" if float(value) >= 50 else "#1f2933"
                ax.text(j, i, f"{int(value)}", ha="center", va="center", color=text_color, fontsize=9)

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Unique mixtures")
    save(fig, "stat_cross_model_scope_support.png")
    return support


def load_timing_results() -> pd.DataFrame:
    timing_path = PROJECT_ROOT / "timings_100_random.json"
    payload = json.loads(timing_path.read_text(encoding="utf-8"))
    total_audio_duration_sec = payload.get("total_audio_duration_sec", 0.0)
    rows = []
    for row in payload["results"]:
        total_processing_time_sec = row.get("total_processing_time_sec", row.get("duration_sec"))
        sample_count = row["sample_count"]
        seconds_per_clip = row.get("sec_per_audio")
        if seconds_per_clip is None:
            seconds_per_clip = total_processing_time_sec / sample_count
        rtfx = row.get("rtfx")
        if rtfx is None and total_processing_time_sec:
            rtfx = total_audio_duration_sec / total_processing_time_sec
        rows.append(
            {
                "model": row["model"],
                "model_label": MODEL_LABELS[row["model"]],
                "sample_count": sample_count,
                "seconds_per_clip": seconds_per_clip,
                "rtfx": rtfx,
                "total_processing_time_sec": total_processing_time_sec,
            }
        )
    return pd.DataFrame(rows)


def plot_timing_bar(timing_df: pd.DataFrame) -> None:
    plot_df = timing_df.sort_values("seconds_per_clip", ascending=True).copy()
    colors = _model_colors()
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    bars = ax.barh(
        plot_df["model_label"],
        plot_df["seconds_per_clip"],
        color=[colors[label] for label in plot_df["model_label"]],
        alpha=0.82,
        edgecolor="#333333",
    )
    ax.set_xlabel("Seconds per 60 s clip")
    ax.set_ylabel("Model")
    ax.set_title("Inference time per clip")
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    for bar, value in zip(bars, plot_df["seconds_per_clip"]):
        ax.text(
            value + 0.35,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}s",
            va="center",
            ha="left",
            fontsize=8,
        )
    ax.set_xlim(0, max(plot_df["seconds_per_clip"]) * 1.18)
    save(fig, "stat_time_per_clip_bar.png")


def plot_timing_tradeoff(cross: pd.DataFrame, timing_df: pd.DataFrame) -> None:
    cross_model_mask = cross["clip_id"].astype(str).str.match(CROSS_MODEL_CLIP_REGEX.pattern, na=False)
    cross = cross[cross_model_mask].copy()
    timing_df = timing_df[
        ["model", "model_label", "seconds_per_clip", "rtfx"]
    ].copy()
    accuracy = cross.groupby(["model", "model_label"], as_index=False)["wer"].mean()
    plot_df = accuracy.merge(timing_df, on=["model", "model_label"])
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    colors = _model_colors()
    for _, row in plot_df.iterrows():
        ax.scatter(row["seconds_per_clip"], row["wer"], s=90, color=colors[row["model_label"]])
        ax.annotate(row["model_label"], (row["seconds_per_clip"], row["wer"]), xytext=(5, 2), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Seconds per 60 s clip (log scale)")
    ax.set_ylabel("Mean WER on cross-model subset")
    ax.grid(True, alpha=0.25)
    save(fig, "stat_accuracy_latency_tradeoff.png")


def plot_real_transfer_wer_change(transfer: pd.DataFrame) -> None:
    if transfer.empty:
        return

    colors = _model_colors()
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    x_positions = [0, 1]
    for _, row in transfer.sort_values("synthetic_rank").iterrows():
        values = [row["synthetic_mean"], row["real_mean"]]
        ax.plot(
            x_positions,
            values,
            marker="o",
            linewidth=2.0,
            markersize=5,
            color=colors[row["model_label"]],
            label=f"{row['model_label']} ({row['delta_mean']:+.3f})",
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(["Synthetic matched subset\n(n=700/model)", "Real CHiME-6 clips\n(n=100/model)"])
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylabel("Mean WER")
    ax.set_title("Transfer from synthetic subset to real conversational data")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title=r"$\Delta$ mean WER", fontsize=7, title_fontsize=8, loc="upper left")
    ax.set_ylim(0, max(transfer["real_mean"].max(), transfer["synthetic_mean"].max()) * 1.18)
    save(fig, "stat_real_transfer_wer_change.png")


def _draw_heatmap(
    ax: plt.Axes,
    data: pd.DataFrame,
    title: str,
    cmap: str,
    fmt: str,
    colorbar_label: str | None,
) -> None:
    image = ax.imshow(data.to_numpy(dtype=float), cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels([f"{float(col):.2f}" for col in data.columns])
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index.tolist())
    ax.set_xlabel("Overlap ratio")
    ax.set_ylabel("SNR condition")
    ax.set_title(title)
    for i, row in enumerate(data.index):
        for j, col in enumerate(data.columns):
            value = data.loc[row, col]
            ax.text(j, i, format(float(value), fmt), ha="center", va="center", color="#1f2933", fontsize=8)
    if colorbar_label is not None:
        cbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(colorbar_label)


def _draw_violin_plot(
    ax: plt.Axes,
    data: list[Any],
    labels: list[str],
    facecolor: str,
    edgecolor: str,
) -> None:
    positions = list(range(1, len(data) + 1))
    violin = ax.violinplot(data, positions=positions, widths=0.7, showextrema=False)
    for body in violin["bodies"]:
        body.set_facecolor(facecolor)
        body.set_edgecolor(edgecolor)
        body.set_alpha(0.82)
        body.set_linewidth(1.0)

    for position, values in zip(positions, data):
        series = pd.Series(values).dropna()
        if series.empty:
            continue
        q1, median, q3 = series.quantile([0.25, 0.50, 0.75])
        ax.vlines(position, q1, q3, color=edgecolor, linewidth=4.0, alpha=0.95)
        ax.hlines(median, position - 0.22, position + 0.22, color="#1f2933", linewidth=1.6)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)


def _model_colors() -> dict[str, str]:
    return {
        "faster-whisper": "#1f77b4",
        "WhisperX": "#ff7f0e",
        "wav2vec2": "#d62728",
        "Parakeet": "#9467bd",
    }


def _json_ready_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        clean_row: dict[str, Any] = {}
        for key, value in row.items():
            if pd.isna(value):
                clean_row[key] = None
            elif isinstance(value, (np.integer,)):
                clean_row[key] = int(value)
            elif isinstance(value, (np.floating,)):
                clean_row[key] = float(value)
            else:
                clean_row[key] = value
        records.append(clean_row)
    return records


def main(k_worst: int = 3) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 120,
        }
    )
    all_results = pd.concat([load_model_results(model) for model in MODELS], ignore_index=True)
    success = all_results[all_results["status"] == "success"].copy()
    wav2 = success[success["model"] == "wav2vec2"].copy()
    cross = success[success["is_cross_model_clip"]].copy()
    timing = load_timing_results()
    real_eval = load_real_eval_wer_orc_results()
    real_transfer = build_real_transferability_summary(cross, real_eval)
    dss_wer_tests = analyse_dss_wer_hypothesis_tests(success)
    wav2_wer_surface = fit_wav2vec2_wer_surface(wav2)
    wav2_effect_sizes = compute_effect_sizes(wav2)

    plot_wav2vec2_snr_violin(wav2)
    plot_wav2vec2_heatmap(wav2)
    plot_wav2vec2_wer_vs_overlap(wav2)
    plot_wav2vec2_wer_vs_snr(wav2)
    wav2_delta_vs_ovr = plot_wav2vec2_delta_vs_overlap(wav2)
    wav2_delta_vs_snr = plot_wav2vec2_delta_vs_snr(wav2)
    plot_noise_type(wav2)
    plot_cross_model_violin(cross)
    plot_cross_model_wer_vs_overlap(cross)
    plot_cross_model_wer_vs_snr(cross)
    cross_support = plot_cross_model_scope_support(cross)
    plot_timing_bar(timing)
    plot_timing_tradeoff(cross, timing)
    plot_real_transfer_wer_change(real_transfer)
    worst_wer_examples = get_k_worst_wer_examples(k=k_worst)
    wer_outlier_conditions = summarise_wer_outlier_conditions_by_model(all_results)

    summary = {
        "source_files": [f"WER_results_{model}.json" for model in MODELS],
        "cross_model_clip_regex": CROSS_MODEL_CLIP_REGEX.pattern,
        "wav2vec2_full": summarise(wav2["wer"]),
        "wav2vec2_by_snr": {
            key: summarise(group["wer"])
            for key, group in wav2.groupby("snr_label", dropna=False)
        },
        "wav2vec2_by_overlap": {
            f"{key:.2f}": summarise(group["wer"])
            for key, group in wav2.groupby("overlap_ratio", dropna=False)
        },
        "wav2vec2_by_noise_type": {
            key: summarise(group["wer"])
            for key, group in wav2.groupby("noise_type", dropna=False)
        },
        "wav2vec2_effect_sizes": wav2_effect_sizes,
        "wav2vec2_wer_surface": wav2_wer_surface,
        "wav2vec2_delta_vs_ovr": {
            f"{overlap_ratio:.2f}": {
                snr_label: float(wav2_delta_vs_ovr.loc[overlap_ratio, snr_label])
                for snr_label in SNR_ORDER
            }
            for overlap_ratio in OVR_ORDER
        },
        "wav2vec2_delta_vs_snr": {
            snr_label: {
                f"{overlap_ratio:.2f}": float(wav2_delta_vs_snr.loc[overlap_ratio, snr_label])
                for overlap_ratio in OVR_ORDER
            }
            for snr_label in SNR_ORDER
        },
        "cross_model": {
            model: summarise(group["wer"])
            for model, group in cross.groupby("model", dropna=False)
        },
        "cross_model_metric_means": (
            cross.groupby("model")[["wer", "dss_wer", "orc_wer", "cpwer"]]
            .mean()
            .to_dict()
        ),
        "cross_model_unique_clip_counts": {
            snr_label: {
                f"{overlap_ratio:.2f}": (
                    None
                    if pd.isna(cross_support.loc[snr_label, overlap_ratio])
                    else int(cross_support.loc[snr_label, overlap_ratio])
                )
                for overlap_ratio in OVR_ORDER
            }
            for snr_label in SNR_ORDER
        },
        "timing": {
            row["model"]: {
                "sample_count": int(row["sample_count"]),
                "seconds_per_clip": float(row["seconds_per_clip"]),
                "rtfx": float(row["rtfx"]),
                "total_processing_time_sec": float(row["total_processing_time_sec"]),
            }
            for row in timing.to_dict(orient="records")
        },
        "cross_model_overlap_slice_means": (
            cross[(cross["snr_label"] == "7.4") & (cross["noise_type"] == "T")]
            .groupby(["model", "overlap_ratio"])["wer"]
            .mean()
            .unstack()
            .to_dict()
        ),
        "cross_model_snr_slice_means": (
            cross[(cross["overlap_ratio"] == 0.14) & (cross["noise_type"] == "T")]
            .groupby(["model", "snr_label"])["wer"]
            .mean()
            .unstack()
            .to_dict()
        ),
        "real_transferability": {
            "source_file": str(REAL_EVAL_WER_ORC_PATH),
            "synthetic_source": "Matched 700-clip subset selected by cross_model_clip_regex",
            "real_source": "100 CHiME-6 S01 clips scored in Output/real_eval_wer_orc_results.json",
            "model_stats": _json_ready_records(real_transfer),
        },
        "dss_wer_hypothesis_tests": {
            "difference": "dss_wer - wer",
            "non_overlap_test": "two-sided Monte Carlo paired sign-flip test",
            "overlap_test": "one-sided lower-tail Monte Carlo paired sign-flip test",
            "n_permutations": int(DSS_PERMUTATION_N),
            "bootstrap_resamples_for_ci": int(DSS_PERMUTATION_N),
            "random_seed": int(DSS_RANDOM_SEED),
            "rows": _json_ready_records(dss_wer_tests),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    worst_payload = {
        "k_per_model": int(k_worst),
        "count_total": int(sum(len(items) for items in worst_wer_examples.values())),
        "items_by_model": worst_wer_examples,
    }
    WORST_WER_PATH.write_text(json.dumps(worst_payload, indent=2, sort_keys=True), encoding="utf-8")
    WER_OUTLIER_CONDITIONS_PATH.write_text(
        json.dumps(wer_outlier_conditions, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote figures to {FIGURE_DIR}")
    print(f"Wrote summary to {SUMMARY_PATH}")
    print(f"Wrote worst-WER examples to {WORST_WER_PATH}")
    print(f"Wrote WER outlier condition summary to {WER_OUTLIER_CONDITIONS_PATH}")


if __name__ == "__main__":
    arg_k = int(os.environ.get("WORST_WER_K", "3"))
    main(k_worst=arg_k)
