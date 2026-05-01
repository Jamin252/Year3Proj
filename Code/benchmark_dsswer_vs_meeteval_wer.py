from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

from meeteval.wer.wer.siso import siso_word_error_rate

SEED = 1234

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "Code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from mrs_beam_wer import _parse_transcript_literal, mrs_wer_beam_2chain


DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "Output" / "manifest.csv"
DEFAULT_ASR_PATH = PROJECT_ROOT / "ASR_transcriptions.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "Documentation" / "dsswer_vs_meeteval_wer_benchmark.json"

DEFAULT_BEAM_WIDTH = 64
DEFAULT_HEURISTIC_WEIGHT = 0.4
DEFAULT_LOOKAHEAD = 16
DEFAULT_MAX_EXPANSIONS = 160_000


def normalize_words(text: object) -> list[str]:
    clean_text = re.sub(r"[^\w\s]", "", str(text).lower())
    return [token for token in clean_text.split() if token]


def load_manifest(path: Path) -> dict[str, list[tuple[Any, ...]]]:
    manifest: dict[str, list[tuple[Any, ...]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            clip_id = (row.get("clip_id") or "").strip()
            if not clip_id:
                continue
            manifest[clip_id] = _parse_transcript_literal(row.get("transcript"))
    return manifest


def load_asr_transcriptions(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_hypothesis_words(hypothesis: object) -> list[str]:
    if isinstance(hypothesis, str):
        return normalize_words(hypothesis)
    if not isinstance(hypothesis, list):
        return []

    words: list[str] = []
    for segment in hypothesis:
        if isinstance(segment, str):
            words.extend(normalize_words(segment))
        elif isinstance(segment, (list, tuple)) and len(segment) >= 2:
            words.extend(normalize_words(segment[1]))
        elif isinstance(segment, dict):
            text = segment.get("text") or segment.get("transcript") or segment.get("hypothesis")
            if text:
                words.extend(normalize_words(text))
    return words


def chronological_reference_words(ref_segments: list[tuple[Any, ...]]) -> list[str]:
    words: list[str] = []
    for segment in ref_segments:
        if len(segment) >= 2:
            words.extend(normalize_words(segment[1]))
    return words


def reference_two_chains(ref_segments: list[tuple[Any, ...]]) -> tuple[list[str], list[str]]:
    speaker_order: list[str] = []
    speaker_words: dict[str, list[str]] = {}

    for segment in ref_segments:
        if len(segment) < 2:
            continue
        speaker = str(segment[0])
        if speaker not in speaker_words:
            speaker_words[speaker] = []
            speaker_order.append(speaker)
        speaker_words[speaker].extend(normalize_words(segment[1]))

    if not speaker_order:
        return [], []
    if len(speaker_order) == 1:
        return speaker_words[speaker_order[0]], []
    return speaker_words[speaker_order[0]], speaker_words[speaker_order[1]]


def get_model_hypothesis(asr_payload: dict[str, Any], clip_id: str, model: str) -> object | None:
    clip_payload = asr_payload.get(clip_id)
    if not isinstance(clip_payload, dict):
        return None
    transcript_payload = clip_payload.get("transcript")
    if not isinstance(transcript_payload, dict):
        return None
    return transcript_payload.get(model)


def build_examples(
    manifest: dict[str, list[tuple[Any, ...]]],
    asr_payload: dict[str, Any],
    model: str,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for clip_id, ref_segments in manifest.items():
        if not ref_segments:
            continue
        hypothesis = get_model_hypothesis(asr_payload, clip_id, model)
        hyp_words = flatten_hypothesis_words(hypothesis)
        if not hyp_words:
            continue

        ref_words = chronological_reference_words(ref_segments)
        chain_a, chain_b = reference_two_chains(ref_segments)
        if not ref_words or not (chain_a or chain_b):
            continue

        examples.append(
            {
                "clip_id": clip_id,
                "ref_words": ref_words,
                "ref_chain_a": chain_a,
                "ref_chain_b": chain_b,
                "hyp_words": hyp_words,
            }
        )
    return examples


def benchmark_dsswer(
    examples: list[dict[str, Any]],
    beam_width: int,
    heuristic_weight: float,
    lookahead: int,
    max_expansions: int,
) -> tuple[dict[str, float], dict[str, float]]:
    scores: dict[str, float] = {}
    timings: dict[str, float] = {}

    for example in examples:
        start = time.perf_counter()
        result = mrs_wer_beam_2chain(
            example["ref_chain_a"],
            example["ref_chain_b"],
            example["hyp_words"],
            beam_width=beam_width,
            heuristic_weight=heuristic_weight,
            normalize=True,
            return_alignment=False,
            lookahead=lookahead,
            max_expansions=max_expansions,
        )
        elapsed = time.perf_counter() - start
        scores[example["clip_id"]] = float(result["wer"])
        timings[example["clip_id"]] = float(elapsed)

    return scores, timings


def benchmark_meeteval_wer(examples: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, float]]:
    scores: dict[str, float] = {}
    timings: dict[str, float] = {}

    for example in examples:
        ref_text = " ".join(example["ref_words"])
        hyp_text = " ".join(example["hyp_words"])
        start = time.perf_counter()
        result = siso_word_error_rate(ref_text, hyp_text)
        elapsed = time.perf_counter() - start
        scores[example["clip_id"]] = float(result.error_rate)
        timings[example["clip_id"]] = float(elapsed)

    return scores, timings


def summarize_timing(total_time_s: float, n_clips: int, ref_tokens: int, compared_tokens: int) -> dict[str, float]:
    return {
        "total_time_s": float(total_time_s),
        "time_per_clip_s": float(total_time_s / n_clips) if n_clips else 0.0,
        "time_per_ref_token_s": float(total_time_s / ref_tokens) if ref_tokens else 0.0,
        "time_per_compared_token_s": float(total_time_s / compared_tokens) if compared_tokens else 0.0,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_manifest(args.manifest)
    asr_payload = load_asr_transcriptions(args.asr_json)
    examples = build_examples(manifest, asr_payload, args.model)

    if len(examples) < args.sample_size:
        raise ValueError(
            f"Only {len(examples)} manifest clips have a usable '{args.model}' hypothesis; "
            f"cannot sample {args.sample_size}."
        )

    rng = random.Random(SEED)
    sampled_examples = rng.sample(examples, args.sample_size)
    sampled_examples.sort(key=lambda item: item["clip_id"])

    ref_token_count = sum(len(example["ref_words"]) for example in sampled_examples)
    hyp_token_count = sum(len(example["hyp_words"]) for example in sampled_examples)
    compared_token_count = ref_token_count + hyp_token_count

    dss_scores, dss_timings = benchmark_dsswer(
        sampled_examples,
        beam_width=args.beam_width,
        heuristic_weight=args.heuristic_weight,
        lookahead=args.lookahead,
        max_expansions=args.max_expansions,
    )
    meeteval_scores, meeteval_timings = benchmark_meeteval_wer(sampled_examples)

    dss_total_time_s = sum(dss_timings.values())
    meeteval_total_time_s = sum(meeteval_timings.values())

    per_clip_rows = []
    for example in sampled_examples:
        clip_id = example["clip_id"]
        ref_tokens = len(example["ref_words"])
        hyp_tokens = len(example["hyp_words"])
        per_clip_rows.append(
            {
                "clip_id": clip_id,
                "ref_tokens": ref_tokens,
                "hyp_tokens": hyp_tokens,
                "compared_tokens": ref_tokens + hyp_tokens,
                "dsswer": dss_scores[clip_id],
                "meeteval_wer": meeteval_scores[clip_id],
                "dsswer_time_s": dss_timings[clip_id],
                "meeteval_wer_time_s": meeteval_timings[clip_id],
            }
        )

    n_clips = len(sampled_examples)
    payload = {
        "sample": {
            "seed": int(args.seed),
            "requested_sample_size": int(args.sample_size),
            "actual_sample_size": int(n_clips),
            "model": args.model,
            "manifest_path": str(args.manifest),
            "asr_json_path": str(args.asr_json),
            "eligible_manifest_clip_count": int(len(examples)),
        },
        "token_counts": {
            "ref_tokens": int(ref_token_count),
            "hyp_tokens": int(hyp_token_count),
            "compared_tokens_ref_plus_hyp": int(compared_token_count),
            "avg_ref_tokens_per_clip": float(ref_token_count / n_clips) if n_clips else 0.0,
            "avg_hyp_tokens_per_clip": float(hyp_token_count / n_clips) if n_clips else 0.0,
            "avg_compared_tokens_per_clip": float(compared_token_count / n_clips) if n_clips else 0.0,
        },
        "dsswer_parameters": {
            "beam_width": int(args.beam_width),
            "heuristic_weight": float(args.heuristic_weight),
            "lookahead": int(args.lookahead),
            "max_expansions": int(args.max_expansions),
            "normalize": True,
        },
        "summary": {
            "dsswer": summarize_timing(dss_total_time_s, n_clips, ref_token_count, compared_token_count),
            "meeteval_wer": summarize_timing(meeteval_total_time_s, n_clips, ref_token_count, compared_token_count),
            "dsswer_to_meeteval_total_time_ratio": (
                float(dss_total_time_s / meeteval_total_time_s) if meeteval_total_time_s else None
            ),
        },
        "per_clip": per_clip_rows,
    }
    return payload


def print_report(payload: dict[str, Any]) -> None:
    sample = payload["sample"]
    tokens = payload["token_counts"]
    dss = payload["summary"]["dsswer"]
    meeteval = payload["summary"]["meeteval_wer"]
    ratio = payload["summary"]["dsswer_to_meeteval_total_time_ratio"]

    print("DSS-WER vs meeteval WER benchmark")
    print(f"Model: {sample['model']}")
    print(f"Seed: {sample['seed']}")
    print(f"Clips: {sample['actual_sample_size']} / eligible {sample['eligible_manifest_clip_count']}")
    print()
    print("Token counts")
    print(f"  Reference tokens: {tokens['ref_tokens']}")
    print(f"  Hypothesis tokens: {tokens['hyp_tokens']}")
    print(f"  Ref+hyp tokens: {tokens['compared_tokens_ref_plus_hyp']}")
    print(f"  Avg reference tokens per clip: {tokens['avg_ref_tokens_per_clip']:.2f}")
    print(f"  Avg hypothesis tokens per clip: {tokens['avg_hyp_tokens_per_clip']:.2f}")
    print(f"  Avg ref+hyp tokens per clip: {tokens['avg_compared_tokens_per_clip']:.2f}")
    print()
    print("Timing")
    print(
        "  DSS-WER:       "
        f"total={dss['total_time_s']:.6f}s, "
        f"per_clip={dss['time_per_clip_s']:.6f}s, "
        f"per_ref_token={dss['time_per_ref_token_s']:.9f}s, "
        f"per_ref+hyp_token={dss['time_per_compared_token_s']:.9f}s"
    )
    print(
        "  meeteval WER:  "
        f"total={meeteval['total_time_s']:.6f}s, "
        f"per_clip={meeteval['time_per_clip_s']:.6f}s, "
        f"per_ref_token={meeteval['time_per_ref_token_s']:.9f}s, "
        f"per_ref+hyp_token={meeteval['time_per_compared_token_s']:.9f}s"
    )
    if ratio is not None:
        print(f"  Total-time ratio DSS-WER / meeteval WER: {ratio:.2f}x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample manifest clip IDs and benchmark DSS-WER against "
            "meeteval SISO WER on cached ASR hypotheses."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--asr-json", type=Path, default=DEFAULT_ASR_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model", default="wav2vec2")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument("--beam-width", type=int, default=DEFAULT_BEAM_WIDTH)
    parser.add_argument("--heuristic-weight", type=float, default=DEFAULT_HEURISTIC_WEIGHT)
    parser.add_argument("--lookahead", type=int, default=DEFAULT_LOOKAHEAD)
    parser.add_argument("--max-expansions", type=int, default=DEFAULT_MAX_EXPANSIONS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_benchmark(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print_report(payload)
    print()
    print(f"Wrote JSON report to {args.output_json}")


if __name__ == "__main__":
    main()
