from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

try:
    import meeteval
except ImportError:
    meeteval = None

from wer_helper import wer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASR_PATH = PROJECT_ROOT / "Output" / "real_asr_transcriptions.json"
DEFAULT_REFERENCE_PATH = PROJECT_ROOT / "Output" / "real_transcript" / "real_transcripts.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "Output" / "real_eval_wer_orc_results.json"


Segment = tuple[str, str]


def _segment_from_any(value: Any) -> Segment | None:
    if isinstance(value, dict):
        speaker = str(value.get("speaker", "unknown"))
        text = value.get("words") or value.get("text") or value.get("transcript")
        if text is None:
            return None
        return speaker, str(text)

    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return str(value[0]), str(value[1])

    return None


def _segments_from_any(values: Any) -> list[Segment]:
    if not isinstance(values, list):
        return []

    segments: list[Segment] = []
    for value in values:
        segment = _segment_from_any(value)
        if segment is not None and segment[1].strip():
            segments.append(segment)
    return segments


def _reference_segments(ref_info: dict[str, Any]) -> list[Segment]:
    reference_segments = _segments_from_any(ref_info.get("reference_segments"))
    if reference_segments:
        return reference_segments
    return _segments_from_any(ref_info.get("transcript"))


def _flatten_text(segments: list[Segment]) -> str:
    return " ".join(text for _, text in segments)


def _speaker_texts(segments: list[Segment]) -> list[str]:
    speaker_order: list[str] = []
    speaker_words: dict[str, list[str]] = {}

    for speaker, text in segments:
        if speaker not in speaker_words:
            speaker_words[speaker] = []
            speaker_order.append(speaker)
        speaker_words[speaker].append(text)

    return [" ".join(speaker_words[speaker]) for speaker in speaker_order if speaker_words[speaker]]


def _safe_wer(reference: str, hypothesis: str) -> float | None:
    if not reference.split():
        return None
    return float(wer(reference, hypothesis))


def _orc_wer(reference_segments: list[Segment], hypothesis_segments: list[Segment]) -> float:
    if meeteval is None:
        raise RuntimeError("ORC-WER requires meeteval. Install with: pip install meeteval")

    return float(
        meeteval.wer.wer.orc.orc_word_error_rate(
            reference=_speaker_texts(reference_segments),
            hypothesis=_speaker_texts(hypothesis_segments),
            reference_sort=False,
            hypothesis_sort=False,
        ).error_rate
    )


def _summary(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None

    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
        "sd": stdev(values) if len(values) > 1 else 0.0,
    }


def evaluate_real_asr(
    asr_path: Path,
    reference_path: Path,
) -> dict[str, Any]:
    references = json.loads(reference_path.read_text(encoding="utf-8"))
    asr_transcriptions = json.loads(asr_path.read_text(encoding="utf-8"))

    by_clip: dict[str, dict[str, Any]] = {}
    per_model_wer: dict[str, list[float]] = {}
    per_model_orc_wer: dict[str, list[float]] = {}

    for clip_id, ref_info in references.items():
        ref_segments = _reference_segments(ref_info)
        if not ref_segments:
            continue

        clip_asr = asr_transcriptions.get(clip_id, {})
        model_transcripts = clip_asr.get("transcript", {}) if isinstance(clip_asr, dict) else {}
        by_clip[clip_id] = {}

        for model_name, raw_hyp_segments in model_transcripts.items():
            hyp_segments = _segments_from_any(raw_hyp_segments)
            if not hyp_segments:
                continue

            wer_score = _safe_wer(_flatten_text(ref_segments), _flatten_text(hyp_segments))
            if wer_score is None:
                continue

            row: dict[str, Any] = {
                "wer": wer_score,
                "hyp_segment_count": len(hyp_segments),
                "ref_segment_count": len(ref_segments),
                "is_segmented_hypothesis": len(hyp_segments) > 1,
            }
            per_model_wer.setdefault(model_name, []).append(wer_score)

            if len(hyp_segments) > 1:
                orc_wer_score = _orc_wer(ref_segments, hyp_segments)
                row["orc_wer"] = orc_wer_score
                per_model_orc_wer.setdefault(model_name, []).append(orc_wer_score)

            by_clip[clip_id][model_name] = row

    summary: dict[str, Any] = {}
    for model_name in sorted(per_model_wer):
        model_summary: dict[str, Any] = {
            "wer": _summary(per_model_wer.get(model_name, [])),
        }
        orc_summary = _summary(per_model_orc_wer.get(model_name, []))
        if orc_summary is not None:
            model_summary["orc_wer"] = orc_summary
        summary[model_name] = model_summary

    return {
        "metrics": {
            "wer": "Computed for every hypothesis by flattening reference and hypothesis segments.",
            "orc_wer": "Computed only when a hypothesis contains multiple segments.",
        },
        "source_files": {
            "asr_transcriptions": str(asr_path),
            "references": str(reference_path),
        },
        "summary": summary,
        "by_clip": by_clip,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate cached real-data ASR transcriptions with WER for all outputs and ORC-WER for segmented outputs.",
    )
    parser.add_argument("--asr-json", type=Path, default=DEFAULT_ASR_PATH)
    parser.add_argument("--reference-json", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = evaluate_real_asr(
        asr_path=args.asr_json.resolve(),
        reference_path=args.reference_json.resolve(),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote real-data WER/ORC-WER evaluation to {args.output_json}")


if __name__ == "__main__":
    main()
