from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import soundfile as sf
import torch
from transformers import pipeline

from asr_helper import (
	transcribe_faster_whisper,
	transcribe_parakeet,
	transcribe_wav2vec2,
	transcribe_whisperx,
)
from helper_class import MixtureMeta, MixtureTranscription
from wer_helper import cpWER, wer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
S01_JSON = PROJECT_ROOT / "Data" / "CHiME6_eval" / "S01.json"
S01_WAV = PROJECT_ROOT / "Data" / "CHiME6_eval" / "S01.wav"

REAL_AUDIO_DIR = PROJECT_ROOT / "Output" / "real"
REAL_TRANSCRIPT_DIR = PROJECT_ROOT / "Output" / "real_transcript"
REAL_ASR_PATH = PROJECT_ROOT / "Output" / "real_asr_transcriptions.json"
REAL_EVAL_PATH = PROJECT_ROOT / "Output" / "real_eval_results.json"
ASR_BATCH_SIZE = 10


def hhmmss_to_seconds(value: str) -> float:
	h, m, s = value.split(":")
	return int(h) * 3600 + int(m) * 60 + float(s)


def load_s01_utterances(path: Path) -> List[Dict[str, Any]]:
	with open(path, "r", encoding="utf-8") as f:
		raw = json.load(f)

	utterances: List[Dict[str, Any]] = []
	for row in raw:
		start_sec = hhmmss_to_seconds(row["start_time"])
		end_sec = hhmmss_to_seconds(row["end_time"])
		if end_sec <= start_sec:
			continue
		utterances.append(
			{
				"start": start_sec,
				"end": end_sec,
				"speaker": row["speaker"],
				"words": row["words"],
				"session_id": row.get("session_id", "S01"),
				"ref": row.get("ref"),
				"location": row.get("location"),
			}
		)

	utterances.sort(key=lambda x: (x["start"], x["end"]))
	return utterances


def _active_crossing_count(utterances: List[Dict[str, Any]], t: float) -> int:
	return sum(1 for u in utterances if u["start"] < t < u["end"])


def _candidate_starts(utterances: List[Dict[str, Any]]) -> List[float]:
	starts = sorted({u["start"] for u in utterances})
	out: List[float] = []
	for t in starts:
		has_start = any(u["start"] == t for u in utterances)
		if has_start and _active_crossing_count(utterances, t) == 0:
			out.append(t)
	return out


def _candidate_ends(utterances: List[Dict[str, Any]]) -> List[float]:
	ends = sorted({u["end"] for u in utterances})
	out: List[float] = []
	for t in ends:
		has_end = any(u["end"] == t for u in utterances)
		if has_end and _active_crossing_count(utterances, t) == 0:
			out.append(t)
	return out


def _utterances_in_window(
	utterances: List[Dict[str, Any]], start_t: float, end_t: float
) -> List[Dict[str, Any]]:
	# Keep only fully-contained transcript items, never partial items.
	return [u for u in utterances if u["start"] >= start_t and u["end"] <= end_t]


def build_real_segments(
	utterances: List[Dict[str, Any]],
	n_clips: int = 100,
	min_duration_sec: float = 60.0,
) -> List[Dict[str, Any]]:
	print(
		f"[segment] Building up to {n_clips} clips (min_duration={min_duration_sec:.1f}s) from {len(utterances)} utterances..."
	)
	starts = _candidate_starts(utterances)
	ends = _candidate_ends(utterances)

	clips: List[Dict[str, Any]] = []
	prev_end = 0.0

	for start_t in starts:
		if len(clips) >= n_clips:
			break
		if start_t < prev_end:
			continue

		valid_ends = [e for e in ends if e > start_t and (e - start_t) >= min_duration_sec]
		if not valid_ends:
			continue

		end_t = valid_ends[0]
		clip_utts = _utterances_in_window(utterances, start_t, end_t)
		if not clip_utts:
			continue

		clip_idx = len(clips) + 1
		clip_id = f"real_{clip_idx:04d}"
		clips.append(
			{
				"clip_id": clip_id,
				"start": start_t,
				"end": end_t,
				"duration": end_t - start_t,
				"transcript": [
					{
						"speaker": u["speaker"],
						"words": u["words"],
						"start": u["start"],
						"end": u["end"],
					}
					for u in clip_utts
				],
			}
		)
		prev_end = end_t
		if len(clips) % 10 == 0 or len(clips) == n_clips:
			print(f"[segment] Built {len(clips)}/{n_clips} clips")

	if len(clips) < n_clips:
		raise RuntimeError(
			f"Only built {len(clips)} clips with duration >= {min_duration_sec}s using full-utterance boundaries."
		)

	print(f"[segment] Done. Built {len(clips)} clips.")

	return clips


def export_real_audio_and_transcripts(
	wav_path: Path,
	clips: List[Dict[str, Any]],
	audio_dir: Path,
	transcript_dir: Path,
) -> Dict[str, Dict[str, Any]]:
	print(f"[export] Writing audio clips to {audio_dir}")
	audio_dir.mkdir(parents=True, exist_ok=True)
	transcript_dir.mkdir(parents=True, exist_ok=True)

	transcript_json_path = transcript_dir / "real_transcripts.json"
	existing_refs: Dict[str, Dict[str, Any]] = {}
	if transcript_json_path.exists() and transcript_json_path.stat().st_size > 0:
		with open(transcript_json_path, "r", encoding="utf-8") as f:
			try:
				existing_refs = json.load(f)
			except json.JSONDecodeError:
				print("[export] Existing transcript JSON is invalid; rebuilding entries.")
				existing_refs = {}

	refs: Dict[str, Dict[str, Any]] = dict(existing_refs)
	clips_by_id = {c["clip_id"]: c for c in clips}
	existing_wav_ids = {
		p.stem
		for p in audio_dir.glob("real_*.wav")
		if p.stem in clips_by_id
	}
	if len(existing_wav_ids) == len(clips):
		print(f"[export] Found all {len(clips)} clip WAVs already present; skipping audio regeneration.")
	else:
		print(
			f"[export] Found {len(existing_wav_ids)}/{len(clips)} existing WAV clips; generating the missing {len(clips) - len(existing_wav_ids)}."
		)

	info = sf.info(str(wav_path))
	sr = info.samplerate

	updated_transcript_entries = 0
	generated_audio = 0
	kept_audio = 0

	with sf.SoundFile(str(wav_path), "r") as sfile:
		total = len(clips)
		for idx, clip in enumerate(clips, start=1):
			clip_id = clip["clip_id"]
			out_wav = audio_dir / f"{clip_id}.wav"
			if out_wav.exists():
				kept_audio += 1
			else:
				start_sample = int(round(clip["start"] * sr))
				end_sample = int(round(clip["end"] * sr))
				sfile.seek(start_sample)
				audio = sfile.read(end_sample - start_sample, dtype="float32")
				sf.write(str(out_wav), audio, sr)
				generated_audio += 1

			ref_segments = [(t["speaker"], t["words"]) for t in clip["transcript"]]
			new_entry = {
				"clip_id": clip_id,
				"audio_path": str(out_wav),
				"start": clip["start"],
				"end": clip["end"],
				"duration": clip["duration"],
				"transcript": clip["transcript"],
				"reference_segments": ref_segments,
			}
			if clip_id not in refs:
				refs[clip_id] = new_entry
				updated_transcript_entries += 1
			elif not refs[clip_id].get("reference_segments"):
				refs[clip_id] = new_entry
				updated_transcript_entries += 1
			else:
				# Keep existing entry by default, but ensure audio path is current.
				refs[clip_id]["audio_path"] = str(out_wav)

			if idx % 10 == 0 or idx == total:
				print(f"[export] Processed {idx}/{total} clips")

	with open(transcript_json_path, "w", encoding="utf-8") as f:
		json.dump(refs, f, indent=2, ensure_ascii=False)

	print(
		f"[export] Audio summary: kept={kept_audio}, generated={generated_audio}, total={len(clips)}"
	)
	print(
		f"[export] Transcript summary: added_or_repaired={updated_transcript_entries}, total_entries={len(refs)}"
	)
	print(f"[export] Saved transcripts to {transcript_json_path}")

	# Return only the requested clip subset for downstream ASR/eval.
	return {clip["clip_id"]: refs[clip["clip_id"]] for clip in clips}


def build_metas_from_refs(refs: Dict[str, Dict[str, Any]]) -> List[MixtureMeta]:
	metas: List[MixtureMeta] = []
	for clip_id, item in refs.items():
		transcript = [(x["speaker"], x["words"]) for x in item["transcript"]]
		metas.append(
			MixtureMeta(
				clip_id=clip_id,
				audio_path=item["audio_path"],
				transcript=transcript,
				overlap_ratio_target=0.0,
				overlap_ratio_actual=0.0,
				max_speakers=4,
				snr_db=None,
				noise_type=None,
				overlap_mask_path="",
				source_files=[],
				noise_files=[],
			)
		)
	metas.sort(key=lambda x: x.clip_id)
	return metas


def run_models(
	metas: List[MixtureMeta],
	model_names: List[str],
	dic: Optional[Dict[str, MixtureTranscription]] = None,
	asr_out_path: Optional[Path] = None,
) -> Dict[str, MixtureTranscription]:
	def _run_model_in_batches(model_label: str, fn, kwargs: Optional[Dict[str, Any]] = None) -> None:
		kwargs = kwargs or {}
		pending = [
			m
			for m in metas
			if not (
				dic.get(m.clip_id) is not None
				and model_label in dic[m.clip_id].transcript
			)
		]
		skipped = len(metas) - len(pending)
		total = len(pending)
		if total == 0:
			print(f"[asr:{model_label}] All clips already transcribed for this model; skipping.")
			return
		print(
			f"[asr:{model_label}] Starting on {total} pending clips (skipped={skipped}, batch_size={ASR_BATCH_SIZE})"
		)
		start = time.perf_counter()
		for i in range(0, total, ASR_BATCH_SIZE):
			batch = pending[i : i + ASR_BATCH_SIZE]
			fn(batch, dic, ind=i, model_name=model_label, **kwargs)
			print(f"[asr:{model_label}] Completed {min(i + ASR_BATCH_SIZE, total)}/{total}")
			if asr_out_path is not None:
				save_asr_outputs(dic, asr_out_path)
				print(f"[asr:{model_label}] Checkpoint saved")
		dur = time.perf_counter() - start
		print(f"[asr:{model_label}] Done in {dur:.1f}s")

	if dic is None:
		dic = {}

	if "faster-whisper" in model_names:
		_run_model_in_batches("faster-whisper", transcribe_faster_whisper)

	if "wav2vec2" in model_names:
		device = 0 if torch.cuda.is_available() else -1
		wav_asr = pipeline(
			"automatic-speech-recognition",
			model="facebook/wav2vec2-large-960h",
			device=device,
		)
		_run_model_in_batches("wav2vec2", transcribe_wav2vec2, {"asr": wav_asr})

	if "parakeet" in model_names:
		_run_model_in_batches("parakeet", transcribe_parakeet)

	if "whisperx" in model_names:
		_run_model_in_batches("whisperx", transcribe_whisperx)

	return dic


def _flatten_text(segments: List[Tuple[str, str]]) -> str:
	return " ".join(seg[1] for seg in segments)


def evaluate_model_outputs(
	refs: Dict[str, Dict[str, Any]],
	hyps: Dict[str, MixtureTranscription],
) -> Dict[str, Any]:
	print(f"[eval] Evaluating {len(refs)} clips...")
	per_clip: Dict[str, Dict[str, Any]] = {}
	per_model_scores: Dict[str, List[float]] = {}

	total = len(refs)
	processed = 0
	for clip_id, ref_info in refs.items():
		ref_segments: List[Tuple[str, str]] = ref_info["reference_segments"]
		per_clip[clip_id] = {}

		hyp_obj = hyps.get(clip_id)
		if hyp_obj is None:
			continue

		for model_name, hyp_segments in hyp_obj.transcript.items():
			if not hyp_segments:
				continue

			hyp_is_segmented = len(hyp_segments) > 1
			if hyp_is_segmented:
				score = cpWER(ref_segments, hyp_segments)
				metric = "cpwer"
			else:
				score = wer(_flatten_text(ref_segments), _flatten_text(hyp_segments))
				metric = "wer"

			per_clip[clip_id][model_name] = {
				"metric": metric,
				"score": score,
				"hyp_segment_count": len(hyp_segments),
				"ref_segment_count": len(ref_segments),
			}
			per_model_scores.setdefault(model_name, []).append(score)

		processed += 1
		if processed % 10 == 0 or processed == total:
			print(f"[eval] Processed {processed}/{total} clips")

	summary: Dict[str, Any] = {}
	for model_name, scores in per_model_scores.items():
		if not scores:
			continue
		summary[model_name] = {
			"count": len(scores),
			"mean": sum(scores) / len(scores),
			"min": min(scores),
			"max": max(scores),
		}

	return {
		"summary": summary,
		"by_clip": per_clip,
	}


def save_asr_outputs(dic: Dict[str, MixtureTranscription], out_path: Path) -> None:
	payload = {k: asdict(v) for k, v in dic.items()}
	with open(out_path, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2, ensure_ascii=False)


def load_asr_outputs(out_path: Path) -> Dict[str, MixtureTranscription]:
	if not out_path.exists() or out_path.stat().st_size == 0:
		return {}
	with open(out_path, "r", encoding="utf-8") as f:
		try:
			raw = json.load(f)
		except json.JSONDecodeError:
			print(f"[asr] Warning: invalid JSON in {out_path}; starting from empty cache.")
			return {}

	dic: Dict[str, MixtureTranscription] = {}
	for clip_id, item in raw.items():
		transcript = item.get("transcript", {})
		normalized: Dict[str, List[Tuple[str, str]]] = {}
		for model_name, segs in transcript.items():
			if isinstance(segs, list):
				normalized[model_name] = [tuple(seg) for seg in segs]
		dic[clip_id] = MixtureTranscription(clip_id=clip_id, transcript=normalized)
	return dic


def main() -> None:
	parser = argparse.ArgumentParser(description="Segment CHiME6 S01 and evaluate ASR models.")
	parser.add_argument("--num-clips", type=int, default=100)
	parser.add_argument("--min-duration", type=float, default=60.0)
	parser.add_argument(
		"--models",
		type=str,
		default="faster-whisper,wav2vec2,parakeet,whisperx",
		help="Comma-separated list: faster-whisper,wav2vec2,parakeet,whisperx",
	)
	parser.add_argument("--skip-asr", action="store_true")
	args = parser.parse_args()
	print(f"[main] Starting pipeline with models={args.models}")

	utterances = load_s01_utterances(S01_JSON)
	clips = build_real_segments(
		utterances,
		n_clips=args.num_clips,
		min_duration_sec=args.min_duration,
	)

	refs = export_real_audio_and_transcripts(
		S01_WAV,
		clips,
		REAL_AUDIO_DIR,
		REAL_TRANSCRIPT_DIR,
	)
	print(f"Exported {len(refs)} clips to {REAL_AUDIO_DIR}")
	print(f"Saved references to {REAL_TRANSCRIPT_DIR / 'real_transcripts.json'}")

	if args.skip_asr:
		print("[main] --skip-asr enabled, stopping after segmentation/export.")
		return

	metas = build_metas_from_refs(refs)
	models = [m.strip() for m in args.models.split(",") if m.strip()]
	loaded_dic = load_asr_outputs(REAL_ASR_PATH)
	print(f"[asr] Loaded cache for {len(loaded_dic)} clips from {REAL_ASR_PATH}")

	dic = run_models(metas, models, dic=loaded_dic, asr_out_path=REAL_ASR_PATH)
	save_asr_outputs(dic, REAL_ASR_PATH)
	print(f"Saved ASR outputs to {REAL_ASR_PATH}")

	eval_payload = evaluate_model_outputs(refs, dic)
	with open(REAL_EVAL_PATH, "w", encoding="utf-8") as f:
		json.dump(eval_payload, f, indent=2, ensure_ascii=False)
	print(f"Saved evaluation to {REAL_EVAL_PATH}")


if __name__ == "__main__":
	main()
