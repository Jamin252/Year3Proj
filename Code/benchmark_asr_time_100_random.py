from __future__ import annotations

import ast
import gc
import json
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
import whisperx
from faster_whisper import WhisperModel
from transformers import pipeline

from asr_helper import clear_gpu_cache, load_mixture_audio, load_mixture_meta
from helper_class import MixtureMeta


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "Output" / "manifest.csv"
RESULT_PATH = PROJECT_ROOT / "timings_100_random.json"

SAMPLE_SIZE = 100
RANDOM_SEED = 1234

def _parse_transcript(raw_value: Any) -> List[tuple]:
    if pd.isna(raw_value) or raw_value == "":
        return []

    text = str(raw_value).strip()
    candidates = [text]
    if '""' in text:
        candidates.append(text.replace('""', '"'))

    for candidate in candidates:
        try:
            parsed = ast.literal_eval(candidate)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            continue
    return []


def _build_meta(row: pd.Series) -> MixtureMeta:
    return MixtureMeta(
        clip_id=str(row["clip_id"]),
        audio_path=str(row["audio_path"]),
        transcript=_parse_transcript(row.get("transcript")),
        overlap_ratio_target=float(row["overlap_ratio_target"]),
        overlap_ratio_actual=float(row["overlap_ratio_actual"]),
        max_speakers=int(row["max_speakers"]),
        snr_db=None if pd.isna(row["snr_db"]) else float(row["snr_db"]),
        noise_type=None if pd.isna(row["noise_type"]) else str(row["noise_type"]),
        overlap_mask_path=str(row["overlap_mask_path"]),
        source_files=[],
        noise_files=[],
    )


def _sample_metas(sample_size: int = SAMPLE_SIZE, seed: int = RANDOM_SEED) -> List[MixtureMeta]:
    manifest = load_mixture_meta(MANIFEST_PATH)
    if len(manifest) < sample_size:
        raise ValueError(f"Only {len(manifest)} rows matched the benchmark filter; need at least {sample_size}.")
    sampled = manifest.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    print(f"Sampled {len(sampled)} clips from {MANIFEST_PATH}")
    print("First 5 sampled clip IDs:")
    for clip_id in sampled["clip_id"].head(5).tolist():
        print(f"- {clip_id}")
    return [_build_meta(row) for _, row in sampled.iterrows()]


def _audio_duration_sec(audio_path: str) -> float:
    with wave.open(audio_path, "rb") as wav_handle:
        frames = wav_handle.getnframes()
        rate = wav_handle.getframerate()
    return float(frames) / float(rate) if rate else 0.0


def _total_audio_duration_sec(metas: List[MixtureMeta]) -> float:
    return float(sum(_audio_duration_sec(meta.audio_path) for meta in metas))


def _time_faster_whisper(metas: List[MixtureMeta]) -> float:
    print("[1/4] Starting faster-whisper timing...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "int8" if torch.cuda.is_available() else "float32"
    model = WhisperModel("large-v3", device=device, compute_type=compute_type)
    clear_gpu_cache(force_gc=True)
    start = time.perf_counter()
    for meta in metas:
        segments, _ = model.transcribe(Path(meta.audio_path))
        list(segments)
    duration = time.perf_counter() - start
    print(f"[1/4] faster-whisper finished in {duration:.3f}s")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    clear_gpu_cache(force_gc=True)
    return duration


def _time_wav2vec2(metas: List[MixtureMeta]) -> float:
    print("[2/4] Starting wav2vec2 timing...")
    device = 0 if torch.cuda.is_available() else -1
    asr = pipeline(
        "automatic-speech-recognition",
        model="facebook/wav2vec2-large-960h",
        device=device,
    )
    clear_gpu_cache(force_gc=True)
    start = time.perf_counter()
    with torch.inference_mode():
        for meta in metas:
            audio, sr = load_mixture_audio(Path(meta.audio_path))
            asr({"array": audio, "sampling_rate": sr})
    duration = time.perf_counter() - start
    print(f"[2/4] wav2vec2 finished in {duration:.3f}s")
    del asr
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    clear_gpu_cache(force_gc=True)
    return duration


def _time_whisperx(metas: List[MixtureMeta]) -> float:
    print("[3/4] Starting whisperx timing...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "int8" if torch.cuda.is_available() else "float32"
    
    try:
        model = whisperx.load_model("large-v2", device, compute_type=compute_type)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[3/4] GPU OOM on whisperx, falling back to CPU...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            device = "cpu"
            compute_type = "float32"
            model = whisperx.load_model("large-v2", device, compute_type=compute_type)
        else:
            raise
    
    clear_gpu_cache(force_gc=True)
    start = time.perf_counter()
    for meta in metas:
        audio = whisperx.load_audio(meta.audio_path)
        result = model.transcribe(audio, batch_size=1)
        [segment["text"] for segment in result["segments"]]
    duration = time.perf_counter() - start
    print(f"[3/4] whisperx finished in {duration:.3f}s")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    clear_gpu_cache(force_gc=True)
    return duration


def _time_parakeet(metas: List[MixtureMeta]) -> float:
    print("[4/4] Starting parakeet timing...")
    temp_input = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    temp_output = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    temp_input_path = Path(temp_input.name)
    temp_output_path = Path(temp_output.name)
    temp_input.close()
    temp_output.close()

    try:
        with open(temp_input_path, "w", encoding="utf-8") as handle:
            for meta in metas:
                handle.write(json.dumps({"audio_filepath": meta.audio_path, "text": ""}, ensure_ascii=False) + "\n")

        start = time.perf_counter()
        print(f"[4/4] Running Parakeet batch of {len(metas)} clips...")
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "NeMo" / "examples" / "asr" / "asr_chunked_inference" / "rnnt" / "speech_to_text_streaming_infer_rnnt.py"),
                "pretrained_name=nvidia/parakeet-tdt-0.6b-v3",
                "model_path=null",
                f"dataset_manifest={temp_input_path}",
                f"output_filename={temp_output_path}",
                "right_context_secs=2.0",
                "chunk_secs=10",
                "left_context_secs=10.0",
                "batch_size=1",
                "cuda=0",
                "decoding.greedy.use_cuda_graph_decoder=False",
                "clean_groundtruth_text=False",
            ],
            check=True,
            cwd=PROJECT_ROOT,
        )
        duration = time.perf_counter() - start
        print(f"[4/4] parakeet finished in {duration:.3f}s")
        return duration
    finally:
        temp_input_path.unlink(missing_ok=True)
        temp_output_path.unlink(missing_ok=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        clear_gpu_cache(force_gc=True)


def main(model_name: str = None) -> None:
    """
    Run benchmark for specific model or all models.
    
    Args:
        model_name: Specific model to run ('faster-whisper', 'wav2vec2', 'whisperx', 'parakeet')
                   If None, runs all models.
    """
    metas = _sample_metas()
    total_audio_duration_sec = _total_audio_duration_sec(metas)
    print(f"Total sample audio duration: {total_audio_duration_sec:.3f}s")
    
    timings = {
        "faster-whisper": _time_faster_whisper,
        "wav2vec2": _time_wav2vec2,
        "whisperx": _time_whisperx,
        "parakeet": _time_parakeet,
    }
    
    # Load existing results if file exists
    if RESULT_PATH.exists():
        with open(RESULT_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        existing_results = payload.get("results", [])
        sample_size = payload.get("sample_size", len(metas))
    else:
        existing_results = []
        sample_size = len(metas)

    # Backfill legacy timing records so partial reruns still produce complete schema.
    for result in existing_results:
        processing_time = result.get("total_processing_time_sec", result.get("duration_sec", 0.0))
        result["total_processing_time_sec"] = processing_time
        if processing_time and processing_time > 0:
            result["rtfx"] = total_audio_duration_sec / processing_time
    
    # Determine which models to run
    if model_name:
        if model_name not in timings:
            print(f"Error: Unknown model '{model_name}'. Choices: {', '.join(timings.keys())}")
            sys.exit(1)
        models_to_run = [(model_name, timings[model_name])]
    else:
        # Run all models not yet in results
        existing_models = {r["model"] for r in existing_results}
        models_to_run = [(name, timer) for name, timer in timings.items() if name not in existing_models]
    
    if not models_to_run:
        print("All models already have timing results!")
        print(f"Results: {RESULT_PATH}")
        return
    
    # Run each model
    for name, timer in models_to_run:
        print(f"\nTiming model: {name}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        duration_sec = timer(metas)
        total_processing_time_sec = duration_sec
        rtfx = total_audio_duration_sec / total_processing_time_sec if total_processing_time_sec > 0 else 0.0
        result = {
            "model": name,
            "sample_count": len(metas),
            "duration_sec": duration_sec,
            "total_processing_time_sec": total_processing_time_sec,
            "sec_per_audio": duration_sec / len(metas),
            "rtfx": rtfx,
            "sample_clip_ids": [meta.clip_id for meta in metas],
        }
        existing_results.append(result)
        print(
            f"{name}: total_processing_time={total_processing_time_sec:.3f}s, "
            f"sec_per_audio={result['sec_per_audio']:.4f}, rtfx={rtfx:.4f}x"
        )
    
    # Save updated results
    payload = {
        "sample_size": sample_size,
        "seed": RANDOM_SEED,
        "sample_filter": None,
        "total_audio_duration_sec": total_audio_duration_sec,
        "results": existing_results,
    }

    with open(RESULT_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(f"\nSaved benchmark timings to {RESULT_PATH}")
    print(f"Completed models: {[r['model'] for r in existing_results]}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(model_name=sys.argv[1])
    else:
        main()