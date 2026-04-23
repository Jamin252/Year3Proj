import json
import csv
import ast
import sys
import argparse
import importlib
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

try:
    meeteval = importlib.import_module("meeteval")
except ImportError:
    meeteval = None

MRS_LOOKAHEAD = 16
MRS_BEAM_WIDTH = 64
MRS_HEURISTIC_WEIGHT = 0.4
MRS_MAX_EXPANSIONS = 160000

# Add Code directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from wer_helper import cpWER, wer
from mrs_beam_wer import mrs_wer_beam_2chain


VALID_METRICS = {"wer", "mrs_wer", "cpwer", "orc_wer"}


def load_asr_transcriptions(asr_json_path: str = "ASR_transcriptions.json") -> Dict:
    with open(asr_json_path, 'r') as f:
        res = json.load(f)
    return res


def load_manifest(manifest_path: str = "Output/manifest.csv") -> Dict:
    manifest = {}
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip_id = row['clip_id']
            transcript_str = row['transcript']
            try:
                transcript = _parse_transcript_literal(transcript_str)
                manifest[clip_id] = transcript
            except (ValueError, SyntaxError) as e:
                print(f"Warning: Could not parse transcript for {clip_id}: {e}")
                manifest[clip_id] = []
    return manifest


def _normalize_csv_doubled_quote_pairs(text: str) -> str:
    """Convert CSV-doubled quote pairs around fields into valid Python quotes.

    Example:
        ('spk', ""THAT'S FINE"", 0.0, 1.0)
    becomes:
        ('spk', "THAT'S FINE", 0.0, 1.0)
    """
    return re.sub(r'""(.*?)""', r'"\1"', text)


def _quote_unquoted_transcript_fields(text: str) -> str:
    """Add quotes around unquoted transcript fields in tuple literals.

    This is a recovery path for malformed inputs such as:
    ('spk', NO BUT THAT'S FINE, 1.0, 2.0)
    where the transcript field lost its surrounding quotes before parsing.
    """

    def _is_quoted(value: str) -> bool:
        value = value.strip()
        return (len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'})

    def _escape_for_double_quotes(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

    def _fix_4tuple(match: re.Match) -> str:
        spk, transcript, start_t, end_t = match.groups()
        transcript = transcript.strip()
        if _is_quoted(transcript):
            return match.group(0)
        transcript = _escape_for_double_quotes(transcript)
        return f"({spk}, \"{transcript}\", {start_t}, {end_t})"

    def _fix_2tuple(match: re.Match) -> str:
        spk, transcript = match.groups()
        transcript = transcript.strip()
        if _is_quoted(transcript):
            return match.group(0)
        transcript = _escape_for_double_quotes(transcript)
        return f"({spk}, \"{transcript}\")"

    spk_pat = r"(?:'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")"
    four_tuple_pat = re.compile(
        rf"\(\s*({spk_pat})\s*,\s*(.*?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\)"
    )
    # For 2-tuples, only repair cases where the second field has no extra top-level
    # commas; this avoids accidentally swallowing timestamp fields from 4-tuples.
    two_tuple_pat = re.compile(rf"\(\s*({spk_pat})\s*,\s*([^,\)]*?)\s*\)")

    fixed = four_tuple_pat.sub(_fix_4tuple, text)
    fixed = two_tuple_pat.sub(_fix_2tuple, fixed)
    return fixed


def _parse_transcript_literal(raw_value: object) -> List[Tuple[str, str, float, float]]:
    """Parse transcript literals while tolerating CSV-escaped doubled quotes.

    Some rows may contain doubled double-quotes ("") when CSV escaping is
    preserved by an upstream read/write path. We try the raw value first, then
    a normalized variant with doubled quotes collapsed.
    """
    if raw_value is None:
        return []

    text = str(raw_value).strip()
    if not text:
        return []

    candidates = [text]
    if '""' in text:
        candidates.append(_normalize_csv_doubled_quote_pairs(text))
        candidates.append(text.replace('""', '"'))

    last_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            parsed = ast.literal_eval(candidate)
            return parsed if isinstance(parsed, list) else []
        except (ValueError, SyntaxError) as err:
            last_error = err

    if last_error is not None:
        repaired = _quote_unquoted_transcript_fields(candidates[-1])
        try:
            parsed = ast.literal_eval(repaired)
            return parsed if isinstance(parsed, list) else []
        except (ValueError, SyntaxError):
            raise last_error
    return []


def get_hypothesis_transcription(
    asr_data: Dict,
    clip_id: str,
    model_name: str
) -> Optional[List[Tuple[str, str]]]:
    if clip_id not in asr_data:
        # print(f"Warning: {clip_id} not found in ASR data")
        return None
    
    if model_name not in asr_data[clip_id].get('transcript', {}):
        # print(f"Warning: {model_name} not found for {clip_id}")
        return None
    
    hyp_data = asr_data[clip_id]['transcript'][model_name]
    
    # Convert to list of tuples if needed
    if isinstance(hyp_data, list) and len(hyp_data) > 0:
        if isinstance(hyp_data[0], list):
            return [tuple(item) for item in hyp_data]
        return hyp_data
    
    return hyp_data


def get_reference_transcription(
    manifest_data: Dict,
    clip_id: str
) -> Optional[List[Tuple[str, str, float, float]]]:
    if clip_id not in manifest_data:
        print(f"Warning: {clip_id} not found in manifest")
        return None
    
    return manifest_data[clip_id]


def _split_into_two_chains(segments: List[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
    if isinstance(segments, str):
        segments = ast.literal_eval(segments)

    speaker_order: List[str] = []
    speaker_words: Dict[str, List[str]] = {}
    for seg in segments:
        if len(seg) < 2:
            continue
        spk = seg[0]
        if spk not in speaker_words:
            speaker_words[spk] = []
            speaker_order.append(spk)
        speaker_words[spk].extend(_normalize_words_from_text(seg[1]))

    if not speaker_order:
        return [], []
    if len(speaker_order) == 1:
        return speaker_words[speaker_order[0]], []

    first_speaker = speaker_order[0]
    second_speaker = speaker_order[1]
    chain_a = speaker_words[first_speaker]
    chain_b = speaker_words[second_speaker]
    return chain_a, chain_b


def _normalize_words_from_text(text: str) -> List[str]:
    clean_text = re.sub(r"[^\w\s]", "", str(text).lower())
    return [tok for tok in clean_text.split() if tok]


def _flatten_transcript_words(segments: List[Tuple[str, str]]) -> List[str]:
    words: List[str] = []
    for segment in segments:
        if len(segment) < 2:
            continue
        words.extend(_normalize_words_from_text(segment[1]))
    return words


def _split_into_speaker_texts(segments: List[Tuple[str, str]]) -> List[str]:
    if isinstance(segments, str):
        segments = ast.literal_eval(segments)

    speaker_order: List[str] = []
    speaker_words: Dict[str, List[str]] = {}
    for seg in segments:
        if len(seg) < 2:
            continue
        spk = seg[0]
        if spk not in speaker_words:
            speaker_words[spk] = []
            speaker_order.append(spk)
        speaker_words[spk].extend(_normalize_words_from_text(seg[1]))

    return [" ".join(speaker_words[spk]) for spk in speaker_order if speaker_words[spk]]


def _normalize_metric_name(raw_metric: str) -> str:
    normalized = raw_metric.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "mrs": "mrs_wer",
        "mrswer": "mrs_wer",
        "orc": "orc_wer",
        "orcwer": "orc_wer",
    }
    return aliases.get(normalized, normalized)


def _parse_metric_selection(raw_metric: Optional[str]) -> Optional[Set[str]]:
    if raw_metric is None:
        return None

    tokens = [tok for tok in raw_metric.split(",") if tok.strip()]
    if not tokens:
        return None

    selected: Set[str] = set()
    for token in tokens:
        metric = _normalize_metric_name(token)
        if metric in {"all", "*"}:
            return None
        if metric not in VALID_METRICS:
            raise ValueError(
                f"Unsupported metric '{token}'. Choose from: wer, mrs_wer, cpwer, orc_wer"
            )
        selected.add(metric)

    return selected if selected else None


def evaluate_wer_for_clip(
    clip_id: str,
    model_name: str,
    asr_data: Dict,
    manifest_data: Dict,
    normalize_ref: bool = True,
    selected_metrics: Optional[Set[str]] = None,
) -> Dict:
    # Get hypothesis
    print(f"Evaluating {clip_id} with model {model_name}...")
    hyp = get_hypothesis_transcription(asr_data, clip_id, model_name)
    if hyp is None:
        print(f"Skipped: Hypothesis not found for {clip_id} and model {model_name}")
        return {
            'clip_id': clip_id,
            'model_name': model_name,
            'status': 'error',
            'error': 'Hypothesis not found'
        }
    
    # Get reference
    ref = get_reference_transcription(manifest_data, clip_id)
    if ref is None:
        return {
            'clip_id': clip_id,
            'model_name': model_name,
            'status': 'error',
            'error': 'Reference not found'
        }
    
    # Normalize reference to (speaker, text) format if it contains timing info
    if normalize_ref and len(ref) > 0 and len(ref[0]) > 2:
        ref_normalized = [(spk, text) for spk, text, *_ in ref]
    else:
        ref_normalized = ref
    
    # Detect if transcription is segmented (multiple segments per speaker/overall)
    is_segmented = len(hyp) > 1 if isinstance(hyp, list) else False
    
    result = {
        'clip_id': clip_id,
        'model_name': model_name,
        'status': 'success',
        'ref_segments': len(ref_normalized),
        'hyp_segments': len(hyp),
        'is_segmented': is_segmented,
        'metrics': {}
    }

    ref_words = _flatten_transcript_words(ref_normalized)
    hyp_words = _flatten_transcript_words(hyp)
    ref_text = ' '.join(ref_words)
    hyp_text = ' '.join(hyp_words)
    ref_chain_a, ref_chain_b = _split_into_two_chains(ref_normalized)
    ref_speaker_texts = _split_into_speaker_texts(ref_normalized)
    hyp_speaker_texts = _split_into_speaker_texts(hyp)
    
    try:
        metric_set = selected_metrics or VALID_METRICS

        if 'wer' in metric_set:
            result['metrics']['wer'] = wer(ref_text, hyp_text)

        if 'mrs_wer' in metric_set:
            beam_result = mrs_wer_beam_2chain(
                ref_chain_a,
                ref_chain_b,
                hyp_words,
                beam_width=MRS_BEAM_WIDTH,
                heuristic_weight=MRS_HEURISTIC_WEIGHT,
                normalize=True,
                return_alignment=False,
                lookahead=MRS_LOOKAHEAD,
                max_expansions=MRS_MAX_EXPANSIONS,
            )
            result['metrics']['mrs_wer'] = beam_result['wer']

        if 'cpwer' in metric_set:
            result['metrics']['cpwer'] = cpWER(ref_normalized, hyp)

        if 'orc_wer' in metric_set:
            if meeteval is None:
                raise RuntimeError(
                    "ORC-WER requested but meeteval is not installed. Install with: pip install meeteval"
                )
            orc_result = meeteval.wer.wer.orc.orc_word_error_rate(
                reference=ref_speaker_texts,
                hypothesis=hyp_speaker_texts,
            )
            result['metrics']['orc_wer'] = float(orc_result.error_rate)

        if {'wer', 'mrs_wer'}.issubset(metric_set):
            result['wer_method'] = 'both'
        elif 'wer' in metric_set:
            result['wer_method'] = 'wer_only'
        elif 'mrs_wer' in metric_set:
            result['wer_method'] = 'mrs_wer_only'
        else:
            result['wer_method'] = 'n/a'

        if is_segmented:
            result['metric_type'] = 'segmented'
        else:
            result['metric_type'] = 'not segmented'
        
        return result
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        result['metrics'] = {}
        return result


def evaluate_wer_batch(
    clip_ids: List[str],
    model_name: str,
    asr_data: Dict,
    manifest_data: Dict,
    verbose: bool = True,
    selected_metrics: Optional[Set[str]] = None,
) -> List[Dict]:
    results = []
    
    for i, clip_id in enumerate(clip_ids):
        result = evaluate_wer_for_clip(
            clip_id,
            model_name,
            asr_data,
            manifest_data,
            selected_metrics=selected_metrics,
        )
        results.append(result)
        
        if verbose and (i + 1) % max(1, len(clip_ids) // 10) == 0:
            print(f"Processed {i + 1}/{len(clip_ids)} clips")
    
    return results


def get_model_names(asr_data: Dict) -> List[str]:
    models = set()
    for clip_data in asr_data.values():
        if 'transcript' in clip_data:
            models.update(clip_data['transcript'].keys())
    return sorted(list(models))


def evaluate_all_models(
    clip_ids: List[str],
    asr_data: Dict,
    manifest_data: Dict,
    verbose: bool = True,
    selected_metrics: Optional[Set[str]] = None,
) -> Dict[str, List[Dict]]:
    models = get_model_names(asr_data)
    results_by_model = {}
    
    for model in models:
        if verbose:
            print(f"\nEvaluating model: {model}")
        results_by_model[model] = evaluate_wer_batch(
            clip_ids,
            model,
            asr_data,
            manifest_data,
            verbose,
            selected_metrics=selected_metrics,
        )
    
    return results_by_model


def compute_summary_statistics(results: List[Dict]) -> Dict:
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'error']
    
    # Separate results so segmented clips can contribute to cpWER and ORC-WER.
    wer_results = [r for r in successful if 'wer' in r.get('metrics', {})]
    mrs_wer_results = [r for r in successful if 'mrs_wer' in r.get('metrics', {})]
    cpwer_results = [r for r in successful if 'cpwer' in r.get('metrics', {})]
    orc_wer_results = [r for r in successful if 'orc_wer' in r.get('metrics', {})]
    wer_segmented_results = [
        r for r in successful if 'wer' in r.get('metrics', {}) and r.get('is_segmented') is True
    ]
    wer_nonsegmented_results = [
        r for r in successful if 'wer' in r.get('metrics', {}) and r.get('is_segmented') is False
    ]
    
    def compute_stats_for_metric(metric_results, metric_name):
        """Compute stats for a specific metric."""
        if not metric_results:
            return {f'{metric_name}_count': 0}
        
        scores = [r['metrics'].get(metric_name) for r in metric_results 
                 if metric_name in r.get('metrics', {})]
        scores = [s for s in scores if s is not None]
        
        if not scores:
            return {f'{metric_name}_count': 0}
        
        import statistics
        return {
            f'{metric_name}_count': len(scores),
            f'{metric_name}_mean': statistics.mean(scores),
            f'{metric_name}_median': statistics.median(scores),
            f'{metric_name}_min': min(scores),
            f'{metric_name}_max': max(scores),
            f'{metric_name}_std': statistics.stdev(scores) if len(scores) > 1 else 0.0
        }
    
    stats = {
        'total': len(results),
        'successful': len(successful),
        'failed': len(failed),
        'wer_count': len(wer_results),
        'mrs_wer_count': len(mrs_wer_results),
        'cpwer_count': len(cpwer_results),
        'orc_wer_count': len(orc_wer_results),
        'wer_segmented_count': len(wer_segmented_results),
        'wer_nonsegmented_count': len(wer_nonsegmented_results),
    }
    
    # Add WER statistics for each clip type.
    wer_nonsegmented_stats = compute_stats_for_metric(wer_nonsegmented_results, 'wer')
    wer_segmented_stats = compute_stats_for_metric(wer_segmented_results, 'wer')
    stats.update({k: v for k, v in wer_nonsegmented_stats.items() if k != 'wer_count'})
    stats.update({f'wer_segmented_{k[len("wer_"):]}' : v for k, v in wer_segmented_stats.items() if k != 'wer_count'})

    # Add MRS-WER statistics
    stats.update(compute_stats_for_metric(mrs_wer_results, 'mrs_wer'))
    
    # Add cpWER statistics
    stats.update(compute_stats_for_metric(cpwer_results, 'cpwer'))

    # Add ORC-WER statistics
    stats.update(compute_stats_for_metric(orc_wer_results, 'orc_wer'))
    
    return stats


def save_wer_results_by_clip(
    results: List[Dict],
    output_dir: str = "WER_Results"
) -> Dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    for result in results:
        clip_id = result.get('clip_id', 'unknown')
        model_name = result.get('model_name', 'unknown')
        
        # Create subdirectory for model if it doesn't exist
        model_dir = output_path / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual clip result
        file_path = model_dir / f"{clip_id}_wer.json"
        
        with open(file_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        saved_files[clip_id] = file_path
    
    return saved_files


def save_wer_results_batch(
    results: List[Dict],
    output_file: str = "WER_results_batch.json"
) -> Path:
    output_path = Path(output_file)
    
    # Add summary statistics to results
    stats = compute_summary_statistics(results)
    
    output_data = {
        'summary': stats,
        'results': results,
        'timestamp': __import__('datetime').datetime.now().isoformat()
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    return output_path


def save_wer_results_by_model(
    results_by_model: Dict[str, List[Dict]],
    output_dir: str = "WER_Results_by_Model"
) -> Dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    for model_name, results in results_by_model.items():
        stats = compute_summary_statistics(results)
        
        output_data = {
            'model': model_name,
            'summary': stats,
            'results': results,
            'timestamp': __import__('datetime').datetime.now().isoformat()
        }
        
        file_path = output_path / f"{model_name}_wer_results.json"
        
        with open(file_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        saved_files[model_name] = file_path
    
    return saved_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate WER for ASR models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python wer_evaluation.py                          # Evaluate all models\n"
               "  python wer_evaluation.py --only-model faster-whisper   # Evaluate only one model\n"
               "  python wer_evaluation.py --model faster-whisper        # Backward-compatible alias\n"
               "  python wer_evaluation.py --list                   # List available models"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Backward-compatible alias for --only-model."
    )
    parser.add_argument(
        "--only-model",
        type=str,
        default=None,
        help="Evaluate only the specified model (e.g., 'faster-whisper'). If not specified, evaluates all models."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        help=(
            "Optional metric selector. Use one or more comma-separated values from "
            "wer,mrs_wer,cpwer,orc_wer. Examples: --metric wer or --metric cpwer,orc_wer"
        ),
    )
    
    args = parser.parse_args()
    
    # Load data
    asr_data = load_asr_transcriptions()
    manifest_data = load_manifest()

    try:
        selected_metrics = _parse_metric_selection(args.metric)
    except ValueError as metric_err:
        print(f"Error: {metric_err}")
        sys.exit(1)
    
    # Get list of all clip IDs from manifest
    clip_ids = list(manifest_data.keys())  # Evaluate all clips
    
    # Get available models
    models = get_model_names(asr_data)
    print(f"Available models: {models}")
    
    # If --list flag is used, just print models and exit
    if args.list:
        print("\nAvailable models:")
        for model in models:
            print(f"  - {model}")
        sys.exit(0)
    
    # Determine which models to evaluate
    if args.model and args.only_model and args.model != args.only_model:
        print(f"Error: --model ({args.model}) and --only-model ({args.only_model}) disagree. Use only one.")
        sys.exit(1)

    selected_model = args.only_model or args.model

    if selected_model:
        # Evaluate specific model
        if selected_model not in models:
            print(f"Error: Model '{selected_model}' not found in available models: {models}")
            sys.exit(1)
        models_to_eval = [selected_model]
    else:
        # Evaluate all models
        models_to_eval = models
    
    # Evaluate selected model(s)
    results_by_model = {}
    for model_name in models_to_eval:
        print(f"\nEvaluating {model_name} on {len(clip_ids)} clips...")
        if selected_metrics is None:
            print("Metrics: all (wer, mrs_wer, cpwer, orc_wer)")
        else:
            print(f"Metrics: {sorted(selected_metrics)}")
        results = evaluate_wer_batch(
            clip_ids,
            model_name,
            asr_data,
            manifest_data,
            verbose=False,
            selected_metrics=selected_metrics,
        )
        results_by_model[model_name] = results
        
        # Save results by clip in individual JSON files
        print("Saving individual clip results...")
        saved_files = save_wer_results_by_clip(results, output_dir="WER_Results")
        print(f"✓ Saved {len(saved_files)} clip result files")
        
        # Save batch results to single JSON file
        print("Saving batch results...")
        batch_file = save_wer_results_batch(results, output_file=f"WER_results_{model_name}.json")
        print(f"✓ Saved batch results to {batch_file}")
        
        # Print summary with separated metrics
        stats = compute_summary_statistics(results)
        print(f"\n{'='*60}")
        print(f"Summary Statistics for {model_name}")
        print(f"{'='*60}")
        print(f"Total clips: {stats['total']}")
        print(f"Successful: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
        print(f"WER clips: {stats.get('wer_count', 0)}")
        print(f"MRS-WER clips: {stats.get('mrs_wer_count', 0)}")
        print(f"cpWER clips: {stats.get('cpwer_count', 0)}")
        print(f"ORC-WER clips: {stats.get('orc_wer_count', 0)}")
        print()

        if stats.get('mrs_wer_count', 0) > 0:
            print(f"All clips (MRS-WER) - {stats['mrs_wer_count']} clips:")
            for key in ['mrs_wer_mean', 'mrs_wer_median', 'mrs_wer_min', 'mrs_wer_max', 'mrs_wer_std']:
                if key in stats:
                    print(f"  {key}: {stats[key]:.4f}")
            print()
        
        if stats.get('wer_nonsegmented_count', 0) > 0:
            print(f"Non-Segmented (WER) - {stats['wer_nonsegmented_count']} clips:")
            for key in ['wer_mean', 'wer_median', 'wer_min', 'wer_max', 'wer_std']:
                if key in stats:
                    print(f"  {key}: {stats[key]:.4f}")
        
        if stats.get('cpwer_count', 0) > 0:
            print()
            print(f"cpWER - {stats['cpwer_count']} clips:")
            for key in ['cpwer_mean', 'cpwer_median', 'cpwer_min', 'cpwer_max', 'cpwer_std']:
                if key in stats:
                    print(f"  {key}: {stats[key]:.4f}")

        if stats.get('orc_wer_count', 0) > 0:
            print()
            print(f"Segmented (ORC-WER) - {stats['orc_wer_count']} clips:")
            for key in ['orc_wer_mean', 'orc_wer_median', 'orc_wer_min', 'orc_wer_max', 'orc_wer_std']:
                if key in stats:
                    print(f"  {key}: {stats[key]:.4f}")

        if stats.get('wer_segmented_count', 0) > 0:
            print()
            print(f"Segmented (WER on concatenated hyp) - {stats['wer_segmented_count']} clips:")
            for key in ['wer_segmented_mean', 'wer_segmented_median', 'wer_segmented_min', 'wer_segmented_max', 'wer_segmented_std']:
                if key in stats:
                    print(f"  {key}: {stats[key]:.4f}")

    # Always write the grouped-by-model JSON files for the models that were evaluated.
    print("\nSaving results by model...")
    model_files = save_wer_results_by_model(results_by_model, output_dir="WER_Results_by_Model")
    print(f"✓ Saved results for {len(model_files)} model(s):")
    for model_name, file_path in model_files.items():
        print(f"  - {model_name}: {file_path}")
