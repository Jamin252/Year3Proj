import json
import csv
import ast
import sys
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add Code directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from wer_helper import cpWER, wer
from mrs_beam_wer import mrs_wer_beam_2chain


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
                transcript = ast.literal_eval(transcript_str)
                manifest[clip_id] = transcript
            except (ValueError, SyntaxError) as e:
                print(f"Warning: Could not parse transcript for {clip_id}: {e}")
                manifest[clip_id] = []
    return manifest


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


def _is_no_overlap_clip(clip_id: str) -> bool:
    parts = clip_id.split("_")
    return len(parts) >= 3 and parts[2] == "0.00"


def evaluate_wer_for_clip(
    clip_id: str,
    model_name: str,
    asr_data: Dict,
    manifest_data: Dict,
    normalize_ref: bool = True
) -> Dict:
    # Get hypothesis
    print(f"Evaluating {clip_id} with model {model_name}...")
    hyp = get_hypothesis_transcription(asr_data, clip_id, model_name)
    if hyp is None:
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
    use_plain_wer = _is_no_overlap_clip(clip_id)
    
    try:
        if is_segmented:
            # Use cpWER for segmented transcriptions
            result['metrics']['cpwer'] = cpWER(ref_normalized, hyp)
            # Original concatenated-WER evaluation kept for reference.
            # result['metrics']['wer'] = wer(ref_text, hyp_text)

            if use_plain_wer:
                result['metrics']['wer'] = wer(ref_text, hyp_text)
                result['wer_method'] = 'wer'
            else:
                ref_chain_a, ref_chain_b = _split_into_two_chains(ref_normalized)
                beam_result = mrs_wer_beam_2chain(
                    ref_chain_a,
                    ref_chain_b,
                    hyp_words,
                    beam_width=64,
                    heuristic_weight=1.0,
                    normalize=True,
                    return_alignment=False,
                )
                result['metrics']['wer'] = beam_result['wer']
                result['wer_method'] = 'mrs'
            result['metric_type'] = 'segmented'  # Both cpWER and concatenated WER
        else:
            # Original concatenated-WER evaluation kept for reference.
            # result['metrics']['wer'] = wer(ref_text, hyp_text)

            if use_plain_wer:
                result['metrics']['wer'] = wer(ref_text, hyp_text)
                result['wer_method'] = 'wer'
            else:
                ref_chain_a, ref_chain_b = _split_into_two_chains(ref_normalized)
                beam_result = mrs_wer_beam_2chain(
                    ref_chain_a,
                    ref_chain_b,
                    hyp_words,
                    beam_width=64,
                    heuristic_weight=1.0,
                    normalize=True,
                    return_alignment=False,
                )
                result['metrics']['wer'] = beam_result['wer']
                result['wer_method'] = 'mrs'
            result['metric_type'] = 'wer'  # Simple WER for single segment
        
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
    verbose: bool = True
) -> List[Dict]:
    results = []
    
    for i, clip_id in enumerate(clip_ids):
        result = evaluate_wer_for_clip(clip_id, model_name, asr_data, manifest_data)
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
    verbose: bool = True
) -> Dict[str, List[Dict]]:
    models = get_model_names(asr_data)
    results_by_model = {}
    
    for model in models:
        if verbose:
            print(f"\nEvaluating model: {model}")
        results_by_model[model] = evaluate_wer_batch(
            clip_ids, model, asr_data, manifest_data, verbose
        )
    
    return results_by_model


def compute_summary_statistics(results: List[Dict]) -> Dict:
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'error']
    
    # Separate results so segmented clips can contribute to both WER and cpWER.
    wer_results = [r for r in successful if 'wer' in r.get('metrics', {})]
    cpwer_results = [r for r in successful if 'cpwer' in r.get('metrics', {})]
    wer_segmented_results = [r for r in successful if 'wer' in r.get('metrics', {}) and 'cpwer' in r.get('metrics', {})]
    wer_nonsegmented_results = [r for r in successful if 'wer' in r.get('metrics', {}) and 'cpwer' not in r.get('metrics', {})]
    
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
        'cpwer_count': len(cpwer_results),
        'wer_segmented_count': len(wer_segmented_results),
        'wer_nonsegmented_count': len(wer_nonsegmented_results),
    }
    
    # Add WER statistics for each clip type.
    wer_nonsegmented_stats = compute_stats_for_metric(wer_nonsegmented_results, 'wer')
    wer_segmented_stats = compute_stats_for_metric(wer_segmented_results, 'wer')
    stats.update({k: v for k, v in wer_nonsegmented_stats.items() if k != 'wer_count'})
    stats.update({f'wer_segmented_{k[len("wer_"):]}' : v for k, v in wer_segmented_stats.items() if k != 'wer_count'})
    
    # Add cpWER statistics
    stats.update(compute_stats_for_metric(cpwer_results, 'cpwer'))
    
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
               "  python wer_evaluation.py --model faster-whisper   # Evaluate specific model\n"
               "  python wer_evaluation.py --list                   # List available models"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model name to evaluate (e.g., 'faster-whisper'). If not specified, evaluates all models."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit"
    )
    
    args = parser.parse_args()
    
    # Load data
    asr_data = load_asr_transcriptions()
    manifest_data = load_manifest()
    
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
    if args.model:
        # Evaluate specific model
        if args.model not in models:
            print(f"Error: Model '{args.model}' not found in available models: {models}")
            sys.exit(1)
        models_to_eval = [args.model]
    else:
        # Evaluate all models
        models_to_eval = models
    
    # Evaluate selected model(s)
    results_by_model = {}
    for model_name in models_to_eval:
        print(f"\nEvaluating {model_name} on {len(clip_ids)} clips...")
        results = evaluate_wer_batch(clip_ids, model_name, asr_data, manifest_data, verbose=False)
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
        print(f"cpWER clips: {stats.get('cpwer_count', 0)}")
        print()
        
        if stats.get('wer_nonsegmented_count', 0) > 0:
            print(f"Non-Segmented (WER) - {stats['wer_nonsegmented_count']} clips:")
            for key in ['wer_mean', 'wer_median', 'wer_min', 'wer_max', 'wer_std']:
                if key in stats:
                    print(f"  {key}: {stats[key]:.4f}")
        
        if stats.get('wer_segmented_count', 0) > 0:
            print()
            print(f"Segmented (cpWER) - {stats['wer_segmented_count']} clips:")
            for key in ['cpwer_mean', 'cpwer_median', 'cpwer_min', 'cpwer_max', 'cpwer_std']:
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
