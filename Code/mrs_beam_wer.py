from __future__ import annotations

import ast
from functools import lru_cache
import json
import re
import random
import string
import time
from dataclasses import dataclass
from math import inf
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from wer_helper import wer


def _strip_punctuation(word: str) -> str:
    """Remove all punctuation from a word for comparison."""
    return word.translate(str.maketrans('', '', string.punctuation))


@dataclass(frozen=True)
class BeamState:
    """
    State for 2-chain overlap-aware edit distance.

    i: number of words consumed from chain_a
    j: number of words consumed from chain_b
    t: number of words consumed from hypothesis
    """
    i: int
    j: int
    t: int


def _normalize_words(words: List[str], normalize: bool) -> List[str]:
    if not normalize:
        return list(words)
    return [_strip_punctuation(w.lower()) for w in words]


def _bow_window_lb(
    rem_a: List[str],
    rem_b: List[str],
    rem_h: List[str],
    window: int,
) -> int:
    """
    Lower bound on edit distance based on bag-of-words mismatch in the next window of words.
    """
    
    if window <= 0:
        return 0

    ref_window = rem_a[:window] + rem_b[:window]
    hyp_window = rem_h[:window]

    counts: Dict[str, int] = {}
    for w in ref_window:
        counts[w] = counts.get(w, 0) + 1
    for w in hyp_window:
        counts[w] = counts.get(w, 0) - 1

    l1 = sum(abs(v) for v in counts.values())
    return (l1 + 1) // 2


def _adaptive_beam_size(
    ranked: List[Tuple[float, int, BeamState]],
    base_beam_width: int,
    max_beam_width: int,
    ambiguity_margin: float,
) -> int:
    """Adaptively increase beam width if many candidates are close to the best score, indicating ambiguity."""
    
    if not ranked:
        return base_beam_width

    best_score = ranked[0][0]
    close = 0
    limit = min(len(ranked), max_beam_width * 2)
    for k in range(limit):
        if ranked[k][0] <= best_score + ambiguity_margin:
            close += 1

    if close >= max(4, base_beam_width // 2):
        return max_beam_width
    return base_beam_width


def _select_stratified_beam(
    ranked: List[Tuple[float, int, BeamState]],
    target_beam_width: int,
    bucket_stride: int = 1,
) -> List[BeamState]:
    if len(ranked) <= target_beam_width:
        return [st for _, _, st in ranked]

    buckets: Dict[int, List[Tuple[float, int, BeamState]]] = {}
    for item in ranked:
        st = item[2]
        key = (st.i - st.j) // max(1, bucket_stride)
        buckets.setdefault(key, []).append(item)

    ordered_keys = sorted(buckets.keys(), key=lambda k: buckets[k][0][0])

    selected: List[BeamState] = []
    ptr = 0
    while len(selected) < target_beam_width:
        progressed = False
        for key in ordered_keys:
            bucket = buckets[key]
            if ptr < len(bucket):
                selected.append(bucket[ptr][2])
                progressed = True
                if len(selected) >= target_beam_width:
                    break
        if not progressed:
            break
        ptr += 1

    return selected


def mrs_wer_beam_2chain(
    chain_a: List[str],
    chain_b: List[str],
    hyp: List[str],
    beam_width: int = 64,
    heuristic_weight: float = 1.0,
    normalize: bool = True,
    return_alignment: bool = False,
    max_expansions: int = 250000,
    lookahead: int = 20,
    lookahead_window: Optional[int] = None,
    adaptive_beam: bool = True,
    max_beam_width: Optional[int] = None,
    ambiguity_margin: float = 1.0,
    stratified_beam: bool = True,
    bucket_stride: int = 1,
) -> Dict[str, object]:
    """Compute overlap-aware WER for two reference speaker chains using beam search.

    Algorithm summary:
    - The reference is represented as two ordered chains (chain_a, chain_b).
    - A search state is (i, j, t): consumed words from chain_a, chain_b, and
      hypothesis.
    - Each expansion applies one edit-like operation:
      match/substitute against A or B, insert in hypothesis, delete from A or B.
    - Candidate states are ranked by: exact_cost + heuristic_weight * heuristic,
      where the heuristic combines a length lower bound and a local bag-of-words
      mismatch lower bound.
    - Beam pruning keeps only the best states (optionally adaptive and stratified)
      to control runtime.
    - The solver runs two passes:
      1) front-first (left-to-right)
      2) end-first (reverse all streams, then map alignment back)
      and returns the better result.

    Pseudocode:
        function MRS_WER_BEAM_2CHAIN(chain_a, chain_b, hyp):
            init front_result = RUN_PASS(direction="front")
            init tail_result  = RUN_PASS(direction="tail")
            return argmin_distance_then_wer(front_result, tail_result)

        function RUN_PASS(direction):
            if direction == "tail": reverse(chain_a, chain_b, hyp)
            start = (0, 0, 0)
            best_cost[start] = 0
            beam = [start]

            while beam not empty and expansions < max_expansions:
                next_candidates = {}
                for each state (i, j, t) in beam:
                    generate valid successors via:
                        match_a/sub_a, match_b/sub_b, insert, delete_a, delete_b
                    relax successor cost and keep best backpointer

                rank each candidate by:
                    f(state) = g(state) + heuristic_weight * h(state)
                pick next beam (adaptive width + optional stratified buckets)

            final = (len(chain_a), len(chain_b), len(hyp))
            reconstruct alignment from backpointers if requested
            if direction == "tail": reverse reconstructed alignment
            return metrics and alignment
    """
    if beam_width <= 0:
        raise ValueError("beam_width must be > 0")
    if max_expansions <= 0:
        raise ValueError("max_expansions must be > 0")
    if heuristic_weight < 0:
        raise ValueError("heuristic_weight must be >= 0")

    if max_beam_width is None:
        max_beam_width = 2 * beam_width

    effective_lookahead = lookahead_window if lookahead_window is not None else lookahead

    def run_single_pass(direction_mode: str, pass_beam_width: int, pass_max_beam_width: int) -> Dict[str, object]:
        if direction_mode not in {"front", "tail"}:
            raise ValueError("direction_mode must be 'front' or 'tail'")

        if direction_mode == "front":
            work_chain_a = list(chain_a)
            work_chain_b = list(chain_b)
            work_hyp = list(hyp)
        else:
            work_chain_a = list(reversed(chain_a))
            work_chain_b = list(reversed(chain_b))
            work_hyp = list(reversed(hyp))

        m = len(work_chain_a)
        n = len(work_chain_b)
        T = len(work_hyp)
        n_ref = m + n

        norm_chain_a = _normalize_words(work_chain_a, normalize)
        norm_chain_b = _normalize_words(work_chain_b, normalize)
        norm_hyp = _normalize_words(work_hyp, normalize)

        @lru_cache(maxsize=None)
        def heuristic(i: int, j: int, t: int) -> float:
            """Lower-is-better ranking heuristic."""
            rem_ref = (m - i) + (n - j)
            rem_hyp = T - t

            length_lb = abs(rem_ref - rem_hyp)
            bow_lb = _bow_window_lb(
                norm_chain_a[i:],
                norm_chain_b[j:],
                norm_hyp[t:],
                window=effective_lookahead,
            )
            return float(max(length_lb, bow_lb))

        # Match always highest priority, then substitution, then insert/delete.
        def op_priority(op_name: str) -> int:
            if op_name.startswith("match"):
                return 0
            if op_name.startswith("sub"):
                return 1
            return 2

        best_cost: Dict[BeamState, int] = {BeamState(0, 0, 0): 0}
        
        # Value: (parent state, operation) for backtracking the best path to each state.
        best_backptr: Dict[BeamState, Tuple[Optional[BeamState], Tuple[str, str, Optional[str], Optional[str]]]] = {}

        beam: List[BeamState] = [BeamState(0, 0, 0)]
        expansions = 0

        while beam and expansions < max_expansions:
            # Key: next state; Value: (cost, priority, parent state, operation)
            next_candidates: Dict[
                BeamState,
                Tuple[int, int, Optional[BeamState], Optional[Tuple[str, str, Optional[str], Optional[str]]]],
            ] = {}

            all_terminal = True
            for state in beam:
                i, j, t = state.i, state.j, state.t
                cost = best_cost[state]

                if not (i == m and j == n and t == T):
                    all_terminal = False

                if i == m and j == n and t == T:
                    prev = next_candidates.get(state)
                    cand = (cost, 99, None, None)
                    if prev is None or cand[0] < prev[0] or (cand[0] == prev[0] and cand[1] < prev[1]):
                        next_candidates[state] = cand
                    continue

                def consider(
                    ns: BeamState,
                    nc: int,
                    op: Tuple[str, str, Optional[str], Optional[str]],
                ) -> None:
                    pri = op_priority(op[0])
                    prev = next_candidates.get(ns)
                    if prev is None or nc < prev[0] or (nc == prev[0] and pri < prev[1]):
                        next_candidates[ns] = (nc, pri, state, op)

                # A-chain: separate match and substitution.
                if i < m and t < T:
                    if norm_chain_a[i] == norm_hyp[t]:
                        consider(BeamState(i + 1, j, t + 1), cost, ("match_a", "A", work_chain_a[i], work_hyp[t]))
                    else:
                        consider(BeamState(i + 1, j, t + 1), cost + 1, ("sub_a", "A", work_chain_a[i], work_hyp[t]))

                # B-chain: separate match and substitution.
                if j < n and t < T:
                    if norm_chain_b[j] == norm_hyp[t]:
                        consider(BeamState(i, j + 1, t + 1), cost, ("match_b", "B", work_chain_b[j], work_hyp[t]))
                    else:
                        consider(BeamState(i, j + 1, t + 1), cost + 1, ("sub_b", "B", work_chain_b[j], work_hyp[t]))

                if t < T:
                    consider(BeamState(i, j, t + 1), cost + 1, ("insert", "H", None, work_hyp[t]))

                if i < m:
                    consider(BeamState(i + 1, j, t), cost + 1, ("delete_a", "A", work_chain_a[i], None))

                if j < n:
                    consider(BeamState(i, j + 1, t), cost + 1, ("delete_b", "B", work_chain_b[j], None))

                expansions += 1
                if expansions >= max_expansions:
                    break

            if all_terminal:
                break

            ranked: List[Tuple[float, int, BeamState]] = []
            for st, (cand_cost, cand_pri, cand_parent, cand_op) in next_candidates.items():
                prev_cost = best_cost.get(st, inf)
                replace = cand_cost < prev_cost
                if cand_cost == prev_cost and cand_parent is not None and cand_op is not None:
                    if st in best_backptr:
                        _, prev_op = best_backptr[st]
                        replace = cand_pri < op_priority(prev_op[0])
                    else:
                        replace = True

                if replace:
                    best_cost[st] = cand_cost
                    if cand_parent is not None and cand_op is not None:
                        best_backptr[st] = (cand_parent, cand_op)

                exact_cost = best_cost[st]
                ranked.append((exact_cost + heuristic_weight * heuristic(st.i, st.j, st.t), exact_cost, st))

            ranked.sort(key=lambda x: (x[0], x[1], -(x[2].i + x[2].j + x[2].t)))

            target_beam_width = beam_width
            if adaptive_beam:
                target_beam_width = _adaptive_beam_size(
                    ranked,
                    base_beam_width=pass_beam_width,
                    max_beam_width=pass_max_beam_width,
                    ambiguity_margin=ambiguity_margin,
                )

            if stratified_beam:
                beam = _select_stratified_beam(
                    ranked,
                    target_beam_width=target_beam_width,
                    bucket_stride=bucket_stride,
                )
            else:
                beam = [st for _, _, st in ranked[:target_beam_width]]

        final_state = BeamState(m, n, T)
        distance = best_cost.get(final_state, inf)
        wer_score = (distance / n_ref) if (normalize and n_ref > 0 and distance != inf) else float(distance)

        alignment = None
        if return_alignment and distance != inf:
            alignment = []
            cur = final_state
            while cur in best_backptr:
                prev, step = best_backptr[cur]
                alignment.append(step)
                if prev is None:
                    break
                cur = prev
            alignment.reverse()

            # Tail pass is solved on reversed streams; convert to original order.
            if direction_mode == "tail":
                alignment = list(reversed(alignment))

        return {
            "distance": distance,
            "wer": wer_score,
            "n_ref": n_ref,
            "alignment": alignment,
            "expansions": expansions,
            "priority_mode": "sub_first",
            "direction_mode": direction_mode,
            "pass_name": f"sub_first_{direction_mode}",
        }

    front_beam_width = beam_width
    front_max_beam_width = max_beam_width
    tail_beam_width = max(8, beam_width // 2)
    tail_max_beam_width = max(tail_beam_width, 2 * tail_beam_width)

    sub_first_front = run_single_pass("front", front_beam_width, front_max_beam_width)
    sub_first_tail = run_single_pass("tail", tail_beam_width, tail_max_beam_width)

    all_runs = [sub_first_front, sub_first_tail]
    best = min(all_runs, key=lambda r: (r["distance"], r["wer"]))

    return {
        "distance": best["distance"],
        "wer": best["wer"],
        "n_ref": best["n_ref"],
        "alignment": best["alignment"],
        "priority_mode": best["priority_mode"],
        "direction_mode": best["direction_mode"],
        "pass_name": best["pass_name"],
        "expansions": best.get("expansions"),
        # Backward-compatible keys (front direction).
        "sub_first": sub_first_front,
        # Kept for compatibility; ins/del-first mode removed.
        "insdel_first": None,
        # Full available run outputs.
        "sub_first_front": sub_first_front,
        "sub_first_tail": sub_first_tail,
        "insdel_first_front": None,
        "insdel_first_tail": None,
    }
def _split_into_two_chains(segments: List[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
    """Split speaker-labeled segments into two ordered word chains."""
    if isinstance(segments, str):
        segments = _parse_transcript_literal(segments)

    speaker_order: List[str] = []
    speaker_words: Dict[str, List[str]] = {}
    for seg in segments:
        if len(seg) < 2:
            continue
        spk = seg[0]
        text = str(seg[1]).lower()
        if spk not in speaker_words:
            speaker_words[spk] = []
            speaker_order.append(spk)
        speaker_words[spk].extend(text.split())

    if not speaker_order:
        return [], []
    if len(speaker_order) == 1:
        return speaker_words[speaker_order[0]], []

    first_speaker = speaker_order[0]
    second_speaker = speaker_order[1]
    chain_a = speaker_words[first_speaker]
    chain_b = speaker_words[second_speaker]
    return chain_a, chain_b

def _get_chronological_reference(segments: List[Tuple[str, str]]) -> List[str]:
    """Extract words from segments in chronological order (preserving original order)."""
    if isinstance(segments, str):
        segments = _parse_transcript_literal(segments)
    
    words = []
    for seg in segments:
        if len(seg) < 2:
            continue
        text = str(seg[1]).lower()
        words.extend(text.split())
    return words


def _flatten_transcript_segments(value: object) -> str:
    """Flatten a transcript list/tuple structure into a single text string."""
    if isinstance(value, str):
        value = _parse_transcript_literal(value)
    if not isinstance(value, list):
        return ""

    parts: List[str] = []
    for seg in value:
        if isinstance(seg, (list, tuple)) and len(seg) >= 2:
            parts.append(str(seg[1]).strip())
        elif isinstance(seg, str):
            parts.append(seg.strip())
    return " ".join(part for part in parts if part).strip()


def benchmark_wav2vec2_sample(
    seed: int = 1234,
    sample_size: int = 50,
    beam_width: int = 64,
    lookahead: int = 16,
    normalize: bool = True,
    heuristic_weight: float = 0.4,
    max_expansions: int = 160000,
    adaptive_beam: bool = True,
    stratified_beam: bool = True,
) -> Dict[str, object]:
    """Benchmark MRS-WER against standard WER on a seeded random wav2vec2 sample."""
    manifest_candidates = [
        Path("Output/manifest.csv"),
        Path("../Output/manifest.csv"),
        Path("/home/jamin/Year3Proj/Output/manifest.csv"),
    ]
    asr_candidates = [
        Path("ASR_transcriptions.json"),
        Path("../ASR_transcriptions.json"),
        Path("/home/jamin/Year3Proj/ASR_transcriptions.json"),
    ]

    manifest_path = next((p for p in manifest_candidates if p.exists()), None)
    asr_path = next((p for p in asr_candidates if p.exists()), None)
    if manifest_path is None:
        raise FileNotFoundError("Could not find Output/manifest.csv")
    if asr_path is None:
        raise FileNotFoundError("Could not find ASR_transcriptions.json")

    manifest_lookup: Dict[str, List[Tuple[str, str, float, float]]] = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        import csv

        reader = csv.DictReader(f)
        for row in reader:
            manifest_lookup[row["clip_id"]] = _parse_transcript_literal(row.get("transcript"))

    with open(asr_path, "r", encoding="utf-8") as f:
        asr_payload = json.load(f)

    wav_lookup: Dict[str, str] = {}
    for clip_id, payload in asr_payload.items():
        transcript_block = payload.get("transcript", {}) if isinstance(payload, dict) else {}
        wav_lookup[clip_id] = _flatten_transcript_segments(transcript_block.get("wav2vec2"))

    available_clip_ids = [
        clip_id
        for clip_id in manifest_lookup.keys()
        if clip_id in wav_lookup and manifest_lookup[clip_id] and wav_lookup[clip_id]
    ]
    if not available_clip_ids:
        raise ValueError("No overlapping wav2vec2 clips found between manifest and ASR transcripts.")

    rng = random.Random(seed)
    chosen_clip_ids = rng.sample(available_clip_ids, k=min(sample_size, len(available_clip_ids)))

    rows: List[Dict[str, object]] = []
    mrs_runtime_s = 0.0
    for clip_id in chosen_clip_ids:
        ref_segments = manifest_lookup[clip_id]
        ref_chain_a, ref_chain_b = _split_into_two_chains(ref_segments)
        hyp_words = wav_lookup[clip_id].lower().split()

        wer_result = wer_with_alignment(_get_chronological_reference(ref_segments), hyp_words, return_alignment=False)
        mrs_t0 = time.perf_counter()
        mrs_result = mrs_wer_beam_2chain(
            ref_chain_a,
            ref_chain_b,
            hyp_words,
            beam_width=beam_width,
            heuristic_weight=heuristic_weight,
            normalize=normalize,
            return_alignment=False,
            lookahead=lookahead,
            max_expansions=max_expansions,
            adaptive_beam=adaptive_beam,
            stratified_beam=stratified_beam,
        )
        mrs_runtime_s += (time.perf_counter() - mrs_t0)

        wer_errors = int(wer_result["distance"])
        mrs_errors = int(mrs_result["distance"])
        error_difference = wer_errors - mrs_errors
        rate_difference = wer_result["wer"] - mrs_result["wer"]

        rows.append({
            "clip_id": clip_id,
            "wer_errors": wer_errors,
            "mrs_errors": mrs_errors,
            "wer": wer_result["wer"],
            "mrs_wer": mrs_result["wer"],
            "difference": error_difference,
            "rate_difference": rate_difference,
        })

    mrs_better = sum(1 for row in rows if row["difference"] > 0)
    wer_better = sum(1 for row in rows if row["difference"] < 0)
    equal_count = sum(1 for row in rows if row["difference"] == 0)

    return {
        "seed": seed,
        "sample_size": len(rows),
        "mrs_better": mrs_better,
        "wer_better": wer_better,
        "equal": equal_count,
        "mrs_runtime_s": mrs_runtime_s,
        "mrs_time_per_sample_s": (mrs_runtime_s / len(rows)) if rows else 0.0,
        "mean_wer": sum(row["wer"] for row in rows) / len(rows),
        "mean_mrs_wer": sum(row["mrs_wer"] for row in rows) / len(rows),
        "mean_difference": sum(row["difference"] for row in rows) / len(rows),
        "mean_rate_difference": sum(row["rate_difference"] for row in rows) / len(rows),
        "rows": rows,
        "manifest_path": str(manifest_path.resolve()),
        "asr_path": str(asr_path.resolve()),
    }


def wer_with_alignment(
    ref_words: List[str],
    hyp_words: List[str],
    return_alignment: bool = True,
) -> Dict[str, object]:
    """Calculate WER using standard edit distance with optional alignment.
    
    Returns:
        {
            "distance": edit distance,
            "wer": WER score,
            "n_ref": number of reference words,
            "alignment": list of (operation, ref_word, hyp_word)
        }
    """
    m = len(ref_words)
    n = len(hyp_words)
    
    # DP table for edit distance
    d = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if _strip_punctuation(ref_words[i-1]) == _strip_punctuation(hyp_words[j-1]):
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(
                    d[i-1][j-1] + 1,  # substitution
                    d[i][j-1] + 1,    # insertion
                    d[i-1][j] + 1     # deletion
                )
    
    distance = d[m][n]
    wer_score = distance / m if m > 0 else 0.0
    
    # Backtrack to get alignment
    alignment = []
    if return_alignment:
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0 and _strip_punctuation(ref_words[i-1]) == _strip_punctuation(hyp_words[j-1]):
                # Match
                alignment.append(("match", "REF", ref_words[i-1], hyp_words[j-1]))
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and d[i-1][j-1] + 1 == d[i][j]:
                # Substitution
                alignment.append(("sub", "REF", ref_words[i-1], hyp_words[j-1]))
                i -= 1
                j -= 1
            elif j > 0 and d[i][j-1] + 1 == d[i][j]:
                # Insertion
                alignment.append(("insert", "HYP", None, hyp_words[j-1]))
                j -= 1
            elif i > 0 and d[i-1][j] + 1 == d[i][j]:
                # Deletion
                alignment.append(("delete", "REF", ref_words[i-1], None))
                i -= 1
            else:
                break
        
        alignment.reverse()
    
    return {
        "distance": distance,
        "wer": wer_score,
        "n_ref": m,
        "alignment": alignment,
    }

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

def _demo() -> None:
    # Your example
    chain_a = ["hello", "world"]
    chain_b = ["goodbye", "earth"]
    hyp = ["hello", "goodbye", "world", "earth"]

    result = mrs_wer_beam_2chain(
        chain_a,
        chain_b,
        hyp,
        beam_width=16,
        heuristic_weight=1.0,
        return_alignment=True,
        lookahead=12
    )

    print("distance =", result["distance"])
    print("wer      =", result["wer"])
    print("alignment:")
    for step in result["alignment"] or []:
        print("  ", step)

    # Example with one wrong word
    hyp2 = ["hello", "goodbye", "mars", "earth"]
    result2 = mrs_wer_beam_2chain(
        chain_a,
        chain_b,
        hyp2,
        beam_width=16,
        heuristic_weight=1.0,
        return_alignment=True,
        lookahead=12
    )

    print("\nsecond example")
    print("distance =", result2["distance"])
    print("wer      =", result2["wer"])
    print("alignment:")
    for step in result2["alignment"] or []:
        print("  ", step)
    
    from pathlib import Path
    import pandas as pd
    import json
    #get clip id mix_0002525_0.40_2_7.4_T
    # get corresponding transcription of parakeet from ASR_transcriptions.json
    df = pd.read_csv("Output/manifest.csv")
    clip_id = "mix_0003323_0.40_2_-5_T"
    with open("ASR_transcriptions.json", 'r') as f:
        hyps = json.load(f)
    
    
    ref = _parse_transcript_literal(df.loc[df['clip_id'] == clip_id, 'transcript'].values[0])

    hyp_text = hyps.get(clip_id, "")
    hyp_text = hyp_text.get("transcript", "").get("wav2vec2", "")[0][1]
    print(hyp_text)
    hyp = hyp_text.lower().split()
    chain_a, chain_b = _split_into_two_chains(ref)
    print("\nreference chain A:", chain_a)
    print("reference chain B:", chain_b)
    
    # Calculate MRS_WER with alignment
    result3 = mrs_wer_beam_2chain(
        chain_a,
        chain_b,
        hyp,
        beam_width=16,
        heuristic_weight=1.0,
        return_alignment=True,
        lookahead=12
    )
    
    print("\nreal example")
    print("distance =", result3["distance"])
    print("mrs_wer  =", result3["wer"])
    
    # Calculate regular WER for comparison using chronological reference
    chron_ref = _get_chronological_reference(ref)
    wer_result = wer_with_alignment(chron_ref, hyp, return_alignment=True)
    print("wer      =", wer_result["wer"])
    print(f"\nComparison: WER={wer_result['wer']:.6f}, MRS_WER={result3['wer']:.6f}")
    print(f"Difference: {abs(wer_result['wer'] - result3['wer']):.6f}")
    print(f"WER uses chronological reference ({len(chron_ref)} words)")
    print(f"MRS_WER uses speaker chains (A={len(chain_a)}, B={len(chain_b)} words)")
    
    # Save alignments to files
    from pathlib import Path
    output_dir = Path('/home/jamin/Year3Proj/Output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save WER alignment
    wer_file = output_dir / 'wer_alignment.txt'
    with open(wer_file, 'w') as f:
        f.write("=" * 120 + "\n")
        f.write("STANDARD WER ALIGNMENT (Chronological Reference)\n")
        f.write("=" * 120 + "\n\n")
        f.write(f"WER Score: {wer_result['wer']:.6f}\n")
        f.write(f"Edit Distance: {wer_result['distance']}\n")
        f.write(f"Reference Words: {wer_result['n_ref']}\n")
        f.write(f"\nReference: {' '.join(chron_ref)}\n")
        f.write(f"Hypothesis: {' '.join(hyp)}\n")
        f.write("\n" + "-" * 120 + "\n")
        f.write("ALIGNMENT SEQUENCE:\n")
        f.write("-" * 120 + "\n")
        f.write(f"{'Operation':<12} {'Type':<8} {'Ref Word':<30} {'Hyp Word':<30}\n")
        f.write("-" * 120 + "\n")
        
        match_count = 0
        sub_count = 0
        insert_count = 0
        delete_count = 0
        
        for op, op_type, ref_word, hyp_word in (wer_result['alignment'] or []):
            f.write(f"{op:<12} {op_type:<8} {str(ref_word):<30} {str(hyp_word):<30}\n")
            if op == "match":
                match_count += 1
            elif op == "sub":
                sub_count += 1
            elif op == "insert":
                insert_count += 1
            elif op == "delete":
                delete_count += 1
        
        f.write("-" * 120 + "\n")
        f.write(f"Summary: Matches={match_count}, Substitutions={sub_count}, Insertions={insert_count}, Deletions={delete_count}\n")
    
    print(f"\nWER alignment saved to: {wer_file}")
    
    # Save MRS_WER alignment
    mrs_file = output_dir / 'mrs_wer_alignment.txt'
    with open(mrs_file, 'w') as f:
        f.write("=" * 120 + "\n")
        f.write("MULTI-REFERENCE SPEAKER WER ALIGNMENT (Two Speaker Chains)\n")
        f.write("=" * 120 + "\n\n")
        selected_priority = result3.get('priority_mode', 'unknown')
        selected_direction = result3.get('direction_mode', 'unknown')
        selected_pass = result3.get('pass_name', 'unknown')
        selected_priority_label = "sub-first" if selected_priority == "sub_first" else str(selected_priority)
        selected_direction_label = "front-first" if selected_direction == "front" else (
            "end-first" if selected_direction == "tail" else str(selected_direction)
        )
        f.write(f"Selected Pass Name: {selected_pass}\n")
        f.write(f"Selected Priority Mode: {selected_priority_label}\n")
        f.write(f"Selected Direction Mode: {selected_direction_label}\n")
        f.write(f"Selected MRS_WER Score: {result3['wer']:.6f}\n")
        f.write(f"Selected Edit Distance: {result3['distance']}\n")
        f.write(f"Total Reference Words: {result3['n_ref']}\n")
        f.write(f"\nReference Chain A: {' '.join(chain_a)}\n")
        f.write(f"Reference Chain B: {' '.join(chain_b)}\n")
        f.write(f"Hypothesis: {' '.join(hyp)}\n")

        def write_mrs_section(title: str, mrs_result: Dict[str, object]) -> None:
            priority_mode = mrs_result.get('priority_mode', 'unknown')
            direction_mode = mrs_result.get('direction_mode', 'unknown')
            pass_name = mrs_result.get('pass_name', 'unknown')
            priority_label = "sub-first" if priority_mode == "sub_first" else str(priority_mode)
            direction_label = "front-first" if direction_mode == "front" else (
                "end-first" if direction_mode == "tail" else str(direction_mode)
            )

            f.write("\n" + "-" * 120 + "\n")
            f.write(f"{title}\n")
            f.write("-" * 120 + "\n")
            f.write(f"Pass Name: {pass_name}\n")
            f.write(f"Priority Mode: {priority_label}\n")
            f.write(f"Direction Mode: {direction_label}\n")
            f.write(f"MRS_WER Score: {mrs_result['wer']:.6f}\n")
            f.write(f"Edit Distance: {mrs_result['distance']}\n")
            f.write(f"Total Reference Words: {mrs_result['n_ref']}\n")
            f.write("-" * 120 + "\n")
            f.write("ALIGNMENT SEQUENCE:\n")
            f.write("-" * 120 + "\n")
            f.write(f"{'Operation':<15} {'Chain':<8} {'Ref Word':<30} {'Hyp Word':<30}\n")
            f.write("-" * 120 + "\n")

            match_a = 0
            match_b = 0
            sub_a = 0
            sub_b = 0
            insert_count = 0
            delete_a = 0
            delete_b = 0

            for op, chain_id, ref_word, hyp_word in (mrs_result['alignment'] or []):
                f.write(f"{op:<15} {chain_id:<8} {str(ref_word):<30} {str(hyp_word):<30}\n")
                if op == "match_a":
                    match_a += 1
                elif op == "match_b":
                    match_b += 1
                elif op == "sub_a":
                    sub_a += 1
                elif op == "sub_b":
                    sub_b += 1
                elif op == "insert":
                    insert_count += 1
                elif op == "delete_a":
                    delete_a += 1
                elif op == "delete_b":
                    delete_b += 1

            f.write("-" * 120 + "\n")
            f.write(f"Summary Chain A: Matches={match_a}, Substitutions={sub_a}, Deletions={delete_a}\n")
            f.write(f"Summary Chain B: Matches={match_b}, Substitutions={sub_b}, Deletions={delete_b}\n")
            f.write(f"Insertions (total): {insert_count}\n")

        write_mrs_section("SELECTED RESULT", result3)
        write_mrs_section("SUB-FIRST PRIORITY RESULT (FRONT-FIRST)", result3['sub_first_front'])
        if result3.get('sub_first_tail') is not None:
            write_mrs_section("SUB-FIRST PRIORITY RESULT (END-FIRST)", result3['sub_first_tail'])
    
    print(f"MRS_WER alignment saved to: {mrs_file}")
    
    # Print alignment sequences
    print("\n" + "=" * 120)
    print("WER ALIGNMENT SEQUENCE:")
    print("=" * 120)
    for op, op_type, ref_word, hyp_word in (wer_result['alignment'] or [])[:20]:
        print(f"{op:<12} {op_type:<8} {str(ref_word):<30} {str(hyp_word):<30}")
    if len(wer_result['alignment'] or []) > 20:
        print(f"... ({len(wer_result['alignment']) - 20} more operations)")
    
    print("\n" + "=" * 120)
    print("MRS_WER ALIGNMENT SEQUENCE:")
    print("=" * 120)
    for op, chain_id, ref_word, hyp_word in (result3['alignment'] or [])[:20]:
        print(f"{op:<15} {chain_id:<8} {str(ref_word):<30} {str(hyp_word):<30}")
    if len(result3['alignment'] or []) > 20:
        print(f"... ({len(result3['alignment']) - 20} more operations)")


if __name__ == "__main__":
    _demo()