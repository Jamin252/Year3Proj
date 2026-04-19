from __future__ import annotations

import ast
import string
from dataclasses import dataclass
from math import inf
from typing import Dict, List, Tuple, Optional


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


def mrs_wer_beam_2chain(
    chain_a: List[str],
    chain_b: List[str],
    hyp: List[str],
    beam_width: int = 64,
    heuristic_weight: float = 1.0,
    normalize: bool = True,
    return_alignment: bool = False,
    max_expansions: int = 250000,
) -> Dict[str, object]:

    if beam_width <= 0:
        raise ValueError("beam_width must be > 0")
    if max_expansions <= 0:
        raise ValueError("max_expansions must be > 0")

    m = len(chain_a)
    n = len(chain_b)
    T = len(hyp)
    n_ref = m + n

    def heuristic(i: int, j: int, t: int) -> float:
        # Admissible heuristic: minimum possible cost to complete from this state.
        # Best case: all remaining hypothesis words can be matched against remaining references.
        # So the heuristic is just the number of remaining hypothesis words.
        remaining_hyp = T - t
        return remaining_hyp

    # Best known exact cost to each state that has ever been discovered.
    best_cost: Dict[BeamState, int] = {BeamState(0, 0, 0): 0}

    # Track backpointers that correspond to best_cost, plus the parent state.
    # Store (parent_state, operation) for best path to each state.
    best_backptr: Dict[BeamState, Tuple[Optional[BeamState], Tuple[str, str, Optional[str], Optional[str]]]] = {}
    
    # Track all possible predecessors and their ops (for finding best path)
    all_transitions: Dict[BeamState, List[Tuple[int, BeamState, Tuple[str, str, Optional[str], Optional[str]]]]] = {}

    beam: List[BeamState] = [BeamState(0, 0, 0)]
    expansions = 0

    # Process until beam empties or all states are terminal.
    while beam and expansions < max_expansions:
        next_candidates: Dict[BeamState, int] = {}

        all_terminal = True
        for state in beam:
            i, j, t = state.i, state.j, state.t
            cost = best_cost[state]

            if not (i == m and j == n and t == T):
                all_terminal = False

            # Terminal state is carried forward.
            if i == m and j == n and t == T:
                next_candidates[state] = min(next_candidates.get(state, inf), cost)
                continue

            # Try match operations FIRST (prefer matching over insert/delete)
            # 1) Match/substitute next A word with next hypothesis word.
            if i < m and t < T:
                sub_cost = 0 if _strip_punctuation(chain_a[i]) == _strip_punctuation(hyp[t]) else 1
                ns = BeamState(i + 1, j, t + 1)
                nc = cost + sub_cost
                if nc < next_candidates.get(ns, inf):
                    next_candidates[ns] = nc
                    op = "match_a" if sub_cost == 0 else "sub_a"
                    if ns not in all_transitions:
                        all_transitions[ns] = []
                    all_transitions[ns].append((nc, state, (op, "A", chain_a[i], hyp[t])))

            # 2) Match/substitute next B word with next hypothesis word.
            if j < n and t < T:
                sub_cost = 0 if _strip_punctuation(chain_b[j]) == _strip_punctuation(hyp[t]) else 1
                ns = BeamState(i, j + 1, t + 1)
                nc = cost + sub_cost
                if nc < next_candidates.get(ns, inf):
                    next_candidates[ns] = nc
                    op = "match_b" if sub_cost == 0 else "sub_b"
                    if ns not in all_transitions:
                        all_transitions[ns] = []
                    all_transitions[ns].append((nc, state, (op, "B", chain_b[j], hyp[t])))

            # Then try insert/delete operations
            # 3) Insert next hypothesis word.
            if t < T:
                ns = BeamState(i, j, t + 1)
                nc = cost + 1
                if nc < next_candidates.get(ns, inf):
                    next_candidates[ns] = nc
                    if ns not in all_transitions:
                        all_transitions[ns] = []
                    all_transitions[ns].append((nc, state, ("insert", "H", None, hyp[t])))

            # 4) Delete next A word.
            if i < m:
                ns = BeamState(i + 1, j, t)
                nc = cost + 1
                if nc < next_candidates.get(ns, inf):
                    next_candidates[ns] = nc
                    if ns not in all_transitions:
                        all_transitions[ns] = []
                    all_transitions[ns].append((nc, state, ("delete_a", "A", chain_a[i], None)))

            # 5) Delete next B word.
            if j < n:
                ns = BeamState(i, j + 1, t)
                nc = cost + 1
                if nc < next_candidates.get(ns, inf):
                    next_candidates[ns] = nc
                    if ns not in all_transitions:
                        all_transitions[ns] = []
                    all_transitions[ns].append((nc, state, ("delete_b", "B", chain_b[j], None)))

            expansions += 1
            if expansions >= max_expansions:
                break

        if all_terminal:
            break

        # Merge into global best_cost and rank by cost + heuristic.
        ranked: List[Tuple[float, int, BeamState]] = []
        for st, cand_cost in next_candidates.items():
            if cand_cost < best_cost.get(st, inf):
                best_cost[st] = cand_cost
                # Update best_backptr based on which transition led to this cost
                if st in all_transitions:
                    for trans_cost, parent, op in all_transitions[st]:
                        if trans_cost == cand_cost:
                            best_backptr[st] = (parent, op)
                            break
            exact_cost = best_cost[st]
            ranked.append((exact_cost + heuristic(st.i, st.j, st.t), exact_cost, st))

        ranked.sort(key=lambda x: (x[0], x[1], x[2].i + x[2].j + x[2].t))
        beam = [st for _, _, st in ranked[:beam_width]]

    final_state = BeamState(m, n, T)
    distance = best_cost.get(final_state, inf)
    wer = (distance / n_ref) if (normalize and n_ref > 0 and distance != inf) else float(distance)

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

    return {
        "distance": distance,
        "wer": wer,
        "n_ref": n_ref,
        "alignment": alignment,
    }
def _split_into_two_chains(segments: List[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
    """Split speaker-labeled segments into two ordered word chains."""
    if isinstance(segments, str):
        segments = ast.literal_eval(segments)

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
    )

    print("\nsecond example")
    print("distance =", result2["distance"])
    print("wer      =", result2["wer"])
    print("alignment:")
    for step in result2["alignment"] or []:
        print("  ", step)
    
    from pathlib import Path
    import pandas as pd
    #get clip id mix_0002525_0.40_2_7.4_T
    # get corresponding transcription of parakeet from ASR_transcriptions.json
    ref = [('218', 'A LAMP COULD NOT HAVE EXPIRED WITH MORE AWFUL EFFECT CATHERINE FOR A FEW MOMENTS WAS MOTIONLESS WITH HORROR IT WAS DONE COMPLETELY NOT A REMNANT OF LIGHT IN THE WICK COULD GIVE HOPE TO THE REKINDLING BREATH DARKNESS IMPENETRABLE AND IMMOVABLE FILLED THE ROOM', 0.0, 15.675), ('28452', 'FOR WHICH SHE HAS A WHOLE HEARTFUL OF LOVE AND THE SIGHT OF WHICH IS BETTER TO HER THAN MEDICINE DURING THE MONTH OF JULY WE EAGERLY WATCHED THE INCOMING STEAMERS AND WELCOMED ALL NEW COMERS WHO LANDED IN CHINIK', 15.8714375, 31.8814375), ('218', 'ACKNOWLEDGED THAT IT WAS BY NO MEANS AN ILL SIZED ROOM AND FURTHER CONFESSED THAT THOUGH AS CARELESS ON SUCH SUBJECTS AS MOST PEOPLE HE DID LOOK UPON A TOLERABLY LARGE EATING ROOM AS ONE OF THE NECESSARIES OF LIFE HE SUPPOSED HOWEVER', 32.138, 45.203), ('28452', "AND WAS WATCHED WITH EAGER EYES BY EVERYONE WE ATE LETTUCE AND RADISHES PICKED FRESH FROM THE GARDEN BEDS WHERE THEY HAD BEEN SOWN BY THE CAPTAIN'S OWN HANDS AND WE FOUND AGEETUK AND MOLLIE TO BE QUITE FAMOUS COOKS", 45.5575625, 61.1225625)]

    hyp_text = "A lamp could not have expired with more awful effect. Catherine, for a few moments, was motionless with horror. It was done completely. Not a remnant of light in the wick could give hope to the rekindling breath. Darkness impenetrable and immovable filled the room. And the sight of which is better to her than medicine. During the month of July, we eagerly watched the incoming steamers and welcomed all newcomers who landed in Chenick. Acknowledged that it was by no means an ill-sized room, and further confessed that, though as careless on such subjects as most people, he did look upon a tolerably large eating room as one of the necessaries of life."
    hyp = hyp_text.lower().split()
    chain_a, chain_b = _split_into_two_chains(ref)
    print("\nreference chain A:", chain_a)
    print("reference chain B:", chain_b)
    result3 = mrs_wer_beam_2chain(
        chain_a,
        chain_b,
        hyp,
        beam_width=16,
        heuristic_weight=1.0,
        return_alignment=True,
    )
    print("\nreal example")
    print("distance =", result3["distance"])
    print("wer      =", result3["wer"])
    print("alignment:")
    for step in result3["alignment"] or []:
        print("  ", step)


if __name__ == "__main__":
    _demo()