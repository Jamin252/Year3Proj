import itertools
from typing import Dict, List, Optional
import copy
import numpy as np


def wer(ref: str, hyp:str, abs:bool = False, sub_cost:int = 1) -> float:
    ref = ref.lower()
    
    ref = ''.join(c for c in ref if c.isalnum() or c.isspace())
    hyp = hyp.lower()
    hyp = ''.join(c for c in hyp if c.isalnum() or c.isspace())
    ref_words = ref.split()
    hyp_words = hyp.split()
    
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
        
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i -1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(d[i-1][j-1] + sub_cost, d[i][j-1] + 1, d[i-1][j] + 1)
    # print(f"REF: {ref}", f"HYP: {hyp}", f"WER: {d[-1][-1]}")
    if abs:
        return d[-1][-1]
    wer = d[-1][-1] / len(ref_words)
    
    return wer

def transcript_to_dict(trans: List[tuple[str, str]]) -> Dict[str, List[str]]:
    d = {}
    # print(trans)
    for (spk, sen) in trans:
        if spk not in d:
            d[spk] = []
        d[spk].append(sen)
    return d

def cpWER(ref: List[tuple[str, str]],hyp: List[tuple[str, str]], DI = False) -> float:
    ref_copy = transcript_to_dict(ref)
    hyp_copy = transcript_to_dict(hyp)
    for spk in ref_copy.keys():
        ref_copy[spk] = ' '.join(ref_copy[spk])
    for spk in hyp_copy.keys():
        hyp_copy[spk] = ' '.join(hyp_copy[spk])
    ref_keys = list(ref_copy.keys())
    hyp_keys = list(hyp_copy.keys())
    if len(ref_keys) < len(hyp_keys):
        ref_keys += [f"dummy" for _ in range(len(hyp_keys) - len(ref_keys))]
    elif len(hyp_keys) < len(ref_keys):
        hyp_keys += [f"dummy" for _ in range(len(ref_keys) - len(hyp_keys))]
    min_wer = float('inf')
    min_perm = None
    t = {}
    for ref_spks in itertools.permutations(ref_keys):
        total_wer = 0
        # print(ref_spks, hyp_keys)
        for ref_spk, hyp_spk in zip(ref_spks, hyp_keys):
            # print(f"Comparing {ref_spk} with {hyp_spk}")
            if ref_spk == "dummy":
                # print(f"Reference speaker {ref_spk} is dummy")
                total_wer += len(hyp_copy[hyp_spk].split())
            elif hyp_spk == "dummy":
                # print(f"Hypothesis speaker {hyp_spk} is dummy")
                total_wer += len(ref_copy[ref_spk].split())
            else:
                # print(f"Calculating WER for {ref_spk} and {hyp_spk}")
                # print(f"Comparing {ref_copy[ref_spk]} with {hyp_copy[hyp_spk]}")
                if (ref_spk, hyp_spk) in t:
                    twer = t[(ref_spk, hyp_spk)]
                else:
                    twer= wer(ref_copy[ref_spk], hyp_copy[hyp_spk], abs = True)
                    t[(ref_spk, hyp_spk)] = twer
                # print(f"WER for {ref_spk} and {hyp_spk}: {twer}")
                total_wer += twer
                
        # print(wer)
        if total_wer < min_wer:
            min_wer = total_wer
            min_perm = ref_spks
    # print(f"Minimum WER: {min_wer} with permutation {min_perm}")
    # print(sum(len(ref_copy[spk].split()) for spk in ref_copy.keys()))
    if not DI:
        return min_wer / sum(len(ref_copy[spk].split()) for spk in ref_copy.keys())
    return min_wer, min_perm, hyp_keys

def spk_WER(ref: List[tuple[str, str]],hyp: List[tuple[str, str]], ref_spk:str, hyp_spk:str) -> float:
    if ref_spk == "dummy":
        return len([sen for spk, sen in hyp if spk == hyp_spk])
    elif hyp_spk == "dummy":
        return len([sen for spk, sen in ref if spk == ref_spk])
    ref_sen = ' '.join([sen for spk, sen in ref if spk == ref_spk])
    hyp_sen = ' '.join([sen for spk, sen in hyp if spk == hyp_spk])
    return wer(ref_sen, hyp_sen, abs = True)
def lev_dist(ref: List[tuple[str, str]], hyp: List[tuple[str, str]], ref_spks: List[str], hyp_spks: List[str], sub_cost: int) -> int:
    # print(f"Calculating Levenshtein distance for permutation {ref_spks} and hyp keys {hyp_spks} with sub cost {sub_cost}")
    # print(f"Reference: {ref}, Hypothesis: {hyp}")
    ref_dict = transcript_to_dict(ref)
    hyp_dict = transcript_to_dict(hyp)
    total_dist = 0
    # print(f"Reference dictionary: {ref_dict}, Hypothesis dictionary: {hyp_dict}")
    for r,h in zip(ref_spks, hyp_spks):
        if r == "dummy" or ref_dict.get(r, "") == "":
            total_dist += len(hyp_dict[h])
        elif h == "dummy" or hyp_dict.get(h, "") == "":
            total_dist += len(ref_dict[r])
        else:
            total_dist += wer(" ".join(ref_dict[r]), " ".join(hyp_dict[h]), abs = True, sub_cost = sub_cost)
    return total_dist

def DIcpWER(ref: List[tuple[str, str]],hyp: List[tuple[str, str]]) -> float:
    
    def greedy_update(ref,hyp,perm, hyp_keys, sub_cost):
        min_wer =  float('inf')
        tmin_wer = lev_dist(ref, hyp, perm, hyp_keys, sub_cost = sub_cost)
        while tmin_wer < min_wer:
            min_wer = tmin_wer
            for i in range(len(hyp)):
                for j in range(len(hyp_keys)):
                    if hyp_keys[j] == "dummy" or hyp_keys[j] == hyp[i][0]:
                        continue
                    
                    hyp_copy = copy.deepcopy(hyp)
                    hyp_spk = hyp_copy[i][0]
                    hyp_spk_ind = hyp_keys.index(hyp_spk)
                    hyp_copy[i] = (hyp_keys[j], hyp_copy[i][1])
                    
                    new_wer = lev_dist(ref, hyp_copy, perm, hyp_keys, sub_cost = sub_cost)
                    if new_wer < tmin_wer:
                        tmin_wer = new_wer
                        hyp = hyp_copy
                        
                    # original_wer = wers[j] + wers[hyp_spk_ind]
                    # new_wer = spk_WER(ref, hyp_copy, perm[j], hyp_keys[j]) + spk_WER(ref, hyp_copy, perm[hyp_spk_ind], hyp_keys[hyp_spk_ind])
                    # if original_wer <= new_wer:
                    #     continue
                    # tmin_wer = tmin_wer - original_wer + new_wer
                    # hyp[i] = (hyp_keys[j], hyp[i][1])
        return min_wer, hyp
    abs_wer, perm, hyp_keys = cpWER(ref, hyp, DI = True)
    # wers = [spk_WER(ref, hyp, ref_spk, hyp_spk) for ref_spk, hyp_spk in zip(perm, hyp_keys)]
    # print(f"Initial WER: {abs_wer} with permutation {perm} and hyp keys {hyp_keys}")
    hyp_copy = copy.deepcopy(hyp)
    wer, hyp_copy = greedy_update(ref, hyp_copy, perm, hyp_keys, sub_cost = 1)
    wer, hyp_copy = greedy_update(ref, hyp_copy, perm, hyp_keys, sub_cost = 2)
    ref_dict = transcript_to_dict(ref)
    return lev_dist(ref, hyp_copy, perm, hyp_keys, sub_cost = 1) / sum(len(ref_dict[spk]) for spk in ref_dict.keys())

def main():
    ref = [("spkA","hello"), ("spkA", "world"), ("spkB", "goodbye"), ("spkB", "UK")]
    hyp = [("spkA", "goodbye"), ("spkA", "world"), ("spkB", "hello"), ("spkB", "UK")]
    print(cpWER(ref, hyp))
    assert cpWER(ref, hyp) == 0.5
    # example where one permutation is better than the other
    hyp2 = [("spkA", "hello"), ("spkA", "world"), ("spkB", "goodbye"), ("spkB", "UK")]
    print(cpWER(ref, hyp2))
    assert cpWER(ref, hyp2) == 0.0
    
    # example where there is more speaker in the hypothesis than the reference
    hyp3 = [("spkA", "hello"), ("spkA", "world"), ("spkB", "goodbye"), ("spkB", "UK"), ("spkC", "extra")]
    print(cpWER(ref, hyp3))
    assert cpWER(ref, hyp3) == 0.25
    # example to test DIcpWER
    hyp4 = [("spkA", "goodbye"), ("spkB", "hello"), ("spkA", "world"), ("spkB", "UK")]
    print(DIcpWER(ref, hyp4))
    assert DIcpWER(ref, hyp4) == 0.0


if __name__ == "__main__":
    main()