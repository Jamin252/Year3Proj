import sys
import os
from Code.mrs_beam_wer import benchmark_wav2vec2_sample

# Parameters
seed = 1234
sample_size = 10
normalize = True
beam_widths = [64, 96, 128, 160, 192]
lookaheads = [20, 24, 28, 32]
heuristic_weights = [0.8, 1.0, 1.2, 1.5]

def patch_heuristic_weight(weight):
    import Code.mrs_beam_wer
    import inspect
    
    # We need to monkeypatch the function or the call within benchmark_wav2vec2_sample
    # Since benchmark_wav2vec2_sample is already defined and calls mrs_wer_beam_2chain with heuristic_weight=1.0,
    # we might need to wrap mrs_wer_beam_2chain.
    
    original_mrs_wer_beam_2chain = Code.mrs_beam_wer.mrs_wer_beam_2chain
    
    def wrapped_mrs_wer_beam_2chain(*args, **kwargs):
        if 'heuristic_weight' in kwargs:
            kwargs['heuristic_weight'] = weight
        return original_mrs_wer_beam_2chain(*args, **kwargs)
    
    Code.mrs_beam_wer.mrs_wer_beam_2chain = wrapped_mrs_wer_beam_2chain
    return original_mrs_wer_beam_2chain

results = []

import Code.mrs_beam_wer
original_func = Code.mrs_beam_wer.mrs_wer_beam_2chain

for bw in beam_widths:
    for lh in lookaheads:
        for hw in heuristic_weights:
            # Monkeypatch
            def make_wrapper(w):
                def wrapped(*args, **kwargs):
                    kwargs['heuristic_weight'] = w
                    return original_func(*args, **kwargs)
                return wrapped
            
            Code.mrs_beam_wer.mrs_wer_beam_2chain = make_wrapper(hw)
            
            try:
                res = benchmark_wav2vec2_sample(
                    seed=seed,
                    sample_size=sample_size,
                    beam_width=bw,
                    lookahead=lh,
                    normalize=normalize
                )
                
                results.append({
                    'beam_width': bw,
                    'lookahead': lh,
                    'heuristic_weight': hw,
                    'mean_difference': res['mean_difference'],
                    'mrs_better': res['mrs_better'],
                    'wer_better': res['wer_better'],
                    'equal': res['equal'],
                    'mrs_time_per_sample_s': res['mrs_time_per_sample_s']
                })
            except Exception as e:
                print(f"Error for bw={bw}, lh={lh}, hw={hw}: {e}")

# Restore original
Code.mrs_beam_wer.mrs_wer_beam_2chain = original_func

# Sorting
results.sort(key=lambda x: x['mean_difference'], reverse=True)

print("Top 10 by mean_difference:")
for i, r in enumerate(results[:10]):
    print(f"{i+1}. BW={r['beam_width']}, LH={r['lookahead']}, HW={r['heuristic_weight']} -> MD={r['mean_difference']:.4f}, MRS={r['mrs_better']}, WER={r['wer_better']}, EQ={r['equal']}, Time={r['mrs_time_per_sample_s']:.4f}s")

if results:
    best = results[0]
    print(f"\nSingle Best Config: BW={best['beam_width']}, LH={best['lookahead']}, HW={best['heuristic_weight']} (MD={best['mean_difference']:.4f})")
    
    max_md = max(r['mean_difference'] for r in results)
    exceeded_1 = any(r['mean_difference'] > 1.0 for r in results)
    print(f"Max observed mean_difference: {max_md:.4f}")
    print(f"Any run exceeded 1.0: {exceeded_1}")
