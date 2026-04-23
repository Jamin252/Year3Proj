import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'Code'))
from mrs_beam_wer import benchmark_wav2vec2_sample

configs = [
    (96, 20), (128, 20), (160, 20), (192, 20),
    (128, 24), (160, 24), (192, 24)
]

seed = 42
sample_size = 50
normalize = True
threshold = 1.0

best_config = None
best_mean_diff = -float('inf')
selected_config = None

for beam_width, lookahead in configs:
    print(f"Evaluating config: beam_width={beam_width}, lookahead={lookahead}")
    results = benchmark_wav2vec2_sample(
        beam_width=beam_width,
        lookahead=lookahead,
        seed=seed,
        sample_size=sample_size,
        normalize=normalize
    )
    
    mrs_better = results['mrs_better']
    wer_better = results['wer_better']
    equal = results['equal']
    mean_difference = results['mean_difference']
    mean_rate_difference = results['mean_rate_difference']
    mrs_runtime_s = results['mrs_runtime_s']
    mrs_time_per_sample_s = results['mrs_time_per_sample_s']
    
    print(f"mrs_better: {mrs_better}, wer_better: {wer_better}, equal: {equal}, "
          f"mean_difference: {mean_difference:.4f}, mean_rate_difference: {mean_rate_difference:.4f}, "
          f"mrs_runtime_s: {mrs_runtime_s:.4f}, mrs_time_per_sample_s: {mrs_time_per_sample_s:.4f}")
    
    if mean_difference > best_mean_diff:
        best_mean_diff = mean_difference
        best_config = (beam_width, lookahead)
        
    if mean_difference > threshold:
        selected_config = (beam_width, lookahead)
        print(f"SELECTED config: beam_width={beam_width}, lookahead={lookahead}")
        break

if not selected_config:
    print(f"Best observed config: beam_width={best_config[0]}, lookahead={best_config[1]} (mean_difference={best_mean_diff:.4f})")
