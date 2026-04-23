from Code.mrs_beam_wer import benchmark_wav2vec2_sample

configs = [(64, 20), (96, 20), (128, 24)]
seed = 42
sample_size = 10

for bw, lh in configs:
    print(f"Testing config: (beam_width={bw}, lookahead={lh})")
    stats = benchmark_wav2vec2_sample(seed=seed, sample_size=sample_size, beam_width=bw, lookahead=lh)
    
    print(f"mean_difference: {stats.get('mean_difference')}")
    print(f"mean_rate_difference: {stats.get('mean_rate_difference')}")
    print(f"mrs_better: {stats.get('mrs_better')}")
    print(f"wer_better: {stats.get('wer_better')}")
    print(f"mrs_time_per_sample_s: {stats.get('mrs_time_per_sample_s')}")
    print("-" * 20)
