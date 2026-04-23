import argparse
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import soundfile as sf
import yaml
import librosa

from helper_class import BaseMixture, MixtureMeta

SUPPORTED_AUDIO_EXTS = {".wav", ".flac"}
EPSILON = 1e-12





def load_audio_files(roots: Path) -> Dict[str, list[Path]]:
    audio_dict: Dict[str, list[Path]] = {}
    for root in roots:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTS:
                folder_name = p.parent.name
                if folder_name not in audio_dict:
                    audio_dict[folder_name] = []
                audio_dict[folder_name].append(p)
    return audio_dict


def load_librispeech_transcripts(root: Path) -> Dict[str, str]:
    transcript_map: Dict[str, str] = {}
    for txt_path in root.rglob("*.txt"):
        for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            utt_id, transcript = parts[0], parts[1].strip()
            transcript_map[utt_id] = transcript
    return transcript_map

def load_VCTK_transcripts(root: Path) -> Dict[str, str]:
    transcript_map: Dict[str, str] = {}
    for txt_path in root.rglob("*.txt"):
        if txt_path.is_file():
            utt_id = txt_path.stem
            transcript = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
            transcript_map[utt_id] = transcript
    return transcript_map

def load_noise_files(root: Path) -> Dict[str, List[Path]]:
    noise_dict: Dict[str, List[Path]] = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTS:
            noise_type = p.parent.name[0].upper()
            if noise_type not in noise_dict:
                noise_dict[noise_type] = []
            noise_dict[noise_type].append(p)
    return noise_dict

    
def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return audio
    audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    # ratio = target_sr / sr
    # target_len = int(len(audio) * ratio)
    # idx = np.linspace(0, len(audio) - 1, target_len)
    # return np.interp(idx, np.arange(len(audio)), audio)
    return audio.astype(np.float32)


# def pad_or_trim(audio: np.ndarray, sample_length: int, offset: int = 0) -> np.ndarray:
#     if len(audio) >= sample_length:
#         return audio[:sample_length]
#     return audio.astype(np.float32) + np.zeros(sample_length - len(audio), dtype=np.float32)


def generate_offsets(audios: List[np.ndarray], n_speakers: int, overlap_ratio_target: float, sample_rate: int, speaker_spacing: tuple[float, float], sample_length: int, std_ratio: float = 0.01) -> tuple[List[int], float, np.ndarray]:
    # print("\n")
    # print(n_speakers, overlap_ratio_target, sample_rate, speaker_spacing, sample_length)
    offsets = [0]
    p = len(audios[0])
    active = np.zeros(sample_length, dtype=np.int32)
    active[:len(audios[0])] += 1
    
    if n_speakers == 1 or overlap_ratio_target <= 0:
        for a in audios[1:]:
            p += max(0, np.random.normal(*speaker_spacing) * sample_rate)
            offsets.append(int(p))
            active[int(p):int(p)+len(a)] += 1
            p += len(a)
        return (offsets,overlap_ratio_target, active, len(active))
    
    def allocate_overlap(caps: List[int]) -> List[int]:
        n_or_target = int(sum(len(a) for a in audios) * (overlap_ratio_target) // (1 + overlap_ratio_target))
        # print(n_or_target, sum(len(a) for a in audios))
        
        avg_overlap = n_or_target / (len(audios) + EPSILON)
        overlaps = np.array([int(np.random.normal(avg_overlap, avg_overlap * std_ratio))for _ in range(len(audios)-1)], dtype=np.int32)
        overlaps = np.clip(overlaps, 0, np.array(caps))
            
        diff = int(n_or_target - sum(overlaps))
        if sum(caps) < n_or_target:
            raise ValueError(f"Cannot achieve target overlap ratio with given audio lengths. Max possible overlap is {sum(caps)} samples, but target is {n_or_target} samples.")
        while diff > 0:
            i = random.randrange(len(overlaps))
            if overlaps[i] < caps[i]:
                overlaps[i] += 1
                diff -= 1
        while diff < 0:
            i = random.randrange(len(overlaps))
            if overlaps[i] > 0:
                overlaps[i] -= 1
                diff += 1
        # print(f"total overlap: {sum(overlaps)}, target: {n_or_target}, overlaps: {sum(overlaps)/(   sum(len(a) for a in audios)):.4f}, average overlap is {n_or_target / sum(len(a) for a in audios):.2f}s")
        return overlaps.tolist()
    # print(f"total overlap: {sum(overlaps)}, target: {n_or_target}, overlaps: {sum(overlaps)/(   sum(len(a) for a in audios)-sum(overlaps)):.4f}")
    caps = [min(len(audios[i-1]), len(audios[i])) for i in range(1, len(audios))]
    done = False
    # print(len(audios[0]), sample_length)
    break_flag = True
    overlaps = allocate_overlap(caps)
    diff = 0
    first = True
    max_attempts = 20
    attempt = 0
    while (first or diff != 0) and attempt < max_attempts:
        first = False
        offsets = [0]
        p = len(audios[0])
        last_p = p
        active = np.zeros(sample_length, dtype=np.int32)
        active[:len(audios[0])] += 1
        for i, (a, offset) in enumerate(zip(audios[1:], overlaps)):
            # o = min(offset, len(a))
            p -= offset
            # if not (p > 0 and active[int(p) - 1] == 1 and diff > 0 and (p + len(a) >= sample_length or active[int(p) + len(a)] == 0)):
            #     print(f"each bool is p > 0: {p > 0}, active[int(p) - 1] == 1: {active[int(p) - 1] == 1}, diff > 0: {diff > 0}, (p + len(a) >= sample_length): {(p + len(a) >= sample_length)}, active[int(p) + len(a)] == 0: {(active[int(p) + len(a)] == 0) if p + len(a) < sample_length else 'N/A'}")
            while p > 0 and active[int(p) - 1] == 1 and diff > 0:
                p -= 1
                if p + len(a) >= sample_length or active[int(p) + len(a)] == 0:
                    diff -= 1
                    overlaps[i] += 1
            # print(f"first while p is {i}")
            while active[int(p)] > 1:
                p += 1
                diff += 1
                overlaps[i] -= 1
            # print(f"second while p is {i}")
            p += len(a)
            # if p >= sample_length:
            #     print(p, i, diff)
            #     caps = caps[:i]
            #     break_flag = True
            #     break
            if p > sample_length:
                active = np.pad(active, (0, p - len(active)), constant_values=0)
            active[p-len(a):p] += 1
            # print(f"sm: {int(sm)}")
            offsets.append(int(p-len(a)))
            p = max(p, last_p)
        print(f"iteration overlaps: {np.sum(active > 1) / np.sum(active >= 1):.4f}, diff: {diff}")
        attempt += 1
    if attempt == max_attempts and diff != 0:
        return None, None, None, None
        
        
    # if diff != 0:
    #     raise ValueError(f"Warning: Could not perfectly achieve target overlap ratio. Remaining diff: {diff} samples.")
    # print(sm, sample_length)
    actual_or = np.sum(active > 1) / np.sum(active >= 1)
    # print(f"Diff is {diff}, n_or_target - overlaps = {n_or_target - sum(overlaps)}, Actual overlap ratio: {actual_or:.4f} (target was {overlap_ratio_target:.4f}), time with 3 speakers: {np.sum(active > 2) / sample_rate:.2f}s")
    offsets = offsets + [sample_length + 1000] * (len(audios) - len(offsets))  # pad with large offsets for any unused speakers
    return (offsets, actual_or, active, len(active))


def add_noise_for_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float, base_rms: float) -> np.ndarray:
    noise_rms = rms(noise)
    # print(f"speech shape: {speech.shape}, noise shape: {noise.shape}, speech_rms: {speech_rms:.6f}, noise_rms: {noise_rms:.6f}")
    alpha = base_rms / (EPSILON + noise_rms * (10 ** (snr_db / 20)))
    return speech + alpha * noise




def build_base_mixture(
    speech_files: Dict[str, List[Path]],
    transcript_map: Dict[str, str],
    n_speakers: int,
    sample_length: int,
    sample_rate: int,
    overlap_ratio_target: float,
    # mean_length: int,
    speaker_spacing: tuple[float, float],
    long_sample_length: int = 0,
    mins_lengths: List[float]= [10,20],
    std_ratio: float = 0.01,
) -> BaseMixture:
    speakers: List[str] = random.sample(list(speech_files.keys()), n_speakers)
    sample_length = max(sample_length, long_sample_length) if overlap_ratio_target > 0.5 else sample_length
    chosen:List[Path] = []
    audio_parts = []
    transcript_parts: List[tuple[str, str]] = []
    # if mean_length <= 0:
    #     raise ValueError("Mean length of audio files must be greater than 0.")
    num_audio = 0
    current_length = 0
    min_length = mins_lengths[0] if overlap_ratio_target <= 0.5 else mins_lengths[1]
    last_speaker = None
    speaker_occurrences: Dict[str, int] = {s: 0 for s in speakers}
    
    while current_length <= sample_length * (1 + overlap_ratio_target):
        speaker = random.choices(speakers, weights=[1.0 / ( speaker_occurrences[s] + EPSILON) if s != last_speaker else 0 for s in speakers], k=1)[0]
        last_speaker = speaker
        speaker_occurrences[speaker] += 1
        speech_path = random.choice(speech_files[speaker])
        # while speech_path in chosen:
        #     speech_path = random.choice(speech_files[speaker])
        chosen.append(speech_path)
        
        audio, sr = sf.read(speech_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = resample(audio.astype(np.float32), sr, sample_rate)
        
        current_length += len(audio)
        num_audio += 1
        utt_id = speech_path.stem
        if utt_id.endswith("_mic1") or utt_id.endswith("_mic2") or utt_id.endswith("_mic3"):
            utt_id = utt_id[:-5]
        transcript = transcript_map.get(utt_id)
        if transcript is None or transcript == "None":
            raise ValueError(f"Transcript not found for {utt_id} (from file {speech_path}).")
        transcript_parts.append((speaker, transcript)) 
        
        if len(audio) < sample_rate * min_length:
            tLength = len(audio)
            while tLength < sample_rate * min_length:
                tspeech_path = random.choice(speech_files[speaker])
                # while tspeech_path in chosen:
                #     tspeech_path = random.choice(speech_files[speaker])
                chosen.append(tspeech_path)
                taudio, sr = sf.read(tspeech_path)
                if taudio.ndim > 1:
                    taudio = np.mean(taudio, axis=1)
                taudio = resample(taudio.astype(np.float32), sr, sample_rate)
                spacing = min(0, max(int(0.01 * sample_rate), int(sample_rate * np.random.normal(*speaker_spacing))))
                audio = np.concatenate([audio, np.zeros(spacing), taudio], axis=0)
                tLength += len(taudio)
                current_length += len(taudio) + spacing
                
                tutt_id = tspeech_path.stem
                if tutt_id.endswith("_mic1") or tutt_id.endswith("_mic2") or tutt_id.endswith("_mic3"):
                    tutt_id = tutt_id[:-5]
                ttranscript = transcript_map.get(tutt_id)
                if ttranscript is None or ttranscript == "None":
                    raise ValueError(f"Transcript not found for {tutt_id} (from file {tspeech_path}).")
                transcript_parts.append((speaker, ttranscript))
                num_audio += 1   
        audio_parts.append(audio)
        
        # utt_id = speech_path.stem
        # transcript = transcript_map.get(utt_id)
        # transcript_parts.append((speaker, transcript))
    # print(set([t[0] for t in transcript_parts]))
    # audio_parts = audio_parts[:-1]
    # print(f"actual min length: {min(len(a)/sample_rate for a in audio_parts):.2f}s, current_length: {current_length/sample_rate:.2f}s, num_audio: {num_audio}")
    # num_audio -= 1
    # transcript_parts = transcript_parts[:-1]
    # print(f"num_audio: {num_audio}")
    # print(chosen[0])
    # for speech_path in chosen:
    #     audio, sr = sf.read(speech_path)
    #     if audio.ndim > 1:
    #         audio = np.mean(audio, axis=1)
    #     audio = resample(audio.astype(np.float32), sr, sample_rate)
    #     # audio = pad_or_trim(audio, n_samples)
    #     audio_parts.append(audio)

    #     utt_id = speech_path.stem
    #     transcript = transcript_map.get(utt_id)
    #     transcript_parts.append((utt_id, transcript))

    actual_or = -1.0
    max_attempts = 3
    attempt = 0
    offsets = None
    while (offsets is None or abs(actual_or - overlap_ratio_target) > 0.01) and attempt < max_attempts:
        (offsets, actual_or, active_mask, actual_length) = generate_offsets(
            audio_parts, n_speakers, 
            float(overlap_ratio_target), 
            sample_rate,  
            speaker_spacing, sample_length, std_ratio=std_ratio)
        attempt += 1
        if offsets is not None:
            print(f"actual overlap ratio: {actual_or:.4f} (target was {overlap_ratio_target:.4f}), actualy length: {actual_length/sample_rate:.2f}s")
    if attempt == max_attempts and (offsets is None or abs(actual_or - overlap_ratio_target) > 0.01):
        return None 
    # (offsets, actual_or, active_mask) = generate_offsets(
    #         audio_parts, n_speakers, 
    #         float(overlap_ratio_target), 
    #         sample_rate,  
    #         speaker_spacing, sample_length)
    mixture = np.zeros(sample_length, dtype=np.float32)
    # print(offsets)
    for i, (w, off) in enumerate(zip(audio_parts, offsets)):
        if off + len(w) >= sample_length:
            mixture = np.pad(mixture, (0, off + len(w) - len(mixture)), constant_values=0)
        transcript_parts[i] = (transcript_parts[i][0], transcript_parts[i][1], off / sample_rate, (off + len(w)) / sample_rate)
        mixture[off:off + len(w)] += w
    
    rms_val = rms(np.concatenate(audio_parts))

    return BaseMixture(
        wave=mixture,
        overlap_mask=active_mask,
        source_files=chosen,
        transcript=transcript_parts,
        overlap_ratio_actual=actual_or,
        rms=rms_val,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic overlapped speech dataset.")
    parser.add_argument("--speech-roots", type=Path, required=True)
    parser.add_argument("--vctk-transcripts", type=Path, required=False)
    parser.add_argument("--noise-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    
    audio_paths = []
    with open(args.speech_roots, "r") as f:
        for line in f.readlines():
            path = line.strip()
            if path:
                audio_paths.append(Path(path))
    if not audio_paths:
        raise ValueError("No valid speech root paths provided in --speech-roots.")
    

    cfg = yaml.safe_load(args.config.read_text())
    random.seed(cfg["random_seed"])
    np.random.seed(cfg["random_seed"])

    sample_rate = cfg["sample_rate"]
    sample_length = int(cfg["clip_duration_s"] * sample_rate)
    factors = cfg["factors"]
    n_per_condition = int(cfg["n_mixtures_per_condition"])
    
    speech_files: Dict[str, List[Path]] = load_audio_files(audio_paths)
    # mean_length = round(np.mean([len(files) for files in speech_files.values()]) if speech_files else 0)
    noise_files: Dict[str, List[Path]] = load_noise_files(args.noise_root)
    transcript_map: Dict[str, str] = load_librispeech_transcripts(audio_paths[0])
    transcript_map |= load_VCTK_transcripts(args.vctk_transcripts) if args.vctk_transcripts else {}
    if not speech_files:
        raise ValueError("No speech files (.wav/.flac) found.")
    if not noise_files:
        raise ValueError("No noise files (.wav/.flac) found.")
    # print(f"Loaded {len(speech_files)} speech files and {sum(len(v) for v in noise_files.values())} noise files.")
    # print(noise_files.keys())
    # print(factors["noise_type"])
    # exit()
    out_audio = args.output_root / "audio"
    # out_mask = args.output_root / "masks"
    out_audio.mkdir(parents=True, exist_ok=True)
    # out_mask.mkdir(parents=True, exist_ok=True)

    rows: List[MixtureMeta] = []
    clip_counter = 0
    total_num_clips = len(factors["overlap_ratio"]) * len(factors["speaker_count"]) * len(factors["snr_db"]) * len(factors["noise_type"]) * n_per_condition
    print(f"Generating {total_num_clips} mixtures...")

    for oratio in factors["overlap_ratio"]:
        print(f"Generating mixtures with target overlap ratio: {oratio:.2f}")
        for k in factors["speaker_count"]:
            # Build base overlapped mixtures once, then vary SNR/noise over the same base
            base_mixtures = []
            for _ in range(10 if oratio == 0 else n_per_condition):
                b = None
                while b is None:
                    b = build_base_mixture(
                        speech_files=speech_files,
                        transcript_map=transcript_map,
                        n_speakers=k,
                        sample_length=sample_length,
                        sample_rate=sample_rate,
                        overlap_ratio_target=float(oratio),
                        # mean_length=mean_length,
                        speaker_spacing=(factors["speaker_spacing"]["mu"], factors["speaker_spacing"]["sigma"]),
                        long_sample_length=int(cfg["long_audio_duration_s"] * sample_rate),
                        mins_lengths = [cfg["min_length_s"], cfg["min_length_s_0.75"]],
                        std_ratio=factors.get("overlap_std_ratio", 0.01)
                    )
                base_mixtures.append(b)
            print(f"For overlap ratio {oratio:.2f} and max speakers {k}, generated {len(base_mixtures)} base mixtures.")
            for base in base_mixtures:
                # print(f"base mixture {oratio:.2f}, speakers: {k}, actual overlap ratio: {base.overlap_ratio_actual:.4f}")
                base_rms = base.rms
                for snr_db in factors["snr_db"]:
                    for noise_type in factors["noise_type"]:
                        mixture = np.copy(base.wave)
                        noise_paths = []
                        if snr_db is not None:
                            if noise_type == "A":
                                candidates = []
                                for nt in ["D", "P", "T"]:
                                    candidates.extend(noise_files.get(nt, []))
                            else:
                                candidates = noise_files.get(noise_type, [])
                            noise = np.array([], dtype=np.float32)
                            while len(noise) < len(mixture):
                                npath = random.choice(candidates)
                                noise_paths.append(npath)
                                tnoise, nsr = sf.read(npath)
                                if tnoise.ndim > 1:
                                    tnoise = np.mean(tnoise, axis=1)
                                # print(f"tnoise shape: {tnoise.shape}, noise shape: {noise.shape}")
                                noise = np.concatenate([noise, resample(tnoise.astype(np.float32), nsr, sample_rate)], axis=0)
                            # print(noise.shape, mixture.shape)
                            noise = noise[:len(mixture)]
                            # noise = pad_or_trim(noise, total_len)
                            mixture = add_noise_for_snr(mixture, noise, float(snr_db), base_rms)

                        # mixture = np.clip(mixture, -1.0, 1.0)
                        clip_id = f"mix_{clip_counter:07d}_{oratio:.2f}_{k}_{snr_db}_{noise_type}"
                        clip_counter += 1
                        print(f"Generated mixture {clip_id}")

                        apath = out_audio / f"{clip_id}.wav"
                        # mpath = out_mask / f"{clip_id}.npy"
                        sf.write(apath, mixture, sample_rate)
                        # np.save(mpath, base.overlap_mask)

                        rows.append(
                            MixtureMeta(
                                clip_id=clip_id,
                                audio_path=str(apath),
                                transcript=base.transcript,
                                overlap_ratio_target=float(oratio),
                                overlap_ratio_actual=base.overlap_ratio_actual,
                                max_speakers=int(k),
                                snr_db=snr_db,
                                noise_type=noise_type,
                                overlap_mask_path="no mask",
                                source_files=base.source_files,
                                noise_files=noise_paths,
                            )
                        )

    manifest = pd.DataFrame([asdict(r) for r in rows])
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(
        f"Wrote {len(manifest)} mixtures to {manifest_path}. "
        f"Loaded {len(transcript_map)} transcript entries from {args.speech_roots}."
    )


if __name__ == "__main__":
    main()
