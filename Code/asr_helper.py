from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Tuple
from helper_class import MixtureMeta, MixtureTranscription, MODEL_NAMES
import pandas as pd
from pathlib import Path
import os
import librosa
import numpy as np
from faster_whisper import WhisperModel
import torch
from transformers import pipeline
import subprocess
import sys
import whisperx
import json
from Fun_ASR.model import FunASRNano
import pickle
import gc
import time

BATCH_SIZE = 10
CHECK_POINT = 100
TIMINGS = {}
def clear_gpu_cache(force_gc: bool = False):
    if force_gc:
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
def asr_model(func):
    def wrapper(metas: List[MixtureMeta], dic: Dict[str, MixtureTranscription], ind: int = 0, model_name: str = "", **kwargs):
        clear_gpu_cache(ind % CHECK_POINT == 0)  # clear GPU cache every CHECK_POINT batches to prevent memory issues
        start_time = time.perf_counter()
        # print(f"Running {func.__name__} on batch {ind}-{ind+BATCH_SIZE- 1}...")
        # print(f"Initial metas: {[meta.clip_id for meta in metas]}")
        # for meta in metas:
        #     print(f"clip_id: {meta.clip_id}, transcript keys in dic: {dic.get(meta.clip_id).transcript.keys() if dic.get(meta.clip_id) is not None else 'N/A'}")
            
        #     if meta.clip_id in dic and func.__name__ in dic[meta.clip_id].transcript:
        #         print(f"Skipping {meta.clip_id} for {func.__name__} as it already has transcription.")
        metas = [meta for meta in metas if not (dic.get(meta.clip_id) is not None and model_name in dic[meta.clip_id].transcript)]
        if len(metas) == 0:
            print(f"All clips in batch {ind}-{ind+BATCH_SIZE-1} already have transcriptions for {model_name}. Skipping this batch.")
            return
        # print([meta.clip_id for meta in metas])
        try:
            return func(metas, dic, ind, model_name, **kwargs)
        except Exception as e:
            print(f"Error occurred while running {func.__name__} on batch {ind}-{ind+BATCH_SIZE-1}. Skipping this batch.")
            raise e
        finally:
            end_time = time.perf_counter()
            name = func.__name__+"_"
            for meta in metas:
                name += f"{meta.clip_id[7:12]}-"
            TIMINGS[name] = end_time - start_time
    return wrapper

def load_mixture_meta(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def load_mixture_audio(path: Path, sr: int = 16000) -> Tuple[np.ndarray, int]:
    audio, sr = librosa.load(path, sr=sr)
    return audio, sr

def record_transcription(clip_id:str, model_name: str, transcription: List[Tuple[str, str]], dic: Dict[str, MixtureTranscription]):
    # print(f"Recording transcription for clip_id: {clip_id}, model_name: {model_name}...")
    if clip_id not in dic:
        dic[clip_id] = MixtureTranscription(clip_id=clip_id, transcript={model_name: transcription})
    else:
        dic[clip_id].transcript[model_name] = transcription


@asr_model
def transcribe_faster_whisper(metas: List[MixtureMeta], dic: Dict[str, MixtureTranscription], ind: int = 0, model_name: str = ""):
    """
    Faster Whisper Transcription
    Transcription format: list of text with no speaker labels
    """
    compute_type = "int8"
    # batch_size = 1
    # model = WhisperModel("large-v3", device="cuda", compute_type=compute_type)
    model_name = MODEL_NAMES[0]
    faster_whiper_model = WhisperModel("large-v3", device="cuda", compute_type="int8")
    for meta in metas:
        segments, info = faster_whiper_model.transcribe(Path(meta.audio_path))
        trascription = [("dummy-speaker", seg.text) for seg in segments]
        record_transcription(meta.clip_id, model_name, trascription, dic)
    # del model

@asr_model
def transcribe_wav2vec2(metas: List[MixtureMeta], dic: Dict[str, MixtureTranscription], ind: int = 0, model_name: str = "", asr=None):
    """
    Wav2Vec2 Transcription
    Transcription format: list of text with no speaker labels
    """
    
    model_name = MODEL_NAMES[1]
    
    with torch.inference_mode():
        for meta in metas:
            audio, sr = load_mixture_audio(Path(meta.audio_path))
            result = asr(audio)
            trascription = [("dummy-speaker", result["text"])]
            record_transcription(meta.clip_id, model_name, trascription, dic)
            del audio, sr, result

@asr_model
def transcribe_parakeet(metas: List[MixtureMeta], dic: Dict[str, MixtureTranscription], ind: int = 0, model_name: str = ""):
    """
    Parakeet Transcription
    Transcription format: list of text with no speaker labels
    """
    model_name = MODEL_NAMES[2]
    root = Path("/home/jamin/Year3Proj")
    temp_input_manifest = root / "temp_parakeet_input.jsonl"
    # print(f"Writing temporary manifest for Parakeet inference to {temp_input_manifest}...")
    with open(temp_input_manifest, "w", encoding="utf-8") as f:
        for meta in metas:
            row = {
                "audio_filepath": meta.audio_path,
                "text": "",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp_output_json = root / "temp_parakeet_output.json"
    subprocess.run(
        [
            sys.executable,
            "/home/jamin/Year3Proj/NeMo/examples/asr/asr_chunked_inference/rnnt/speech_to_text_streaming_infer_rnnt.py",
            'pretrained_name=nvidia/parakeet-tdt-0.6b-v3',
            'model_path=null',
            f'dataset_manifest={temp_input_manifest}',
            f'output_filename={temp_output_json}',
            'right_context_secs=2.0',
            'chunk_secs=10',
            'left_context_secs=10.0',
            'batch_size=1',
            'cuda=0',
            'decoding.greedy.use_cuda_graph_decoder=False',
            'clean_groundtruth_text=False',
        ],
        check=True,
        cwd=Path(os.pardir),  # important because audio_dir is relative
        )
    with open(temp_output_json, "r") as f:
        for line in f:
            line = line.strip()
            item = json.loads(line)
            clip_id = Path(item["audio_filepath"]).stem
            trascription = [("dummy-speaker", item["pred_text"])]
            record_transcription(clip_id, model_name, trascription, dic)
    # remove temporary files
    temp_input_manifest.unlink()
    temp_output_json.unlink()

@asr_model
def transcribe_whisperx(metas: List[MixtureMeta], dic: Dict[str, MixtureTranscription], ind: int = 0, model_name: str = ""):
    """
    WhisperX Transcription
    Transcription format: list of text with speaker labels (e.g., "speaker_1", "speaker_2", etc.)
    """
    device = "cuda"   # or "cpu"
    batch_size = 1
    compute_type = "int8"
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)
    for meta in metas:
        audio_file = meta.audio_path
        
        audio = whisperx.load_audio(audio_file)

        result = model.transcribe(audio, batch_size=batch_size)
        record_transcription(meta.clip_id, "whisperx", [("dummy-speaker", r["text"]) for r in result["segments"]], dic)
    
    del model

@asr_model
def transcribe_funasr(metas: List[MixtureMeta], dic: Dict[str, MixtureTranscription], ind: int = 0, model_name: str = ""):
    model_dir = "FunAudioLLM/Fun-ASR-Nano-2512"
    m, kwargs = FunASRNano.from_pretrained(model=model_dir, device="cuda:0")
    m.eval()
    for meta in metas:
        

        wav_path = meta.audio_path
        res = m.inference(data_in=[wav_path], **kwargs)
        with open(Path("funasr_output.pkl"), "wb") as f:
            pickle.dump(res, f)
        text = res[0][0]["text"]
        record_transcription(meta.clip_id, "funasr", [("dummy-speaker", text)], dic)



def main():
    meta_df = load_mixture_meta(Path("Output", "manifest.csv"))
    # meta_df = meta_df[meta_df["clip_id"].str.contains(r"(^mix_[0-9]+_0\.(00|14|20|40)_2_7\.4_T$)|(^mix_[0-9]+_0\.14_2_(None|7\.4|0|-5)_T$)")]
    meta_df = meta_df[meta_df["clip_id"].str.contains("mix_0001238_0.14_2_None_T")]
    # print(meta_df.head())
    dic_path = Path("ASR_transcriptions.json")
    dic = {}
    if dic_path.exists():
        with open(dic_path, "r", encoding="utf-8") as f:
            if dic_path.stat().st_size > 0:  # check if file is not empty
                dic = {k: MixtureTranscription(**v) for k, v in json.load(f).items()}
    # print(dic.keys())
    timing_file = Path("timings.json")
    if timing_file.exists():
        try:
            with open(timing_file, "r") as f:
                TIMINGS.update(json.load(f))
        except json.JSONDecodeError:
            print(f"Warning: {timing_file} is not a valid JSON file. Starting with empty timings.")
            

    # audio_list = [meta_df[meta_df.clip_id.str.startswith("mix_0.14_2_None_D_0000144")].head(1), meta_df[meta_df.clip_id.str.startswith("mix_0.14_2_None_P_0000145")].head(1)]
    # run asr model on all meta rows 100 at a time (every row) and store the transcriptions in a dictionary with clip_id as key and transcription as value and update the json file after every 100 rows
    start_ind = 0
    num_files = 4000
    ind=start_ind
    # while i < len(meta_df):
    while ind < min(num_files + start_ind, len(meta_df)):

        print(f"Processing batch {ind}-{ind+BATCH_SIZE - 1} inclusive at time {datetime.now()}: ...--------\n\n\n")
        batch = meta_df.iloc[ind:ind+BATCH_SIZE]
        audio_list = [batch.iloc[j] for j in range(len(batch))]
        # transcribe_faster_whisper(audio_list, dic, ind = ind, model_name="faster-whisper")
        # print(dic)
        # transcribe_wav2vec2(audio_list, dic, ind = ind, model_name="wav2vec2", asr=wav_asr)
        transcribe_parakeet(audio_list, dic, ind = ind, model_name="parakeet")
        # transcribe_whisperx(audio_list, dic, ind = ind, model_name="whisperx")
        # transcribe_funasr(audio_list, dic, ind = ind, model_name="funasr")
        # print(dic.keys())
        with open(Path("ASR_transcriptions.json"), "w", encoding="utf-8") as f:
            # print(dic)
            try:
                dic_to_dump = {k: asdict(v) for k, v in dic.items()}
                # print(f"dic_to_dump: {dic_to_dump}")
                json.dump(dic_to_dump, f, ensure_ascii=False, indent=4)
            except KeyboardInterrupt:
                print("KeyboardInterrupt detected. Saving current transcriptions to ASR_transcriptions.json before exiting...")
                dic_to_dump = {k: asdict(v) for k, v in dic.items()}
                with open(Path("ASR_transcriptions.json"), "w", encoding="utf-8") as f:
                    json.dump(dic_to_dump, f, ensure_ascii=False, indent=4)
                print("Transcriptions saved. Exiting now.")
                sys.exit(0)
        # save timings after every batch
        temp = {f"{k}": dur for k, dur in TIMINGS.items()}
        
            
        with open(timing_file, "w") as f:
            json.dump(temp, f, indent=4)
        ind += BATCH_SIZE
    
    # transcribe_funasr(audio_list, dic)
    
    # {'transcribe_faster_whisper': 26.503355436001584, 'transcribe_wav2vec2': 6.618090244999621, 'transcribe_parakeet': 26.387749383000482, 'transcribe_whisperx': 24.36091910899995, 'transcribe_funasr': 73.51357104999988}


if __name__ == "__main__":
    main()