from dataclasses import dataclass, asdict
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict

MODEL_NAMES = ["faster-whisper", "wav2vec2", "parakeet", "nemo", "granite", "phi-4"]

@dataclass
class BaseMixture:
    wave: np.ndarray
    overlap_mask: np.ndarray
    source_files: List[Path]
    transcript: List[tuple[str, str]]
    overlap_ratio_actual: float
    rms: float


@dataclass
class MixtureMeta:
    clip_id: str
    audio_path: str
    transcript: List[tuple[str, str]]
    overlap_ratio_target: float
    overlap_ratio_actual: float
    max_speakers: int
    snr_db: Optional[float]
    noise_type: Optional[str]
    overlap_mask_path: str
    source_files: List[str]
    noise_files: List[str]
    
@dataclass
class MixtureTranscription:
    clip_id: str
    transcript: Dict[str,List[tuple[str, str]]]