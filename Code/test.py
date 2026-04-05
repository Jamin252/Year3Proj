from typing import Dict
from pathlib import Path

root = Path("/home/jamin/Year3Proj/Data/txt")
transcript_map: Dict[str, str] = {}
for txt_path in root.rglob("*.txt"):
    if txt_path.is_file():
        utt_id = txt_path.stem
        transcript = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
        transcript_map[utt_id] = transcript
print(transcript_map)