from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "Output" / "manifest.csv"
DEFAULT_AUDIO_DIR = PROJECT_ROOT / "Output" / "audio"


def load_manifest_clip_ids(manifest_path: Path) -> set[str]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "clip_id" not in reader.fieldnames:
            raise ValueError(f"{manifest_path} does not contain a clip_id column")

        clip_ids = {
            (row.get("clip_id") or "").strip()
            for row in reader
            if (row.get("clip_id") or "").strip()
        }

    if not clip_ids:
        raise ValueError(f"No clip IDs found in {manifest_path}")
    return clip_ids


def find_audio_files(audio_dir: Path) -> list[Path]:
    return sorted(path for path in audio_dir.iterdir() if path.is_file() and path.suffix.lower() == ".wav")


def preview_paths(label: str, paths: list[Path], limit: int) -> None:
    if not paths:
        return

    print(f"\n{label}:")
    for path in paths[:limit]:
        print(f"  {path}")
    remaining = len(paths) - limit
    if remaining > 0:
        print(f"  ... {remaining} more")


def prune_audio_to_manifest(
    manifest_path: Path,
    audio_dir: Path,
    delete_extra: bool,
    move_extra_dir: Path | None,
    preview_limit: int,
) -> None:
    manifest_path = manifest_path.resolve()
    audio_dir = audio_dir.resolve()
    move_extra_dir = move_extra_dir.resolve() if move_extra_dir is not None else None

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    manifest_clip_ids = load_manifest_clip_ids(manifest_path)
    audio_files = find_audio_files(audio_dir)
    audio_by_clip_id = {path.stem: path for path in audio_files}

    extra_audio_files = [path for clip_id, path in audio_by_clip_id.items() if clip_id not in manifest_clip_ids]
    missing_audio_files = [
        audio_dir / f"{clip_id}.wav"
        for clip_id in sorted(manifest_clip_ids)
        if clip_id not in audio_by_clip_id
    ]

    print(f"Manifest clip IDs: {len(manifest_clip_ids)}")
    print(f"Audio .wav files: {len(audio_files)}")
    print(f"Matched audio files: {len(audio_files) - len(extra_audio_files)}")
    print(f"Extra audio files: {len(extra_audio_files)}")
    print(f"Missing manifest audio files: {len(missing_audio_files)}")

    preview_paths("Extra audio files not present in manifest", extra_audio_files, preview_limit)
    preview_paths("Manifest clip IDs missing .wav files", missing_audio_files, preview_limit)

    if move_extra_dir is not None:
        move_extra_dir.mkdir(parents=True, exist_ok=True)
        for path in extra_audio_files:
            shutil.move(str(path), str(move_extra_dir / path.name))
        print(f"\nMoved {len(extra_audio_files)} extra files to {move_extra_dir}")
    elif delete_extra:
        for path in extra_audio_files:
            path.unlink()
        print(f"\nDeleted {len(extra_audio_files)} extra files from {audio_dir}")
    else:
        print("\nDry run only. Use --delete-extra or --move-extra-dir to keep only manifest clips.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep Output/audio .wav files whose file stem appears as a clip_id in Output/manifest.csv.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--delete-extra", action="store_true", help="Delete .wav files not listed in manifest.csv.")
    parser.add_argument(
        "--move-extra-dir",
        type=Path,
        default=None,
        help="Move extra .wav files here instead of deleting them.",
    )
    parser.add_argument("--preview-limit", type=int, default=20)
    args = parser.parse_args()

    if args.delete_extra and args.move_extra_dir is not None:
        parser.error("--delete-extra and --move-extra-dir cannot be used together")
    return args


def main() -> None:
    args = parse_args()
    prune_audio_to_manifest(
        manifest_path=args.manifest,
        audio_dir=args.audio_dir,
        delete_extra=args.delete_extra,
        move_extra_dir=args.move_extra_dir,
        preview_limit=max(0, args.preview_limit),
    )


if __name__ == "__main__":
    main()
