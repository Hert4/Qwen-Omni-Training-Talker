"""
Prepare LY TTS dataset for Qwen2.5-Omni-3B TTS fine-tuning.

Downloads quangdung/ly-tts-dataset from HF Hub, resamples audio to 24kHz,
filters by duration, and writes JSONL with the conversation format expected
by the trainer.

Output JSONL schema (one sample per line):
{
  "messages": [
    {"role": "system", "content": "<system prompt>"},
    {"role": "user", "content": "<the text to speak>"},
    {"role": "assistant", "content": "<the text to speak>"}
  ],
  "audios": ["/abs/path/to/clip_24k.wav"],
  "speaker": "Chelsie"  # female - matches LY's voice
}
"""

from pathlib import Path
import json
import argparse

import numpy as np
import soundfile as sf
import librosa
from datasets import load_dataset
from tqdm import tqdm


SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)


def normalize_text(text: str) -> str:
    """Light cleanup. Don't be aggressive - Vietnamese diacritics must survive."""
    text = text.strip()
    text = " ".join(text.split())  # collapse whitespace
    return text


def decode_audio_bytes(audio_bytes: bytes, audio_shape) -> np.ndarray:
    """Decode raw audio bytes to float32 array in [-1, 1].

    Auto-detects int16 vs float32 by comparing byte length against expected sample count.
    """
    n_samples = int(audio_shape[0]) if hasattr(audio_shape, "__len__") else int(audio_shape)
    bytes_per_sample = len(audio_bytes) // n_samples

    if bytes_per_sample == 2:
        wav = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif bytes_per_sample == 4:
        wav = np.frombuffer(audio_bytes, dtype=np.float32).copy()
    else:
        raise ValueError(
            f"Cannot infer audio dtype: {len(audio_bytes)} bytes / {n_samples} samples "
            f"= {bytes_per_sample} bytes per sample (expected 2 or 4)"
        )
    return wav


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="quangdung/ly-tts-dataset")
    parser.add_argument("--output_dir", type=Path, default=Path("datasets/ly-tts-vi"))
    parser.add_argument("--target_sr", type=int, default=24000)
    parser.add_argument("--min_duration", type=float, default=1.0,
                        help="Skip clips shorter than this (seconds)")
    parser.add_argument("--max_duration", type=float, default=15.0,
                        help="Skip clips longer than this (seconds) - long audio kills GPU memory")
    parser.add_argument("--speaker", default="Chelsie",
                        help="Pin to Qwen voice preset. LY is female -> Chelsie")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    audio_out = args.output_dir / "audio_24k"
    jsonl_out = args.output_dir / "processed" / "train.jsonl"
    audio_out.mkdir(parents=True, exist_ok=True)
    jsonl_out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset, split=args.split)
    print(f"Loaded {len(ds)} raw samples. Filtering and resampling...")

    # quangdung/ly-tts-dataset schema: 'file_name', 'audio_bytes' (raw bytes), 'audio_shape' (list),
    # 'sampling_rate' (int), 'text' (string). Audio is mono, already 24kHz.
    # Probe the first sample to find the text column (other TTS datasets may use 'transcription' etc.).
    sample0 = ds[0]
    text_key = None
    for candidate in ("text", "transcription", "sentence", "transcript", "normalized_text"):
        if candidate in sample0:
            text_key = candidate
            break
    if text_key is None:
        raise RuntimeError(
            f"Cannot find text column. Keys: {list(sample0.keys())}. "
            "Edit the script to set text_key manually."
        )
    print(f"Using text column: '{text_key}'")

    has_audio_bytes = "audio_bytes" in sample0
    has_audio_feature = "audio" in sample0
    if not (has_audio_bytes or has_audio_feature):
        raise RuntimeError(
            f"Cannot find audio column. Keys: {list(sample0.keys())}. "
            "Expected 'audio_bytes' (quangdung/ly-tts-dataset) or 'audio' (HF Audio feature)."
        )
    print(f"Audio schema: {'audio_bytes (raw)' if has_audio_bytes else 'audio (HF feature)'}")

    kept = 0
    skipped_short = 0
    skipped_long = 0
    skipped_empty = 0

    with open(jsonl_out, "w", encoding="utf-8") as fout:
        iterator = ds if args.max_samples is None else ds.select(range(min(args.max_samples, len(ds))))
        for idx, row in enumerate(tqdm(iterator, desc="Processing")):
            text = normalize_text(str(row[text_key]))
            if not text:
                skipped_empty += 1
                continue

            if has_audio_bytes:
                wav = decode_audio_bytes(row["audio_bytes"], row["audio_shape"])
                sr = int(row["sampling_rate"])
            else:
                audio_data = row["audio"]
                wav = np.asarray(audio_data["array"], dtype=np.float32)
                sr = audio_data["sampling_rate"]

            # Mono
            if wav.ndim > 1:
                wav = wav.mean(axis=1)

            duration = len(wav) / sr
            if duration < args.min_duration:
                skipped_short += 1
                continue
            if duration > args.max_duration:
                skipped_long += 1
                continue

            # Resample to 24kHz (BigVGAN sample rate)
            if sr != args.target_sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=args.target_sr)

            # Peak normalize gently (avoid clipping later)
            peak = np.abs(wav).max()
            if peak > 0.99:
                wav = wav * (0.99 / peak)

            out_path = audio_out / f"ly_{idx:06d}.wav"
            sf.write(out_path, wav, args.target_sr, subtype="PCM_16")

            record = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": text},
                ],
                "audios": [str(out_path.resolve())],
                "speaker": args.speaker,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print()
    print(f"Done. Kept {kept} samples.")
    print(f"  - skipped (empty text): {skipped_empty}")
    print(f"  - skipped (too short < {args.min_duration}s): {skipped_short}")
    print(f"  - skipped (too long > {args.max_duration}s): {skipped_long}")
    print(f"JSONL: {jsonl_out}")
    print(f"Audio: {audio_out}")


if __name__ == "__main__":
    main()
