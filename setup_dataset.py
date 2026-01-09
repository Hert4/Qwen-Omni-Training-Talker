import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm

"""
in current code init in hf point to dataset
"""

# Đọc file parquet
parquet_files = list(Path('beyoru/misa-tts-LPAnh').glob('*.parquet')) # base dataset on hf

if not parquet_files:
    raise FileNotFoundError("Không tìm thấy file parquet nào!")

df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

# ✅ SHUFFLE DATASET
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

output_dir = Path('datasets/misa-tts-LPAnh/processed')
output_dir.mkdir(parents=True, exist_ok=True)
audio_dir = output_dir / 'audios'
audio_dir.mkdir(exist_ok=True)
jsonl_path = output_dir / 'train.jsonl'

with open(jsonl_path, 'w', encoding='utf-8') as f_out:
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Lưu audio
        audio_path = audio_dir / f'audio_{idx}.wav'
        with open(audio_path, 'wb') as f:
            f.write(row['audio']['bytes'])

        entry = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a high-quality Text-to-Speech (TTS) model. "
                        "Your task is to convert text into natural, fluent, "
                        "and realistic speech."
                    )
                },
                {
                    "role": "user",
                    "content": row["text"]
                },
                {
                    "role": "assistant",
                    "content": "<audio>"
                }
            ],
            "audios": [str(audio_path)],
            "speaker": "Ethan"
        }

        f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Đã xử lý {len(df)} mẫu dữ liệu")
