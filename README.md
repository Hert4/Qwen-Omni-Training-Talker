# Qwen-Omni Talker Fine-tuning for TTS

Fine-tuning the **Talker** component of Qwen-Omni models cho Text-to-Speech (TTS), giữ nguyên **Thinker** frozen (hoặc LoRA) và **Code2Wav / Token2Wav** đóng băng hoàn toàn.

Hỗ trợ hai dòng model:

| Model | Talker design | Script directory |
|---|---|---|
| `Qwen/Qwen2.5-Omni-3B` | Single-codebook (qwen-tts-tokenizer, vocab 8448) | [`Qwen2.5/`](Qwen2.5) |
| `Qwen/Qwen3-Omni-30B-A3B-Instruct` (MoE) | 16-codebook MTP + Mimi codec | [`Qwen3/`](Qwen3) |

---

## Architecture overview

```
text  ──►  Thinker  ──►  hidden states  ──►  Talker  ──►  codec tokens  ──►  Code2Wav / Token2Wav  ──►  waveform
         (frozen/LoRA)                      (TRAINED)                       (frozen)
```

- **Thinker**: language model backbone — frozen mặc định (hoặc LoRA r=8).
- **Talker**: autoregressive codec decoder — đây là phần được train.
- **MTP (code_predictor)**: chỉ có trên Qwen3-Omni, predict 15 codec layer phụ (layers 1–15) sau khi Talker predict layer 0. Train cùng Talker.
- **Code2Wav / Token2Wav**: vocoder — luôn frozen, không học được waveform stage này.

Loss = `talker_loss + 2.0 * mtp_avg_loss` (Qwen3) hoặc chỉ `talker_loss` (Qwen2.5).

---

## Repository layout

```
Qwen2.5/
  setup_dataset.py             # quangdung/ly-tts-dataset → JSONL 24kHz
  train_qwen25omni_tts.py      # LoRA train Thinker + Talker (single-codebook)
  inference.py                 # generate audio từ checkpoint

Qwen3/
  setup_dataset.py             # beyoru/misa-tts-LPAnh parquet → JSONL
  trainer.py                   # Talker-only full fine-tune (Thinker frozen)
  train_lora_full.py           # Thinker+Talker+MTP LoRA (recommended)
  merge_lora.py                # merge LoRA adapter → base model
  merge_lora_omni.py           # merge LoRA cho cả 3 module Omni
  interface.py                 # voicebot loop với Transformers / vLLM
```

---

## Setup

```bash
pip install torch transformers peft datasets soundfile librosa tqdm
# Qwen3-Omni cần transformers >= bản hỗ trợ Qwen3OmniMoe
# vLLM (optional, dùng cho inference.py): pip install vllm
```

Tải weights:

```bash
# base models
huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct  --local-dir models/Qwen3-Omni-30B-A3B-Instruct
huggingface-cli download kyutai/mimi                       --local-dir models/mimi
# hoặc với Qwen2.5
huggingface-cli download Qwen/Qwen2.5-Omni-3B              --local-dir models/Qwen2.5-Omni-3B
```

---

## Quick start — Qwen3-Omni (30B MoE)

```bash
# 1. Chuẩn bị dataset (Vietnamese TTS, single-speaker "Ethan")
cd Qwen3
python setup_dataset.py
# → datasets/misa-tts-LPAnh/processed/train.jsonl

# 2. Train (LoRA trên cả Thinker + Talker + MTP — recommended)
python train_lora_full.py
# checkpoints saved to models/checkpoint_step_{N}/

# 3. Merge LoRA vào base model
python merge_lora.py --checkpoint models/checkpoint_epoch_1 --output models/Qwen3-Omni-FT

# 4. Inference / voicebot loop
python interface.py
```

Train flag chính trong `train_lora_full.py`:

```python
BATCH_SIZE = 4
TRAIN_THINKER = True   # LoRA r=8 trên Thinker
TRAIN_TALKER  = True   # LoRA r=8 trên Talker
TRAIN_MTP     = True   # LoRA r=8 trên code_predictor
```

Nếu muốn fine-tune full-weight Talker (no LoRA) + Thinker frozen, dùng `trainer.py`.

---

## Quick start — Qwen2.5-Omni (3B)

```bash
cd Qwen2.5
python setup_dataset.py --dataset quangdung/ly-tts-dataset --speaker Chelsie
python train_qwen25omni_tts.py
python inference.py
```

Lưu ý design difference: Qwen2.5-Omni Talker chỉ dùng **một codebook duy nhất** → không có MTP loss. Vì qwen-tts-tokenizer encoder không public, script dùng Mimi codec làm proxy (xem docstring đầu file `train_qwen25omni_tts.py` — option `(2)` ưu tiên hơn nếu dataset có sẵn `codes` field).

---

## Dataset format

JSONL, mỗi sample một dòng:

```json
{
  "messages": [
    {"role": "system", "content": "You are a high-quality TTS model..."},
    {"role": "user",   "content": "Văn bản cần đọc"},
    {"role": "assistant", "content": "<audio>"}
  ],
  "audios": ["/abs/path/to/clip_24k.wav"],
  "speaker": "Ethan"
}
```

- Audio phải resample về **24kHz mono** (Mimi requirement).
- `speaker` phải nằm trong `talker_config.speaker_id` của model — Qwen3-Omni: `Ethan`, `Chelsie`, …

---

## Hardware

- **Qwen3-Omni 30B**: tối thiểu 1× H100/H200 80GB cho LoRA train, BF16 + sdpa. Full Talker fine-tune (`trainer.py`) cần thêm GPU memory cho optimizer state.
- **Qwen2.5-Omni 3B**: 1× A100 40GB là đủ.

Gradient checkpointing được bật mặc định trên Talker.

---

## Known issues / notes

- Token2Wav (Qwen2.5) và Code2Wav (Qwen3) frozen → nếu codec token distribution sau train drift quá xa pretrained, chất lượng audio sẽ tệ. Giữ LR Talker thấp (2e-5).
- MTP loss weight = 2.0 vì 15 layer phụ "easier" hơn layer 0 → cần upweight để force model học đầy đủ codec hierarchy.
- `interface.py` mặc định dùng Transformers backend; bật `USE_TRANSFORMERS = False` để chạy vLLM (nhanh hơn nhưng cần cài thêm).

---

## License

Inherit từ Qwen-Omni base models — xem repo Qwen/Qwen3-Omni và Qwen/Qwen2.5-Omni trên Hugging Face.
