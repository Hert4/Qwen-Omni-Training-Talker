"""
Qwen2.5-Omni-3B TTS fine-tuning with LoRA on Thinker + Talker.

Key architectural differences from the Qwen3-Omni-30B-MoE script this is adapted from:
- Qwen2.5-Omni Talker uses a SINGLE-codebook qwen-tts-tokenizer (vocab_size=8448),
  not the 16-codebook MTP design of Qwen3-Omni. So we drop the code_predictor /
  MTP loss entirely.
- Token2Wav is a DiT + BigVGAN module (frozen here). No `code2wav.num_quantizers`.
- Talker special tokens come from `talker.config`:
    tts_codec_start_token_id, tts_codec_end_token_id, tts_codec_pad_token_id,
    tts_text_start_token_id, tts_text_end_token_id, tts_text_pad_token_id
- Since the qwen-tts-tokenizer ENCODER is not public, we cannot recover ground-truth
  codec tokens from waveform. Two options here:
  (1) MIMI_PATH: use Kyutai Mimi encoder and map indices into the codec vocab.
      Risky - the embedding space won't align with the frozen Token2Wav decoder,
      audio quality will degrade.
  (2) HF_DATASET_HAS_CODES: if your dataset already ships codec token sequences
      under a 'codes' field, use those directly. Best path if available.
  Default is (1). See encode_audio_to_codes() to swap.

Tested target: 1x H100 / H200 with BF16, sdpa attention.
"""

from pathlib import Path
from dataclasses import dataclass
import json
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    get_cosine_schedule_with_warmup,
)
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model, TaskType


# ==========================================================================
# Configuration (override via CLI flags - see bottom of file)
# ==========================================================================
@dataclass
class Config:
    # Paths
    dataset_path: Path = Path("datasets/ly-tts-vi/processed/train.jsonl")
    model_path: str = "Qwen/Qwen2.5-Omni-3B"
    mimi_path: str = "kyutai/mimi"  # used as audio -> codec proxy; see notes above
    output_dir: Path = Path("checkpoints/qwen25omni-ly-tts")

    # Training
    batch_size: int = 2
    num_epochs: int = 3
    gradient_accumulation_steps: int = 4
    learning_rate_thinker: float = 5e-6
    learning_rate_talker: float = 2e-5
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    save_every_n_steps: int = 250
    log_every_n_steps: int = 1
    max_samples: int | None = None
    seed: int = 42

    # What to train
    train_thinker: bool = True
    train_talker: bool = True

    # Speaker (Qwen2.5-Omni-3B has Chelsie female / Ethan male)
    default_speaker: str = "Chelsie"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    # Memory
    use_gradient_checkpointing: bool = True
    attn_implementation: str = "sdpa"
    dtype: str = "bfloat16"

    # Audio
    target_sr: int = 24000
    max_audio_seconds: float = 12.0  # safety cap during loading


CFG = Config()


# ==========================================================================
# Dataset
# ==========================================================================
class TTSDataset(Dataset):
    def __init__(self, jsonl_path: Path, max_samples: int | None = None):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                d = json.loads(line)
                self.samples.append({
                    "messages": d["messages"],
                    "audio_path": Path(d["audios"][0]),
                    "speaker": d.get("speaker", CFG.default_speaker),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ==========================================================================
# Audio -> codec tokens
# ==========================================================================
def load_codec_encoder(device, dtype):
    """
    Loads Kyutai Mimi as audio -> discrete code encoder.

    Note: Mimi natively produces multi-codebook output (8 quantizers). Qwen2.5-Omni
    Talker uses a single-codebook codec. We flatten Mimi codes by taking the first
    codebook only and re-scaling indices into the Talker codec vocab range.

    This is a lossy approximation. For best results, replace this with the actual
    qwen-tts-tokenizer encoder once/if Alibaba releases it, or use a dataset that
    ships pre-computed codes.
    """
    from transformers import MimiModel, AutoFeatureExtractor
    mimi = MimiModel.from_pretrained(CFG.mimi_path, torch_dtype=dtype).to(device)
    mimi.eval()
    for p in mimi.parameters():
        p.requires_grad_(False)
    feat = AutoFeatureExtractor.from_pretrained(CFG.mimi_path)
    return mimi, feat


@torch.no_grad()
def encode_audio_to_codes(audio_path: Path, mimi, feat_extractor, device, codec_vocab_size: int):
    """
    Returns a 1-D LongTensor of codec token ids, shape (T,).
    Values are bounded into [0, codec_vocab_size - num_special_tokens) so they do
    not collide with codec_start/end/pad/mask special token ids.
    """
    wav, sr = sf.read(audio_path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != CFG.target_sr:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=CFG.target_sr)
        sr = CFG.target_sr

    feats = feat_extractor(wav, sampling_rate=sr, return_tensors="pt")
    audio_t = torch.as_tensor(feats["input_values"], dtype=mimi.dtype)
    if audio_t.ndim == 2:
        audio_t = audio_t.unsqueeze(1)
    audio_t = audio_t.to(device)

    out = mimi.encode(audio_t)
    codes = out.audio_codes  # (B=1, num_q, T)

    # Take first codebook only -> (T,)
    codes_q0 = codes[0, 0, :].long().cpu()

    # Bound into the safe regular-token range to avoid colliding with the
    # special token ids near the top of the vocab (start=8293, end=8294, pad=8292, mask=8296).
    # 8192 is the natural ceiling for a single-codebook codec of vocab 8448.
    safe_ceiling = min(8192, codec_vocab_size - 256)
    codes_q0 = codes_q0 % safe_ceiling
    return codes_q0  # 1-D LongTensor


# ==========================================================================
# Collate
# ==========================================================================
def make_collate_fn(processor, mimi, feat_extractor, device, codec_vocab_size):
    def collate(batch_samples):
        # Build prompt text via chat template
        # For TTS we want the assistant turn to contain the text the speaker says.
        formatted = []
        speakers = []
        audio_paths = []
        for s in batch_samples:
            text = processor.apply_chat_template(
                s["messages"], add_generation_prompt=False, tokenize=False
            )
            formatted.append(text)
            speakers.append(s["speaker"])
            audio_paths.append(s["audio_path"])

        # Tokenize prompts (no real audio/image/video input)
        batch = processor(
            text=formatted,
            audio=None,
            images=None,
            videos=None,
            return_tensors="pt",
            padding=True,
        )

        # Encode target audio -> codec token sequence
        codes_list = [
            encode_audio_to_codes(p, mimi, feat_extractor, device, codec_vocab_size)
            for p in audio_paths
        ]
        code_lens = [len(c) for c in codes_list]
        max_len = max(code_lens)

        # Pad with codec_pad_token_id, but we mark padding positions as -100 in labels
        # later so they don't contribute to loss.
        # Here just pad with 0; the label mask will handle ignore.
        padded = torch.zeros((len(codes_list), max_len), dtype=torch.long)
        for i, c in enumerate(codes_list):
            padded[i, : len(c)] = c

        return {
            "thinker_inputs": {
                "input_ids": batch["input_ids"],
                "attention_mask": batch.get("attention_mask"),
            },
            "target_codes": padded,
            "code_lens": code_lens,
            "speakers": speakers,
        }

    return collate


# ==========================================================================
# Talker forward + loss for a single sample
# ==========================================================================
def compute_talker_loss(model, thinker_hidden, thinker_input_ids,
                        target_codes, code_len, speaker, device, batch_idx=0):
    """
    Build a Talker training step:

      [Talker prefix: thinker context + codec_start]  ->  predict codec tokens  ->  codec_end

    Inputs:
      thinker_hidden: (B, T_text, H) last-layer thinker hidden states (BF16)
      thinker_input_ids: (B, T_text) - used to find positions of im_start, role tokens, etc.
      target_codes: (B, T_code_max) padded codec token ids
      code_len: int, real length of this sample's codec tokens
    """
    talker = model.talker
    tcfg = talker.config

    codec_start = tcfg.tts_codec_start_token_id
    codec_end = tcfg.tts_codec_end_token_id
    codec_pad = tcfg.tts_codec_pad_token_id

    # Slice this batch element
    h = thinker_hidden[batch_idx : batch_idx + 1]  # (1, T_text, H)
    sample_codes = target_codes[batch_idx, :code_len].to(device)  # (T_code,)

    # Talker text projection - maps thinker hidden -> talker hidden dim
    # In Qwen2.5-Omni, talker has a `text_projection` (or equivalent) that bridges
    # thinker's hidden_size into talker's embedding_size. Both happen to be the same
    # in 3B (2048) and 7B (3584), so this is often identity-like, but we still
    # call it to keep the contract correct.
    if hasattr(talker, "text_projection") and talker.text_projection is not None:
        text_hidden = talker.text_projection(h)
    else:
        text_hidden = h

    # Talker codec embedding (input embeddings of the talker LM)
    talker_embed_layer = talker.get_input_embeddings()
    # Unwrap PEFT base layer if LoRA-wrapped
    if hasattr(talker_embed_layer, "base_layer"):
        talker_embed_layer_base = talker_embed_layer.base_layer
    else:
        talker_embed_layer_base = talker_embed_layer

    codec_start_embed = talker_embed_layer_base(
        torch.tensor([[codec_start]], device=device)
    )  # (1, 1, H_talker)

    # Build full prefix: thinker hidden (projected) + codec_start
    prefix_embeds = torch.cat([text_hidden, codec_start_embed], dim=1)  # (1, T_text+1, H)
    prefix_len = prefix_embeds.shape[1]

    # Body: embeddings of target codec tokens (teacher-forced, all but last)
    body_codes = sample_codes.unsqueeze(0)  # (1, T_code)
    body_embeds = talker_embed_layer_base(body_codes)  # (1, T_code, H)

    # Append codec_end
    codec_end_embed = talker_embed_layer_base(
        torch.tensor([[codec_end]], device=device)
    )
    full_embeds = torch.cat([prefix_embeds, body_embeds, codec_end_embed], dim=1)
    # Shape now: (1, T_text + 1 + T_code + 1, H)

    # Build labels:
    # - prefix positions -> -100 (no loss on text-side)
    # - body positions -> the codec target ids
    # - last position predicts codec_end
    pad_label = torch.full((1, prefix_len - 1), -100, dtype=torch.long, device=device)
    body_labels = body_codes.to(device)
    end_label = torch.tensor([[codec_end]], device=device)
    # We're going to compute loss with the standard shift: shift_logits[:-1] vs shift_labels[1:]
    # So labels must be aligned with the *input* sequence.
    labels = torch.cat([pad_label, body_labels, end_label], dim=1)
    # full_embeds length = prefix_len + T_code + 1
    # labels length should match: (prefix_len - 1) + T_code + 1 = prefix_len + T_code
    # We need it to be full_embeds.length so shift cuts down correctly.
    # full_embeds.shape[1] = prefix_len + T_code + 1
    # labels.shape[1] = prefix_len + T_code
    # That's one short; we prepend one extra -100 so they match:
    labels = torch.cat(
        [torch.tensor([[-100]], device=device, dtype=torch.long), labels], dim=1
    )
    assert labels.shape[1] == full_embeds.shape[1], (
        f"Label/input length mismatch: {labels.shape[1]} vs {full_embeds.shape[1]}"
    )

    attn_mask = torch.ones(full_embeds.shape[:2], dtype=torch.long, device=device)

    # Forward through talker LM
    talker_out = talker(
        inputs_embeds=full_embeds,
        attention_mask=attn_mask,
        return_dict=True,
    )
    logits = talker_out.logits  # (1, T, V_talker)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Mask out pad token in labels just in case
    shift_labels = shift_labels.masked_fill(shift_labels == codec_pad, -100)

    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )
    return loss


# ==========================================================================
# LoRA application
# ==========================================================================
def apply_lora(submodule, name: str):
    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM if name == "thinker" else TaskType.FEATURE_EXTRACTION,
        r=CFG.lora_r,
        lora_alpha=CFG.lora_alpha,
        lora_dropout=CFG.lora_dropout,
        target_modules=list(CFG.lora_target_modules),
        bias="none",
    )
    wrapped = get_peft_model(submodule, cfg)
    print(f"\n=== LoRA on {name} ===")
    wrapped.print_trainable_parameters()
    return wrapped


def freeze_all(module, name: str):
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()
    print(f"Frozen: {name}")


# ==========================================================================
# Main training loop
# ==========================================================================
def train(cfg: Config):
    torch.manual_seed(cfg.seed)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[cfg.dtype]

    print("Loading Qwen2.5-Omni-3B ...")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        cfg.model_path,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation=cfg.attn_implementation,
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(cfg.model_path)

    device = next(model.parameters()).device
    if device.type == "meta":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Freeze the audio decoder (Token2Wav: DiT + BigVGAN)
    if hasattr(model, "token2wav"):
        freeze_all(model.token2wav, "token2wav (DiT + BigVGAN)")

    # LoRA on Thinker
    if cfg.train_thinker:
        model.thinker = apply_lora(model.thinker, "thinker")
        model.thinker.train()
        if cfg.use_gradient_checkpointing:
            model.thinker.gradient_checkpointing_enable()
            # PEFT model needs this for grad ckpt to work properly
            if hasattr(model.thinker, "enable_input_require_grads"):
                model.thinker.enable_input_require_grads()
    else:
        freeze_all(model.thinker, "thinker")

    # LoRA on Talker
    if cfg.train_talker:
        # Talker has an internal `.model` (the transformer) - we LoRA-wrap that
        # part so input/output embeddings and the LM head are reachable from the
        # outer Talker module.
        talker_inner = model.talker.model if hasattr(model.talker, "model") else model.talker
        wrapped = apply_lora(talker_inner, "talker")
        if hasattr(model.talker, "model"):
            model.talker.model = wrapped
        else:
            model.talker = wrapped
        model.talker.train()
        if cfg.use_gradient_checkpointing:
            inner = model.talker.model if hasattr(model.talker, "model") else model.talker
            if hasattr(inner, "gradient_checkpointing_enable"):
                inner.gradient_checkpointing_enable()
            if hasattr(inner, "enable_input_require_grads"):
                inner.enable_input_require_grads()
    else:
        freeze_all(model.talker, "talker")

    # Codec encoder (frozen)
    print("Loading Mimi codec encoder (used as audio -> token proxy)...")
    mimi, feat_extractor = load_codec_encoder(device, dtype)
    codec_vocab_size = model.talker.config.vocab_size  # 8448

    # Dataset
    dataset = TTSDataset(cfg.dataset_path, max_samples=cfg.max_samples)
    print(f"Dataset: {len(dataset)} samples")

    collate_fn = make_collate_fn(processor, mimi, feat_extractor, device, codec_vocab_size)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False,
    )

    # Optimizer with per-module LR
    param_groups = []
    if cfg.train_thinker:
        thinker_params = [p for p in model.thinker.parameters() if p.requires_grad]
        param_groups.append({"params": thinker_params, "lr": cfg.learning_rate_thinker})
        print(f"Thinker trainable params: {sum(p.numel() for p in thinker_params):,}")
    if cfg.train_talker:
        talker_params = [p for p in model.talker.parameters() if p.requires_grad]
        param_groups.append({"params": talker_params, "lr": cfg.learning_rate_talker})
        print(f"Talker trainable params: {sum(p.numel() for p in talker_params):,}")

    if not param_groups:
        raise ValueError("Nothing to train. Enable at least one of train_thinker/train_talker.")

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=0.01)

    total_steps = (len(dataloader) * cfg.num_epochs) // cfg.gradient_accumulation_steps
    warmup_steps = max(1, int(cfg.warmup_ratio * total_steps))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    print(f"\n{'='*60}")
    print(f"Total steps: {total_steps} | Warmup: {warmup_steps}")
    print(f"Effective batch size: {cfg.batch_size * cfg.gradient_accumulation_steps}")
    print(f"{'='*60}\n")

    global_step = 0
    for epoch in range(cfg.num_epochs):
        epoch_losses = {"thinker": 0.0, "talker": 0.0, "total": 0.0}
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")
        for batch_idx, batch_data in enumerate(pbar):
            if batch_idx % cfg.gradient_accumulation_steps == 0:
                optimizer.zero_grad(set_to_none=True)

            thinker_inputs = batch_data["thinker_inputs"]
            input_ids = thinker_inputs["input_ids"].to(device)
            attn_mask = thinker_inputs["attention_mask"]
            if attn_mask is not None:
                attn_mask = attn_mask.to(device)

            target_codes = batch_data["target_codes"].to(device)
            code_lens = batch_data["code_lens"]
            speakers = batch_data["speakers"]

            # Thinker labels: standard causal LM, mask padding
            labels = input_ids.clone()
            if processor.tokenizer.pad_token_id is not None:
                labels[labels == processor.tokenizer.pad_token_id] = -100

            # --- Thinker forward ---
            # Need last-layer hidden states for talker conditioning.
            thinker_out = model.thinker(
                input_ids=input_ids,
                attention_mask=attn_mask,
                labels=labels if cfg.train_thinker else None,
                output_hidden_states=True,
                return_dict=True,
            )

            # last-layer hidden states (B, T, H)
            last_hidden = thinker_out.hidden_states[-1]

            thinker_loss = thinker_out.loss if cfg.train_thinker else torch.tensor(0.0, device=device)

            # --- Talker loss per sample (cannot vectorize easily) ---
            bsz = input_ids.shape[0]
            talker_loss_accum = torch.tensor(0.0, device=device)
            n_ok = 0
            for i in range(bsz):
                try:
                    tl = compute_talker_loss(
                        model=model,
                        thinker_hidden=last_hidden.detach() if not cfg.train_thinker else last_hidden,
                        thinker_input_ids=input_ids,
                        target_codes=target_codes,
                        code_len=code_lens[i],
                        speaker=speakers[i],
                        device=device,
                        batch_idx=i,
                    )
                    talker_loss_accum = talker_loss_accum + tl
                    n_ok += 1
                except Exception as e:
                    print(f"\n[warn] sample {i} skipped: {e}")
                    continue
            if n_ok == 0:
                print("[warn] entire batch failed, skipping")
                continue
            talker_loss = talker_loss_accum / n_ok

            # Combined
            loss_terms = []
            if cfg.train_thinker:
                loss_terms.append(thinker_loss)
            if cfg.train_talker:
                loss_terms.append(talker_loss)
            total_loss = sum(loss_terms) / cfg.gradient_accumulation_steps
            total_loss.backward()

            if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                all_params = [p for g in param_groups for p in g["params"]]
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                global_step += 1

            # Logging
            epoch_losses["thinker"] += thinker_loss.item() if cfg.train_thinker else 0.0
            epoch_losses["talker"] += talker_loss.item() if cfg.train_talker else 0.0
            epoch_losses["total"] += total_loss.item() * cfg.gradient_accumulation_steps
            n_batches += 1

            if batch_idx % cfg.log_every_n_steps == 0:
                lr_now = scheduler.get_last_lr()[0]
                msg = {
                    "ep": epoch + 1,
                    "step": batch_idx,
                    "gstep": global_step,
                    "lr": f"{lr_now:.2e}",
                }
                if cfg.train_thinker:
                    msg["thinker"] = f"{thinker_loss.item():.4f}"
                if cfg.train_talker:
                    msg["talker"] = f"{talker_loss.item():.4f}"
                msg["total"] = f"{total_loss.item() * cfg.gradient_accumulation_steps:.4f}"
                pbar.set_postfix(msg)

            # Checkpoint
            if global_step > 0 and global_step % cfg.save_every_n_steps == 0 \
                    and (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                save_checkpoint(model, processor, cfg, tag=f"step_{global_step}")

        # End of epoch summary
        print(f"\n=== Epoch {epoch+1} done ===")
        for k, v in epoch_losses.items():
            print(f"  avg {k}: {v / max(1, n_batches):.4f}")
        save_checkpoint(model, processor, cfg, tag=f"epoch_{epoch+1}")

    print("\nTraining complete.")


def save_checkpoint(model, processor, cfg: Config, tag: str):
    out = cfg.output_dir / tag
    out.mkdir(parents=True, exist_ok=True)
    if cfg.train_thinker:
        model.thinker.save_pretrained(out / "thinker_lora")
    if cfg.train_talker:
        # Save whichever submodule got LoRA-wrapped
        target = model.talker.model if hasattr(model.talker, "model") else model.talker
        target.save_pretrained(out / "talker_lora")
    processor.save_pretrained(out)
    with open(out / "training_config.json", "w") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in cfg.__dict__.items()}, f, indent=2)
    print(f"[ckpt] saved -> {out}")


# ==========================================================================
# CLI
# ==========================================================================
def parse_args_into_cfg():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", type=Path, default=CFG.dataset_path)
    p.add_argument("--model_path", default=CFG.model_path)
    p.add_argument("--mimi_path", default=CFG.mimi_path)
    p.add_argument("--output_dir", type=Path, default=CFG.output_dir)
    p.add_argument("--batch_size", type=int, default=CFG.batch_size)
    p.add_argument("--num_epochs", type=int, default=CFG.num_epochs)
    p.add_argument("--gradient_accumulation_steps", type=int, default=CFG.gradient_accumulation_steps)
    p.add_argument("--lr_thinker", type=float, default=CFG.learning_rate_thinker)
    p.add_argument("--lr_talker", type=float, default=CFG.learning_rate_talker)
    p.add_argument("--save_every_n_steps", type=int, default=CFG.save_every_n_steps)
    p.add_argument("--max_samples", type=int, default=CFG.max_samples)
    p.add_argument("--no_thinker", action="store_true", help="Freeze thinker, only train talker")
    p.add_argument("--no_talker", action="store_true", help="Freeze talker, only train thinker")
    p.add_argument("--lora_r", type=int, default=CFG.lora_r)
    p.add_argument("--lora_alpha", type=int, default=CFG.lora_alpha)
    p.add_argument("--no_grad_ckpt", action="store_true")
    p.add_argument("--speaker", default=CFG.default_speaker)
    a = p.parse_args()

    CFG.dataset_path = a.dataset_path
    CFG.model_path = a.model_path
    CFG.mimi_path = a.mimi_path
    CFG.output_dir = a.output_dir
    CFG.batch_size = a.batch_size
    CFG.num_epochs = a.num_epochs
    CFG.gradient_accumulation_steps = a.gradient_accumulation_steps
    CFG.learning_rate_thinker = a.lr_thinker
    CFG.learning_rate_talker = a.lr_talker
    CFG.save_every_n_steps = a.save_every_n_steps
    CFG.max_samples = a.max_samples
    CFG.train_thinker = not a.no_thinker
    CFG.train_talker = not a.no_talker
    CFG.lora_r = a.lora_r
    CFG.lora_alpha = a.lora_alpha
    CFG.use_gradient_checkpointing = not a.no_grad_ckpt
    CFG.default_speaker = a.speaker
    return CFG


if __name__ == "__main__":
    cfg = parse_args_into_cfg()
    train(cfg)
