"""
Modified training script for Text-to-Speech fine-tuning on Qwen3-Omni
Key changes:
1. Remove audio input processing
2. Use text-only input for Thinker
3. Train Talker to generate voice from text
"""

from pathlib import Path
import json
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoFeatureExtractor,
    MimiModel,
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
)
from peft import LoraConfig, get_peft_model, TaskType

# Configuration
DATASET_PATH = Path("datasets/misa-tts-LPAnh/processed/train.jsonl")
MODEL_PATH = "models/Qwen3-Omni-30B-A3B-Instruct"
MIMI_REPO_ID = "models/mimi"
OUTPUT_DIR = Path("models")
SPEAKER = "Ethan"
NUM_CODE_GROUPS = 16
USE_AUDIO_IN_VIDEO = False

# ===== TRAINING HYPERPARAMETERS =====
MAX_SAMPLES = None  # None = use all data, hoặc đặt số cụ thể như 1000
NUM_EPOCHS = 1      # Số epoch train
STEPS_PER_EPOCH = None  # None = train hết dataset, hoặc đặt số steps như 500
GRADIENT_ACCUMULATION_STEPS = 4  # Tăng nếu thiếu VRAM
LEARNING_RATE_THINKER = 1e-4     # LR cho Thinker (LoRA)
LEARNING_RATE_TALKER = 2e-4      # LR cho Talker
SAVE_EVERY_N_STEPS = 500         # Lưu checkpoint mỗi N steps
LORA_R = 64                      # LoRA rank (giảm xuống 8 nếu thiếu VRAM)
LORA_ALPHA = 128                  # LoRA alpha

def load_dataset(jsonl_path: Path, max_samples: int = None):
    """Load TTS dataset from JSONL - uses original messages format"""
    samples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            data = json.loads(line)
            
            # Use original messages (already has system, user, assistant)
            if 'messages' in data and 'audios' in data:
                samples.append({
                    'messages': data['messages'],  # Keep original format
                    'audio_path': Path(data['audios'][0]),
                    'speaker': data.get('speaker', SPEAKER)
                })
    return samples

def encode_audio_to_codes(audio_path: Path, feature_extractor, mimi_model, device):
    """Convert audio file to codec codes"""
    import librosa
    
    # Read audio and resample to 24kHz if needed
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    
    # Resample to 24kHz if needed (Mimi requires 24kHz)
    target_sr = 24000
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    
    audio_inputs = feature_extractor(audio, sampling_rate=sr, return_tensors="pt")
    audio_tensor = torch.as_tensor(audio_inputs["input_values"], dtype=mimi_model.dtype)
    
    if audio_tensor.ndim == 2:
        audio_tensor = audio_tensor.unsqueeze(1)
    
    audio_tensor = audio_tensor.to(device)
    
    with torch.no_grad():
        codes = mimi_model.encode(audio_tensor).audio_codes
    
    return codes.to(device)

def align_codebook_dim(codes: torch.Tensor, target_quantizers: int):
    """Align codec codes to target quantizer dimension"""
    current = codes.shape[1]
    if current == target_quantizers:
        return codes
    if current < target_quantizers:
        raise ValueError(f"Codes have {current} quantizers but need {target_quantizers}")
    return codes[:, :target_quantizers, :]

def prepare_tts_conversation(messages: list, processor):
    """Prepare conversation for TTS using original messages from dataset"""
    
    # Apply chat template directly to original messages
    formatted_text = processor.apply_chat_template(
        messages, 
        add_generation_prompt=False, 
        tokenize=False
    )
    
    # Process (no audio/image/video inputs for TTS)
    batch = processor(
        text=formatted_text,
        audio=None,
        images=None,
        videos=None,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
    )
    
    return batch, messages

def build_talker_prefix_tts(model, thinker_outputs, input_ids, speaker_name, device):
    """Build Talker prefix for TTS (no audio input processing)"""
    config = model.config
    thinker_embed = thinker_outputs.hidden_states[0].to(device)
    accept_layer = config.talker_config.accept_hidden_layer
    thinker_hidden = thinker_outputs.hidden_states[accept_layer].to(device)
    
    im_start_positions = torch.nonzero(input_ids[0] == config.im_start_token_id).view(-1)
    im_start_indexes = torch.cat(
        (im_start_positions, torch.tensor([input_ids.shape[1]], device=device)),
        dim=0,
    )
    
    # Multimodal mask (all False for text-only TTS)
    multimodal_mask = torch.zeros_like(input_ids, dtype=torch.bool).to(device)
    
    # Special tokens for Talker
    talker_special_tokens = torch.tensor(
        [[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]],
        device=device,
        dtype=input_ids.dtype,
    )
    
    thinker_embeddings = model.thinker.get_input_embeddings()
    if hasattr(thinker_embeddings, 'base_layer'):
        thinker_embeddings = thinker_embeddings.base_layer
    
    tts_bos_embed, tts_eos_embed, tts_pad_embed = (
        model.talker.text_projection(thinker_embeddings(talker_special_tokens))
        .to(device)
        .chunk(3, dim=1)
    )
    
    speaker_id = config.talker_config.speaker_id.get(speaker_name.lower())
    if speaker_id is None:
        raise ValueError(f"Speaker {speaker_name} not found")
    
    talker_input_embeds, talker_input_ids = [], []
    trailing_text_hidden = None
    
    for i in range(len(im_start_indexes) - 1):
        im_start_index = im_start_indexes[i]
        segment_end_index = im_start_indexes[i + 1]
        role_token = input_ids[0][im_start_index + 1]
        
        if role_token == config.user_token_id:
            # User part: use the same logic as model's method but for text-only
            user_part = model._get_talker_user_parts(
                im_start_index,
                segment_end_index,
                multimodal_mask,
                thinker_hidden,
                thinker_embed,
            )
            talker_input_embeds.append(user_part)
            talker_input_ids.append(input_ids[:, im_start_index:segment_end_index])
            
        elif role_token == config.assistant_token_id and i == len(im_start_indexes) - 2:
            # Assistant part with TTS preparation
            assistant_embeds, assistant_ids, trailing_text_hidden = model._get_talker_assistant_parts(
                im_start_index,
                segment_end_index,
                speaker_id,
                thinker_embed,
                tts_pad_embed,
                tts_bos_embed,
                tts_eos_embed,
            )
            talker_input_embeds.append(assistant_embeds)
            talker_input_ids.append(assistant_ids)
    
    if trailing_text_hidden is None:
        raise RuntimeError("Failed to build trailing_text_hidden")
    
    talker_input_embed = torch.cat([embed.to(device) for embed in talker_input_embeds], dim=1)
    talker_input_id = torch.cat([ids.to(device) for ids in talker_input_ids], dim=1)
    
    return talker_input_embed, talker_input_id, trailing_text_hidden.to(device), tts_pad_embed

def train_tts():
    """Main training function for TTS fine-tuning"""
    
    # Load model and processor
    print("Loading model...")
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype="auto",
        device_map="auto",
        attn_implementation="sdpa",
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
    feature_extractor = AutoFeatureExtractor.from_pretrained(MIMI_REPO_ID)
    
    # Setup device
    talker_device = next(model.talker.parameters()).device
    if talker_device.type == "meta":
        execution_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        execution_device = talker_device
    
    print(f"Using device: {execution_device}")
    
    # Load Mimi codec model
    mimi_model = MimiModel.from_pretrained(
        MIMI_REPO_ID, 
        torch_dtype=model.talker.dtype
    ).to(execution_device)
    mimi_model.eval()
    for param in mimi_model.parameters():
        param.requires_grad_(False)
    
    # Freeze Code2Wav
    for param in model.code2wav.parameters():
        param.requires_grad_(False)
    
    # ===== FREEZE THINKER HOÀN TOÀN =====
    print("Freezing Thinker (không train)...")
    for param in model.thinker.parameters():
        param.requires_grad_(False)
    model.thinker.eval()  # Set to eval mode
    print("✅ Thinker frozen\n")
    
    # ===== CHỈ TRAIN TALKER =====
    print("Enabling Talker for training...")
    model.talker.train()
    
    # Gradient checkpointing
    model.talker.model.gradient_checkpointing_enable()
    
    # Load dataset
    print("Loading dataset...")
    samples = load_dataset(DATASET_PATH, max_samples=MAX_SAMPLES)
    print(f"Loaded {len(samples)} samples")
    
    # Optimizer - CHỈ TRAIN TALKER
    talker_params = [p for p in model.talker.parameters() if p.requires_grad]
    
    print(f"Trainable parameters:")
    print(f"  - Talker: {sum(p.numel() for p in talker_params):,}")
    print(f"  - Thinker: 0 (frozen)")
    print()
    
    optimizer = torch.optim.AdamW(
        talker_params,
        lr=LEARNING_RATE_TALKER
    )
    
    # Training settings
    num_mtp_layers = NUM_CODE_GROUPS - 1
    codec_eos_id = model.config.talker_config.codec_eos_token_id
    
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Total samples: {len(samples)}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Steps per epoch: {STEPS_PER_EPOCH if STEPS_PER_EPOCH else 'Full dataset'}")
    print(f"Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"LR Talker: {LEARNING_RATE_TALKER}")
    print(f"Training mode: TALKER ONLY (Thinker frozen)")
    print(f"Save checkpoint every {SAVE_EVERY_N_STEPS} steps")
    print("="*60 + "\n")
    
    print("Starting training...")
    
    global_step = 0
    
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        num_steps = STEPS_PER_EPOCH if STEPS_PER_EPOCH else len(samples)
        
        # Shuffle samples each epoch
        import random
        random.shuffle(samples)
        
        for step, sample in enumerate(tqdm(samples[:num_steps], desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")):
            
            if step % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.zero_grad(set_to_none=True)
            
            # Prepare input using original messages
            batch, conversation = prepare_tts_conversation(sample['messages'], processor)
            batch = batch.to(execution_device).to(model.thinker.dtype)
            
            # Encode target audio
            target_codes = encode_audio_to_codes(
                sample['audio_path'], 
                feature_extractor, 
                mimi_model, 
                execution_device
            ).long()
            target_codes = align_codebook_dim(target_codes, model.code2wav.config.num_quantizers)
            
            # Create labels for Thinker (mask input, predict assistant response)
            input_ids = batch["input_ids"]
            thinker_labels = input_ids.clone()
            
            # Find assistant response position
            assistant_token_id = model.config.assistant_token_id
            im_start_token_id = model.config.im_start_token_id
            
            for idx in range(input_ids.shape[1] - 1):
                if input_ids[0, idx] == im_start_token_id:
                    if idx + 1 < input_ids.shape[1] and input_ids[0, idx + 1] == assistant_token_id:
                        response_start_idx = idx + 3
                        thinker_labels[:, :response_start_idx] = -100
                        break
            
            thinker_labels = thinker_labels.to(execution_device)
            
            # Forward Thinker (frozen - chỉ để lấy hidden states)
            with torch.no_grad():  # Không tính gradient cho Thinker
                thinker_outputs = model.thinker(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    output_hidden_states=True,
                    return_dict=True,
                )
            thinker_loss = torch.tensor(0.0, device=execution_device)  # Không train Thinker
            
            # Build Talker prefix
            talker_input_embed, talker_input_ids, trailing_text_hidden, tts_pad_embed = build_talker_prefix_tts(
                model=model,
                thinker_outputs=thinker_outputs,
                input_ids=batch["input_ids"],
                speaker_name=sample['speaker'],
                device=execution_device,
            )
            
            # Prepare Talker training (Layer 0)
            layer0_codes = target_codes[:, 0, :]
            num_codec_tokens = layer0_codes.shape[1]
            
            layer0_embeds = model.talker.get_input_embeddings()(layer0_codes.to(execution_device))
            predictor_embeds = model.talker.code_predictor.get_input_embeddings()
            
            # Sum all layer embeddings
            all_layer_embeds_sum = layer0_embeds.clone()
            for j in range(len(predictor_embeds)):
                layer_j_codes = target_codes[:, j + 1, :]
                emb = predictor_embeds[j](layer_j_codes.to(execution_device))
                all_layer_embeds_sum = all_layer_embeds_sum + emb
            
            # Build shifted inputs (teacher forcing)
            text_len = trailing_text_hidden.shape[1]
            codec_input_embeds_list = []
            
            for pos in range(num_codec_tokens):
                if pos == 0:
                    continue
                prev_pos = pos - 1
                text_hidden = trailing_text_hidden[:, prev_pos:prev_pos+1, :] if prev_pos < text_len else tts_pad_embed
                pos_embed = all_layer_embeds_sum[:, prev_pos:prev_pos+1, :] + text_hidden
                codec_input_embeds_list.append(pos_embed)
            
            # EOS input
            last_pos = num_codec_tokens - 1
            eos_text_hidden = trailing_text_hidden[:, last_pos:last_pos+1, :] if last_pos < text_len else tts_pad_embed
            eos_input_embed = all_layer_embeds_sum[:, last_pos:last_pos+1, :] + eos_text_hidden
            codec_input_embeds_list.append(eos_input_embed)
            
            # Concatenate
            if codec_input_embeds_list:
                codec_input_embeds = torch.cat(codec_input_embeds_list, dim=1).to(model.talker.dtype)
                full_inputs_embeds = torch.cat([talker_input_embed, codec_input_embeds], dim=1)
            else:
                full_inputs_embeds = talker_input_embed
            
            # Labels
            prefix_len = talker_input_embed.shape[1]
            labels_prefix = torch.full((1, prefix_len - 1), -100, dtype=torch.long, device=execution_device)
            labels_code = layer0_codes.to(execution_device)
            labels_eos = torch.tensor([[codec_eos_id]], device=execution_device)
            labels = torch.cat([labels_prefix, labels_code, labels_eos], dim=1)
            
            # Attention mask
            seq_len = full_inputs_embeds.shape[1]
            attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=execution_device)
            
            # Forward Talker - call the full forward pass
            # The Talker model handles its own logits computation internally
            talker_outputs = model.talker(
                inputs_embeds=full_inputs_embeds,
                attention_mask=attention_mask,
                trailing_text_hidden=trailing_text_hidden,
                tts_pad_embed=tts_pad_embed,
                output_hidden_states=True,
                return_dict=True,
            )
            
            # Manually compute loss (Talker's forward returns logits)
            if hasattr(talker_outputs, 'logits') and talker_outputs.logits is not None:
                talker_logits = talker_outputs.logits
            else:
                # If no logits attribute, try to get from last_hidden_state
                # Talker might have a different structure
                raise RuntimeError("Cannot get logits from Talker outputs")
            
            shift_logits = talker_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Compute cross entropy loss
            talker_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            
            # MTP Training (Layers 1-15)
            talker_hidden = talker_outputs.hidden_states[0][-1] if isinstance(talker_outputs.hidden_states, tuple) else talker_outputs.hidden_states[-1]
        
            # Extract hidden states for codec token positions
            codec_hidden_start = prefix_len - 1
            codec_hidden_end = prefix_len - 1 + num_codec_tokens
            codec_hidden = talker_hidden[:, codec_hidden_start:codec_hidden_end, :]
            
            mtp_total_loss = 0.0
            code_predictor = model.talker.code_predictor
            hidden_dim = codec_hidden.shape[2]
            
            layer0_embed_for_mtp = model.talker.get_input_embeddings()(layer0_codes.to(execution_device))
            hidden_flat = codec_hidden.reshape(-1, 1, hidden_dim)
            layer0_flat = layer0_embed_for_mtp.reshape(-1, 1, hidden_dim)
            
            for mtp_layer_idx in range(num_mtp_layers):
                embed_list = [hidden_flat, layer0_flat]
                
                for prev_layer in range(mtp_layer_idx):
                    prev_codes = target_codes[:, prev_layer + 1, :].to(execution_device)
                    prev_embed = predictor_embeds[prev_layer](prev_codes)
                    prev_embed_flat = prev_embed.reshape(-1, 1, hidden_dim)
                    embed_list.append(prev_embed_flat)
                
                mtp_inputs = torch.cat(embed_list, dim=1).to(model.talker.dtype)
                target_layer_codes = target_codes[:, mtp_layer_idx + 1, :].to(execution_device)
                target_labels = target_layer_codes.reshape(-1)
                
                mtp_outputs = code_predictor(
                    inputs_embeds=mtp_inputs,
                    generation_steps=mtp_layer_idx,
                    use_cache=False,
                )
                
                mtp_logits = mtp_outputs.logits[:, -1, :]
                mtp_layer_loss = F.cross_entropy(mtp_logits, target_labels)
                mtp_total_loss += mtp_layer_loss
            
            mtp_avg_loss = mtp_total_loss / num_mtp_layers
            
            # Total loss
            total_loss = (thinker_loss + talker_loss + 2.0 * mtp_avg_loss) / GRADIENT_ACCUMULATION_STEPS
            epoch_loss += total_loss.item() * GRADIENT_ACCUMULATION_STEPS
            
            total_loss.backward()
            
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                global_step += 1
            
            if step % 10 == 0:
                print(f"[Epoch {epoch+1}, Step {step}/{num_steps}] thinker={thinker_loss.item():.4f}, talker={talker_loss.item():.4f}, mtp={mtp_avg_loss.item():.4f}, total={total_loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}")
            
            # Save checkpoint periodically
            if global_step > 0 and global_step % SAVE_EVERY_N_STEPS == 0:
                checkpoint_path = OUTPUT_DIR / f"checkpoint_step_{global_step}"
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                
                # Only save Talker state dict
                torch.save(
                    model.talker.state_dict(),
                    checkpoint_path / "talker_state_dict.pt"
                )
                
                # Save processor for convenience
                processor.save_pretrained(checkpoint_path)
                
                print(f"\n💾 Saved checkpoint at step {global_step}")
                print(f"   - Talker weights: {checkpoint_path / 'talker_state_dict.pt'}\n")
        
        avg_epoch_loss = epoch_loss / num_steps
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} completed")
        print(f"Average loss: {avg_epoch_loss:.4f}")
        print(f"Total steps: {global_step}")
        print(f"{'='*60}\n")
        
        # Save checkpoint at end of epoch
        checkpoint_path = OUTPUT_DIR / f"checkpoint_epoch_{epoch+1}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Only save Talker state dict
        torch.save(
            model.talker.state_dict(),
            checkpoint_path / "talker_state_dict.pt"
        )
        
        # Save processor
        processor.save_pretrained(checkpoint_path)
        
        print(f"✅ Saved epoch {epoch+1} checkpoint")
        print(f"   - Talker weights: {checkpoint_path / 'talker_state_dict.pt'}\n")
    
    print("\nTraining completed!")

if __name__ == "__main__":
    train_tts()
