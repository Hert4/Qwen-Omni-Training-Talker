"""
Modified training script for Text-to-Speech fine-tuning on Qwen3-Omni
With TRUE BATCH SIZE = 16 (not gradient accumulation)
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
from torch.utils.data import Dataset, DataLoader
import librosa


# Configuration
DATASET_PATH = Path("datasets/misa-tts-LPAnh/processed/train.jsonl")
MODEL_PATH = "models/Qwen3-Omni-30B-A3B-Instruct"
MIMI_REPO_ID = "models/mimi"
OUTPUT_DIR = Path("models")
SPEAKER = "Ethan"
NUM_CODE_GROUPS = 16
USE_AUDIO_IN_VIDEO = False

        
# ===== TRAINING HYPERPARAMETERS =====
BATCH_SIZE = 8             # TRUE batch size
MAX_SAMPLES = None
NUM_EPOCHS = 1
STEPS_PER_EPOCH = None
GRADIENT_ACCUMULATION_STEPS = 1  # No accumulation needed
LEARNING_RATE_TALKER = 2e-5
SAVE_EVERY_N_STEPS = 500
LORA_R = 16
LORA_ALPHA = 32


class TTSDataset(Dataset):
    """Dataset for TTS training"""
    def __init__(self, jsonl_path: Path, max_samples: int = None):
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                data = json.loads(line)
                
                if 'messages' in data and 'audios' in data:
                    self.samples.append({
                        'messages': data['messages'],
                        'audio_path': Path(data['audios'][0]),
                        'speaker': data.get('speaker', SPEAKER)
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def encode_audio_to_codes(audio_path: Path, feature_extractor, mimi_model, device):
    """Convert audio file to codec codes"""
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


def collate_fn(batch_samples, processor, feature_extractor, mimi_model, device, model):
    """Collate multiple samples into a batch - returns CPU tensors for DataLoader"""
    
    # Prepare text inputs
    formatted_texts = []
    speakers = []
    audio_paths = []
    conversations = []
    
    for sample in batch_samples:
        formatted_text = processor.apply_chat_template(
            sample['messages'], 
            add_generation_prompt=False, 
            tokenize=False
        )
        formatted_texts.append(formatted_text)
        speakers.append(sample['speaker'])
        audio_paths.append(sample['audio_path'])
        conversations.append(sample['messages'])
    
    # Batch process texts (returns CPU tensors)
    batch = processor(
        text=formatted_texts,
        audio=None,
        images=None,
        videos=None,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
    )
    
    # Encode all audios and move back to CPU
    target_codes_list = []
    for audio_path in audio_paths:
        codes = encode_audio_to_codes(
            audio_path, 
            feature_extractor, 
            mimi_model, 
            device
        )
        codes = align_codebook_dim(codes, model.code2wav.config.num_quantizers)
        # Move to CPU for DataLoader
        target_codes_list.append(codes.cpu())
    
    # Find max audio length
    max_audio_len = max(codes.shape[2] for codes in target_codes_list)
    
    # Pad audio codes to same length (on CPU)
    padded_codes = []
    audio_lengths = []
    for codes in target_codes_list:
        audio_lengths.append(codes.shape[2])
        if codes.shape[2] < max_audio_len:
            padding = torch.zeros(
                (codes.shape[0], codes.shape[1], max_audio_len - codes.shape[2]),
                dtype=codes.dtype
            )
            codes = torch.cat([codes, padding], dim=2)
        padded_codes.append(codes)
    
    # Stack into batch (on CPU)
    target_codes = torch.cat(padded_codes, dim=0)
    
    return {
        'batch': batch,
        'target_codes': target_codes,
        'speakers': speakers,
        'conversations': conversations,
        'audio_lengths': audio_lengths
    }


def build_talker_prefix_tts(model, thinker_outputs, input_ids, speaker_name, device, batch_idx=0):
    """Build Talker prefix for TTS - works with batched inputs"""
    config = model.config
    
    # Extract for specific batch index
    thinker_embed = thinker_outputs.hidden_states[0][batch_idx:batch_idx+1].to(device)
    accept_layer = config.talker_config.accept_hidden_layer
    thinker_hidden = thinker_outputs.hidden_states[accept_layer][batch_idx:batch_idx+1].to(device)
    sample_input_ids = input_ids[batch_idx:batch_idx+1]
    
    im_start_positions = torch.nonzero(sample_input_ids[0] == config.im_start_token_id).view(-1)
    im_start_indexes = torch.cat(
        (im_start_positions, torch.tensor([sample_input_ids.shape[1]], device=device)),
        dim=0,
    )
    
    # Multimodal mask (all False for text-only TTS)
    multimodal_mask = torch.zeros_like(sample_input_ids, dtype=torch.bool).to(device)
    
    # Special tokens for Talker
    talker_special_tokens = torch.tensor(
        [[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]],
        device=device,
        dtype=sample_input_ids.dtype,
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
        role_token = sample_input_ids[0][im_start_index + 1]
        
        if role_token == config.user_token_id:
            user_part = model._get_talker_user_parts(
                im_start_index,
                segment_end_index,
                multimodal_mask,
                thinker_hidden,
                thinker_embed,
            )
            talker_input_embeds.append(user_part)
            talker_input_ids.append(sample_input_ids[:, im_start_index:segment_end_index])
            
        elif role_token == config.assistant_token_id and i == len(im_start_indexes) - 2:
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


def compute_sample_loss(model, thinker_outputs, input_ids, target_codes, speaker, 
                       device, batch_idx, audio_length):
    """Compute loss for a single sample in the batch"""
    
    # Build Talker prefix
    talker_input_embed, talker_input_ids, trailing_text_hidden, tts_pad_embed = build_talker_prefix_tts(
        model=model,
        thinker_outputs=thinker_outputs,
        input_ids=input_ids,
        speaker_name=speaker,
        device=device,
        batch_idx=batch_idx
    )
    
    # Extract this sample's codes
    sample_codes = target_codes[batch_idx:batch_idx+1, :, :audio_length]
    
    # Prepare Talker training (Layer 0)
    layer0_codes = sample_codes[:, 0, :]
    num_codec_tokens = layer0_codes.shape[1]
    
    layer0_embeds = model.talker.get_input_embeddings()(layer0_codes.to(device))
    predictor_embeds = model.talker.code_predictor.get_input_embeddings()
    
    # Sum all layer embeddings
    all_layer_embeds_sum = layer0_embeds.clone()
    for j in range(len(predictor_embeds)):
        layer_j_codes = sample_codes[:, j + 1, :]
        emb = predictor_embeds[j](layer_j_codes.to(device))
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
    labels_prefix = torch.full((1, prefix_len - 1), -100, dtype=torch.long, device=device)
    labels_code = layer0_codes.to(device)
    codec_eos_id = model.config.talker_config.codec_eos_token_id
    labels_eos = torch.tensor([[codec_eos_id]], device=device)
    labels = torch.cat([labels_prefix, labels_code, labels_eos], dim=1)
    
    # Attention mask
    seq_len = full_inputs_embeds.shape[1]
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    
    # Forward Talker
    talker_outputs = model.talker(
        inputs_embeds=full_inputs_embeds,
        attention_mask=attention_mask,
        trailing_text_hidden=trailing_text_hidden,
        tts_pad_embed=tts_pad_embed,
        output_hidden_states=True,
        return_dict=True,
    )
    
    # Compute Talker loss
    if hasattr(talker_outputs, 'logits') and talker_outputs.logits is not None:
        talker_logits = talker_outputs.logits
    else:
        raise RuntimeError("Cannot get logits from Talker outputs")
    
    shift_logits = talker_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    talker_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100
    )
    
    # MTP Training (Layers 1-15)
    talker_hidden = talker_outputs.hidden_states[0][-1] if isinstance(talker_outputs.hidden_states, tuple) else talker_outputs.hidden_states[-1]
    
    codec_hidden_start = prefix_len - 1
    codec_hidden_end = prefix_len - 1 + num_codec_tokens
    codec_hidden = talker_hidden[:, codec_hidden_start:codec_hidden_end, :]
    
    mtp_total_loss = 0.0
    code_predictor = model.talker.code_predictor
    hidden_dim = codec_hidden.shape[2]
    num_mtp_layers = NUM_CODE_GROUPS - 1
    
    layer0_embed_for_mtp = model.talker.get_input_embeddings()(layer0_codes.to(device))
    hidden_flat = codec_hidden.reshape(-1, 1, hidden_dim)
    layer0_flat = layer0_embed_for_mtp.reshape(-1, 1, hidden_dim)
    
    for mtp_layer_idx in range(num_mtp_layers):
        embed_list = [hidden_flat, layer0_flat]
        
        for prev_layer in range(mtp_layer_idx):
            prev_codes = sample_codes[:, prev_layer + 1, :].to(device)
            prev_embed = predictor_embeds[prev_layer](prev_codes)
            prev_embed_flat = prev_embed.reshape(-1, 1, hidden_dim)
            embed_list.append(prev_embed_flat)
        
        mtp_inputs = torch.cat(embed_list, dim=1).to(model.talker.dtype)
        target_layer_codes = sample_codes[:, mtp_layer_idx + 1, :].to(device)
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
    
    return talker_loss, mtp_avg_loss


def train_tts():
    """Main training function for TTS fine-tuning with TRUE batch size"""
    
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
    
    # Freeze Thinker
    print("Freezing Thinker...")
    for param in model.thinker.parameters():
        param.requires_grad_(False)
    model.thinker.eval()
    print("✅ Thinker frozen\n")
    
    # Enable Talker for training
    print("Enabling Talker for training...")
    model.talker.train()
    model.talker.model.gradient_checkpointing_enable()
    
    # Load dataset
    print("Loading dataset...")
    dataset = TTSDataset(DATASET_PATH, max_samples=MAX_SAMPLES)
    print(f"Loaded {len(dataset)} samples")
    
    # Create DataLoader
    def collate_wrapper(batch):
        return collate_fn(batch, processor, feature_extractor, mimi_model, execution_device, model)
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,  # Set to False since collate_fn uses GPU
        collate_fn=collate_wrapper
    )
    
    # Optimizer
    talker_params = [p for p in model.talker.parameters() if p.requires_grad]
    
    print(f"Trainable parameters:")
    print(f"  - Talker: {sum(p.numel() for p in talker_params):,}")
    print()
    
    optimizer = torch.optim.AdamW(talker_params, lr=LEARNING_RATE_TALKER)
    
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Total samples: {len(dataset)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Steps per epoch: {len(dataloader)}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"LR Talker: {LEARNING_RATE_TALKER}")
    print(f"Training mode: TALKER ONLY (Thinker frozen)")
    print(f"Save checkpoint every {SAVE_EVERY_N_STEPS} steps")
    print("="*60 + "\n")
    
    print("Starting training...")
    
    global_step = 0
    
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")):
            
            if batch_idx % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.zero_grad(set_to_none=True)
            
            batch = batch_data['batch']
            target_codes = batch_data['target_codes']
            speakers = batch_data['speakers']
            audio_lengths = batch_data['audio_lengths']
            
            batch = {k: v.to(execution_device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            target_codes = target_codes.to(execution_device)
            
            # Forward Thinker (frozen)
            with torch.no_grad():
                thinker_outputs = model.thinker(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    output_hidden_states=True,
                    return_dict=True,
                )
            
            # Process each sample in batch
            batch_size_actual = batch["input_ids"].shape[0]
            batch_talker_loss = 0.0
            batch_mtp_loss = 0.0
            
            for i in range(batch_size_actual):
                try:
                    talker_loss, mtp_loss = compute_sample_loss(
                        model=model,
                        thinker_outputs=thinker_outputs,
                        input_ids=batch["input_ids"],
                        target_codes=target_codes,
                        speaker=speakers[i],
                        device=execution_device,
                        batch_idx=i,
                        audio_length=audio_lengths[i]
                    )
                    
                    batch_talker_loss += talker_loss
                    batch_mtp_loss += mtp_loss
                    
                except Exception as e:
                    print(f"\n⚠️ Error processing sample {i} in batch: {e}")
                    continue
            
            # Average over successful samples
            avg_talker_loss = batch_talker_loss / batch_size_actual
            avg_mtp_loss = batch_mtp_loss / batch_size_actual
            
            total_loss = (avg_talker_loss + 2.0 * avg_mtp_loss) / GRADIENT_ACCUMULATION_STEPS
            epoch_loss += total_loss.item() * GRADIENT_ACCUMULATION_STEPS
            
            total_loss.backward()
            
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                global_step += 1
            
            if batch_idx % 10 == 0:
                print(f"[Epoch {epoch+1}, Step {batch_idx}/{len(dataloader)}] "
                      f"talker={avg_talker_loss.item():.4f}, "
                      f"mtp={avg_mtp_loss.item():.4f}, "
                      f"total={total_loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}")
            
            # Save checkpoint periodically
            if global_step > 0 and global_step % SAVE_EVERY_N_STEPS == 0:
                checkpoint_path = OUTPUT_DIR / f"checkpoint_step_{global_step}"
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                
                torch.save(
                    model.talker.state_dict(),
                    checkpoint_path / "talker_state_dict.pt"
                )
                processor.save_pretrained(checkpoint_path)
                
                print(f"\n💾 Saved checkpoint at step {global_step}")
                print(f"   - Talker weights: {checkpoint_path / 'talker_state_dict.pt'}\n")
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} completed")
        print(f"Average loss: {avg_epoch_loss:.4f}")
        print(f"Total steps: {global_step}")
        print(f"{'='*60}\n")
        
        # Save checkpoint at end of epoch
        checkpoint_path = OUTPUT_DIR / f"checkpoint_epoch_{epoch+1}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        torch.save(
            model.talker.state_dict(),
            checkpoint_path / "talker_state_dict.pt"
        )
        processor.save_pretrained(checkpoint_path)
        
        print(f"✅ Saved epoch {epoch+1} checkpoint")
        print(f"   - Talker weights: {checkpoint_path / 'talker_state_dict.pt'}\n")
    
    print("\nTraining completed!")


if __name__ == "__main__":
    train_tts()
