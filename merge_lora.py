
"""
Script để merge LoRA adapter và Talker weights vào base model Qwen3-Omni
"""

import torch
from pathlib import Path
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from peft import PeftModel

# ===== CẤU HÌNH =====
BASE_MODEL_PATH = "models/Qwen3-Omni-30B-A3B-Instruct"
CHECKPOINT_PATH = "output/checkpoints/checkpoint_epoch_1"  # Thay đổi theo checkpoint bạn muốn
OUTPUT_PATH = "models/Qwen3-Omni-30B-A3B-Instruct-FT"  # Nơi lưu model đã merge

def merge_lora_and_talker():
    """Merge LoRA adapter và Talker weights vào base model"""
    
    print("="*60)
    print("MERGE LORA + TALKER VÀO BASE MODEL")
    print("="*60)
    print(f"Base model: {BASE_MODEL_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print("="*60 + "\n")
    
    # ===== BƯỚC 1: Load base model =====
    print("📦 Loading base model...")
    base_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,  # Hoặc "auto"
        device_map="auto",
    )
    print("✅ Base model loaded\n")
    
    # ===== BƯỚC 2: Load và merge LoRA adapter cho Thinker =====
    lora_path = Path(CHECKPOINT_PATH) / "thinker_lora"
    
    if lora_path.exists():
        print(f"🔧 Loading LoRA adapter from {lora_path}...")
        base_model.thinker = PeftModel.from_pretrained(
            base_model.thinker,
            str(lora_path),
            is_trainable=False
        )
        print("✅ LoRA adapter loaded\n")
        
        print("🔄 Merging LoRA into base model...")
        base_model.thinker = base_model.thinker.merge_and_unload()
        print("✅ LoRA merged successfully\n")
    else:
        print(f"⚠️  LoRA adapter not found at {lora_path}, skipping...\n")
    
    # ===== BƯỚC 3: Load Talker weights =====
    talker_weights_path = Path(CHECKPOINT_PATH) / "talker_state_dict.pt"
    
    if talker_weights_path.exists():
        print(f"🔧 Loading Talker weights from {talker_weights_path}...")
        talker_state_dict = torch.load(talker_weights_path, map_location="cpu")
        base_model.talker.load_state_dict(talker_state_dict)
        print("✅ Talker weights loaded\n")
    else:
        print(f"⚠️  Talker weights not found at {talker_weights_path}, skipping...\n")
    
    # ===== BƯỚC 4: Load processor =====
    print("📝 Loading processor...")
    processor = Qwen3OmniMoeProcessor.from_pretrained(CHECKPOINT_PATH)
    print("✅ Processor loaded\n")
    
    # ===== BƯỚC 5: Save merged model =====
    output_path = Path(OUTPUT_PATH)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Saving merged model to {output_path}...")
    base_model.save_pretrained(
        output_path,
        safe_serialization=True,  # Sử dụng safetensors
        max_shard_size="5GB"  # Chia nhỏ file nếu quá lớn
    )
    processor.save_pretrained(output_path)
    print("✅ Model saved successfully\n")
    
    print("="*60)
    print("HOÀN TẤT!")
    print("="*60)
    print(f"Model đã merge được lưu tại: {output_path}")
    print("\nBạn có thể sử dụng model này như sau:")
    print(f"""
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    "{output_path}",
    device_map="auto"
)
processor = Qwen3OmniMoeProcessor.from_pretrained("{output_path}")
    """)

def verify_model():
    """Verify merged model có thể load được không"""
    
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    output_path = Path(OUTPUT_PATH)
    
    if not output_path.exists():
        print(f"❌ Model chưa được merge tại {output_path}")
        return
    
    try:
        print("🔍 Verifying merged model...")
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            output_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",  # Load vào CPU để test
        )
        processor = Qwen3OmniMoeProcessor.from_pretrained(output_path)
        
        print("✅ Model verification successful!")
        print(f"   - Thinker parameters: {sum(p.numel() for p in model.thinker.parameters()):,}")
        print(f"   - Talker parameters: {sum(p.numel() for p in model.talker.parameters()):,}")
        print(f"   - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        del model, processor
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Merge LoRA và Talker vào base model")
    parser.add_argument("--base_model", type=str, default=BASE_MODEL_PATH, help="Path to base model")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH, help="Path to checkpoint")
    parser.add_argument("--output", type=str, default=OUTPUT_PATH, help="Path to output merged model")
    parser.add_argument("--verify", action="store_true", help="Verify model after merge")
    
    args = parser.parse_args()
    
    # Update paths
    BASE_MODEL_PATH = args.base_model
    CHECKPOINT_PATH = args.checkpoint
    OUTPUT_PATH = args.output
    
    # Merge
    merge_lora_and_talker()
    
    # Verify if requested
    if args.verify:
        verify_model()
