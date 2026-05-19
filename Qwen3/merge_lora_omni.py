"""
Script để merge LoRA adapters (Thinker, Talker, MTP) vào base model Qwen3-Omni
Phù hợp với training script sử dụng LoRA cho cả 3 modules
"""

import torch
from pathlib import Path
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from peft import PeftModel
import shutil
import json

# ===== CẤU HÌNH =====
BASE_MODEL_PATH = "models/Qwen3-Omni-30B-A3B-Instruct"
CHECKPOINT_PATH = "models/checkpoint_epoch_1"  # Thay đổi theo checkpoint bạn muốn
OUTPUT_PATH = "models/Qwen3-Omni-30B-A3B-Instruct-ft-ep1"  # Nơi lưu model đã merge


def check_checkpoint_structure(checkpoint_path):
    """Kiểm tra cấu trúc checkpoint"""
    checkpoint_path = Path(checkpoint_path)
    
    print("\n🔍 Checking checkpoint structure...")
    print(f"Checkpoint path: {checkpoint_path}")
    
    # Check LoRA adapters
    thinker_lora_path = checkpoint_path / "thinker_lora"
    talker_lora_path = checkpoint_path / "talker_lora"
    mtp_lora_path = checkpoint_path / "mtp_lora"
    
    has_thinker_lora = thinker_lora_path.exists()
    has_talker_lora = talker_lora_path.exists()
    has_mtp_lora = mtp_lora_path.exists()
    
    print(f"  {'✅' if has_thinker_lora else '❌'} Thinker LoRA: {thinker_lora_path}")
    if has_thinker_lora:
        lora_files = list(thinker_lora_path.glob("*"))
        print(f"     Files: {[f.name for f in lora_files[:5]]}")
    
    print(f"  {'✅' if has_talker_lora else '❌'} Talker LoRA: {talker_lora_path}")
    if has_talker_lora:
        lora_files = list(talker_lora_path.glob("*"))
        print(f"     Files: {[f.name for f in lora_files[:5]]}")
    
    print(f"  {'✅' if has_mtp_lora else '❌'} MTP LoRA: {mtp_lora_path}")
    if has_mtp_lora:
        lora_files = list(mtp_lora_path.glob("*"))
        print(f"     Files: {[f.name for f in lora_files[:5]]}")
    
    # Check processor
    has_processor = (checkpoint_path / "tokenizer_config.json").exists()
    print(f"  {'✅' if has_processor else '❌'} Processor config")
    
    # Check training config
    training_config_path = checkpoint_path / "training_config.json"
    has_training_config = training_config_path.exists()
    print(f"  {'✅' if has_training_config else '❌'} Training config")
    
    if has_training_config:
        with open(training_config_path, 'r') as f:
            training_config = json.load(f)
        print(f"     Training config: {training_config}")
    else:
        training_config = None
    
    return {
        'has_thinker_lora': has_thinker_lora,
        'has_talker_lora': has_talker_lora,
        'has_mtp_lora': has_mtp_lora,
        'has_processor': has_processor,
        'training_config': training_config
    }


def merge_lora_adapters(base_model_path, checkpoint_path, output_path, verify=False):
    """Merge tất cả LoRA adapters vào base model"""
    
    base_model_path = Path(base_model_path)
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    
    print("\n" + "="*60)
    print("MERGE LORA ADAPTERS (THINKER + TALKER + MTP) VÀO BASE MODEL")
    print("="*60)
    print(f"📁 Base model: {base_model_path}")
    print(f"📁 Checkpoint: {checkpoint_path}")
    print(f"📁 Output: {output_path}")
    print("="*60)
    
    # Kiểm tra checkpoint structure
    checkpoint_info = check_checkpoint_structure(checkpoint_path)
    
    if not any([checkpoint_info['has_thinker_lora'], 
                checkpoint_info['has_talker_lora'], 
                checkpoint_info['has_mtp_lora']]):
        print("\n❌ ERROR: Checkpoint không có bất kỳ LoRA adapter nào!")
        return False
    
    # ===== BƯỚC 1: Load base model =====
    print("\n" + "="*60)
    print("BƯỚC 1: Loading base model")
    print("="*60)
    
    try:
        base_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print("✅ Base model loaded successfully")
        
        # Print model info
        total_params = sum(p.numel() for p in base_model.parameters())
        thinker_params = sum(p.numel() for p in base_model.thinker.parameters())
        talker_params = sum(p.numel() for p in base_model.talker.parameters())
        
        print(f"\n📊 Base model info:")
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Thinker parameters: {thinker_params:,}")
        print(f"   - Talker parameters: {talker_params:,}")
        
    except Exception as e:
        print(f"❌ Failed to load base model: {e}")
        return False
    
    # ===== BƯỚC 2: Merge Thinker LoRA =====
    if checkpoint_info['has_thinker_lora']:
        print("\n" + "="*60)
        print("BƯỚC 2: Merging Thinker LoRA adapter")
        print("="*60)
        
        lora_path = checkpoint_path / "thinker_lora"
        
        try:
            print(f"🔧 Loading Thinker LoRA adapter từ {lora_path}...")
            
            # Load LoRA adapter lên Thinker
            thinker_with_lora = PeftModel.from_pretrained(
                base_model.thinker,
                str(lora_path),
                is_trainable=False
            )
            
            print("✅ Thinker LoRA adapter loaded")
            
            # Print trainable parameters info
            print("\n📊 Thinker LoRA adapter info:")
            thinker_with_lora.print_trainable_parameters()
            
            # Merge LoRA vào base model
            print("\n🔄 Merging Thinker LoRA weights into base model...")
            merged_thinker = thinker_with_lora.merge_and_unload()
            
            # Replace thinker với merged version
            base_model.thinker = merged_thinker
            
            print("✅ Thinker LoRA merged successfully!")
            
        except Exception as e:
            print(f"❌ Failed to merge Thinker LoRA: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("\n⚠️  Bỏ qua BƯỚC 2: Không có Thinker LoRA adapter")
    
    # ===== BƯỚC 3: Merge Talker LoRA =====
    if checkpoint_info['has_talker_lora']:
        print("\n" + "="*60)
        print("BƯỚC 3: Merging Talker LoRA adapter")
        print("="*60)
        
        lora_path = checkpoint_path / "talker_lora"
        
        try:
            print(f"🔧 Loading Talker LoRA adapter từ {lora_path}...")
            
            # Load LoRA adapter lên talker.model
            talker_model_with_lora = PeftModel.from_pretrained(
                base_model.talker.model,
                str(lora_path),
                is_trainable=False
            )
            
            print("✅ Talker LoRA adapter loaded")
            
            # Print trainable parameters info
            print("\n📊 Talker LoRA adapter info:")
            talker_model_with_lora.print_trainable_parameters()
            
            # Merge LoRA vào base model
            print("\n🔄 Merging Talker LoRA weights into base model...")
            merged_talker_model = talker_model_with_lora.merge_and_unload()
            
            # Replace talker.model với merged version
            base_model.talker.model = merged_talker_model
            
            print("✅ Talker LoRA merged successfully!")
            
        except Exception as e:
            print(f"❌ Failed to merge Talker LoRA: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("\n⚠️  Bỏ qua BƯỚC 3: Không có Talker LoRA adapter")
    
    # ===== BƯỚC 4: Merge MTP LoRA =====
    if checkpoint_info['has_mtp_lora']:
        print("\n" + "="*60)
        print("BƯỚC 4: Merging MTP (code_predictor) LoRA adapter")
        print("="*60)
        
        lora_path = checkpoint_path / "mtp_lora"
        
        try:
            print(f"🔧 Loading MTP LoRA adapter từ {lora_path}...")
            
            # Load LoRA adapter lên talker.code_predictor.model
            mtp_model_with_lora = PeftModel.from_pretrained(
                base_model.talker.code_predictor.model,
                str(lora_path),
                is_trainable=False
            )
            
            print("✅ MTP LoRA adapter loaded")
            
            # Print trainable parameters info
            print("\n📊 MTP LoRA adapter info:")
            mtp_model_with_lora.print_trainable_parameters()
            
            # Merge LoRA vào base model
            print("\n🔄 Merging MTP LoRA weights into base model...")
            merged_mtp_model = mtp_model_with_lora.merge_and_unload()
            
            # Replace talker.code_predictor.model với merged version
            base_model.talker.code_predictor.model = merged_mtp_model
            
            print("✅ MTP LoRA merged successfully!")
            
        except Exception as e:
            print(f"❌ Failed to merge MTP LoRA: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("\n⚠️  Bỏ qua BƯỚC 4: Không có MTP LoRA adapter")
    
    # ===== BƯỚC 5: Load processor =====
    print("\n" + "="*60)
    print("BƯỚC 5: Loading processor")
    print("="*60)
    
    try:
        if checkpoint_info['has_processor']:
            processor = Qwen3OmniMoeProcessor.from_pretrained(checkpoint_path)
            print("✅ Processor loaded từ checkpoint")
        else:
            processor = Qwen3OmniMoeProcessor.from_pretrained(base_model_path)
            print("✅ Processor loaded từ base model")
    except Exception as e:
        print(f"❌ Failed to load processor: {e}")
        return False
    
    # ===== BƯỚC 6: Save merged model =====
    print("\n" + "="*60)
    print("BƯỚC 6: Saving merged model")
    print("="*60)
    
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Saving merged model to {output_path}...")
        print("   This may take a few minutes...")
        
        base_model.save_pretrained(
            output_path,
            safe_serialization=True,
            max_shard_size="5GB"
        )
        
        processor.save_pretrained(output_path)
        
        # Copy config files if needed
        config_files = ["generation_config.json", "special_tokens_map.json"]
        for config_file in config_files:
            src = checkpoint_path / config_file
            if not src.exists():
                src = base_model_path / config_file
            
            if src.exists():
                dst = output_path / config_file
                if not dst.exists():
                    shutil.copy2(src, dst)
                    print(f"   📄 Copied {config_file}")
        
        # Save merge info
        merge_info = {
            'base_model': str(base_model_path),
            'checkpoint': str(checkpoint_path),
            'merged_components': {
                'thinker_lora': checkpoint_info['has_thinker_lora'],
                'talker_lora': checkpoint_info['has_talker_lora'],
                'mtp_lora': checkpoint_info['has_mtp_lora'],
            },
            'training_config': checkpoint_info['training_config']
        }
        
        with open(output_path / "merge_info.json", 'w') as f:
            json.dump(merge_info, f, indent=2)
        
        print("✅ Model saved successfully")
        
    except Exception as e:
        print(f"❌ Failed to save model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ===== SUMMARY =====
    print("\n" + "="*60)
    print("✅ HOÀN TẤT!")
    print("="*60)
    print(f"📁 Merged model saved at: {output_path}")
    
    merged_components = []
    if checkpoint_info['has_thinker_lora']:
        merged_components.append("Thinker LoRA")
    if checkpoint_info['has_talker_lora']:
        merged_components.append("Talker LoRA")
    if checkpoint_info['has_mtp_lora']:
        merged_components.append("MTP LoRA")
    
    print(f"🔧 Merged components: {', '.join(merged_components)}")
    
    print("\n📝 Cách sử dụng model:")
    print(f"""
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    "{output_path}",
    device_map="auto",
    torch_dtype="auto"
)
processor = Qwen3OmniMoeProcessor.from_pretrained("{output_path}")
""")
    
    # Verify if requested
    if verify:
        verify_merged_model(output_path)
    
    return True


def verify_merged_model(output_path):
    """Verify merged model có thể load được không"""
    
    print("\n" + "="*60)
    print("VERIFICATION - Kiểm tra model đã merge")
    print("="*60)
    
    output_path = Path(output_path)
    
    if not output_path.exists():
        print(f"❌ Model chưa tồn tại tại {output_path}")
        return False
    
    try:
        print("🔍 Loading merged model for verification...")
        
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            output_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        
        processor = Qwen3OmniMoeProcessor.from_pretrained(output_path)
        
        print("✅ Model loaded successfully!")
        
        # Print statistics
        total_params = sum(p.numel() for p in model.parameters())
        thinker_params = sum(p.numel() for p in model.thinker.parameters())
        talker_params = sum(p.numel() for p in model.talker.parameters())
        
        print(f"\n📊 Merged model statistics:")
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Thinker parameters: {thinker_params:,}")
        print(f"   - Talker parameters: {talker_params:,}")
        print(f"   - Vocab size: {len(processor.tokenizer)}")
        
        # Check if LoRA modules still exist (shouldn't after merge)
        lora_modules_found = []
        for name, module in model.named_modules():
            if 'lora' in name.lower():
                lora_modules_found.append(name)
        
        if lora_modules_found:
            print(f"⚠️  WARNING: Model vẫn chứa {len(lora_modules_found)} LoRA modules (có thể chưa merge đúng)")
            for name in lora_modules_found[:5]:
                print(f"     - {name}")
            if len(lora_modules_found) > 5:
                print(f"     ... và {len(lora_modules_found) - 5} modules khác")
        else:
            print("✅ No LoRA modules found (đã merge thành công)")
        
        # Check merge info
        merge_info_path = output_path / "merge_info.json"
        if merge_info_path.exists():
            with open(merge_info_path, 'r') as f:
                merge_info = json.load(f)
            print(f"\n📋 Merge info:")
            print(f"   - Base model: {merge_info.get('base_model', 'N/A')}")
            print(f"   - Checkpoint: {merge_info.get('checkpoint', 'N/A')}")
            print(f"   - Merged components: {merge_info.get('merged_components', {})}")
        
        # Cleanup
        del model, processor
        torch.cuda.empty_cache()
        
        print("\n✅ Verification completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters (Thinker, Talker, MTP) vào base model"
    )
    parser.add_argument(
        "--base_model", 
        type=str, 
        default=BASE_MODEL_PATH, 
        help="Path to base model"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=CHECKPOINT_PATH, 
        help="Path to checkpoint (chứa thinker_lora/, talker_lora/, mtp_lora/)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=OUTPUT_PATH, 
        help="Path to output merged model"
    )
    parser.add_argument(
        "--verify", 
        action="store_true", 
        help="Verify model after merge"
    )
    parser.add_argument(
        "--no-verify",
        dest='verify',
        action="store_false",
        help="Skip verification"
    )
    
    parser.set_defaults(verify=True)
    
    args = parser.parse_args()
    
    # Run merge
    success = merge_lora_adapters(
        base_model_path=args.base_model,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        verify=args.verify
    )
    
    if success:
        print("\n🎉 Merge completed successfully!")
    else:
        print("\n❌ Merge failed!")
        exit(1)
