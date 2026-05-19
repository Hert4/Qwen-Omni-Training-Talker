import os
os.environ['VLLM_USE_V1'] = '1'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
import warnings
import numpy as np

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeProcessor

### AFTER TRAINING MODEL AND UPLOAD TO HF
MODEL_PATH = "beyoru/Qwen3-Omni-30B-A3B-Instruct-checkpoint-500"
USE_TRANSFORMERS = True  # HF FOR DEFAULT GENERATION
TRANSFORMERS_USE_FLASH_ATTN2 = False
USE_AUDIO_IN_VIDEO = False
RETURN_AUDIO = True

def _load_model_processor():
    if USE_TRANSFORMERS:
        from transformers import Qwen3OmniMoeForConditionalGeneration
        
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            MODEL_PATH, 
            # device_map="cuda", 
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )        
        target_device = next(model.parameters()).device
        if hasattr(model, 'talker') and model.talker is not None:
            model.talker = model.talker.to(target_device)
    else:
        from vllm import LLM
        model = LLM(
            model=MODEL_PATH, 
            trust_remote_code=True, 
            gpu_memory_utilization=0.95,
            tensor_parallel_size=torch.cuda.device_count(),
            limit_mm_per_prompt={'image': 1, 'video': 3, 'audio': 3},
            max_num_seqs=1,
            max_model_len=32768,
            seed=1234,
        )

    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
    return model, processor

def run_model(model, processor, messages, return_audio, use_audio_in_video):
    if USE_TRANSFORMERS:
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
        inputs = processor(text=text, audio=audios, images=images, videos=videos, 
                          return_tensors="pt", padding=True, use_audio_in_video=use_audio_in_video)
        inputs = inputs.to(model.device).to(model.dtype)
        text_ids, audio = model.generate(
            **inputs, 
            temperature=1e-2,
            thinker_return_dict_in_generate=True,
            thinker_max_new_tokens=8192, 
            thinker_do_sample=False,
            use_audio_in_video=use_audio_in_video,
            return_audio=return_audio
        )
        response = processor.batch_decode(
            text_ids.sequences[:, inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        if audio is not None:
            audio = np.array(audio.reshape(-1).detach().cpu().numpy() * 32767).astype(np.int16)
        return response, audio
    else:
        from vllm import SamplingParams
        sampling_params = SamplingParams(temperature=1e-2, top_p=0.1, top_k=1, max_tokens=8192)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
        inputs = {
            'prompt': text, 
            'multi_modal_data': {}, 
            "mm_processor_kwargs": {"use_audio_in_video": use_audio_in_video}
        }
        if images is not None: inputs['multi_modal_data']['image'] = images
        if videos is not None: inputs['multi_modal_data']['video'] = videos
        if audios is not None: inputs['multi_modal_data']['audio'] = audios
        outputs = model.generate(inputs, sampling_params=sampling_params)
        response = outputs[0].outputs[0].text
        return response, None


import scipy.io.wavfile as wavfile
from datetime import datetime

def conversation_loop(model, processor):
    """Vòng lặp conversation cho phép chat liên tục"""
    
    # System message
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are voicebot AI assistant, note that for each conversation, make sure in conversation you have to give emotional in your text (laugh, sing, sad, happy, etc,...). Only response in Vietnamese only, note that only use speaking language only. DO NOT explain or describe the context"}
            ]
        }
    ]
    
    conversation_count = 0
    
    print("\n" + "="*60)
    print("🤖 VOICEBOT READY - Bắt đầu trò chuyện!")
    print("="*60)
    print("Lệnh:")
    print("  - Gõ tin nhắn và Enter để chat")
    print("  - 'clear' - Xóa lịch sử hội thoại")
    print("  - 'quit' hoặc 'exit' - Thoát")
    print("="*60 + "\n")
    
    while True:
        try:
            # Nhập input từ user
            user_input = input("👤 You: ").strip()
            
            if not user_input:
                continue
                
            # Kiểm tra lệnh thoát
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Tạm biệt! Hẹn gặp lại!")
                break
            
            # Kiểm tra lệnh clear
            if user_input.lower() == 'clear':
                messages = [messages[0]]  # Giữ lại system message
                conversation_count = 0
                print("\n🔄 Đã xóa lịch sử hội thoại!\n")
                continue
            
            # Thêm tin nhắn user vào messages
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_input}
                ]
            })
            
            # Chạy model
            print("\n🤖 Bot: ", end="", flush=True)
            response, audio = run_model(
                model=model,
                messages=messages,
                processor=processor,
                return_audio=RETURN_AUDIO,
                use_audio_in_video=USE_AUDIO_IN_VIDEO
            )
            
            print(response)
            
            # Thêm response vào messages để giữ context
            messages.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": response}
                ]
            })
            
            # Lưu audio nếu có
            if audio is not None:
                conversation_count += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"audio_output_{conversation_count}_{timestamp}.wav"
                sample_rate = 24000
                wavfile.write(output_path, sample_rate, audio)
                print(f"🔊 Audio saved: {output_path}")
            
            print()  # Xuống dòng cho dễ nhìn
            
        except KeyboardInterrupt:
            print("\n\n👋 Đã dừng bởi Ctrl+C. Tạm biệt!")
            break
        except Exception as e:
            print(f"\n❌ Lỗi: {e}")
            print("Tiếp tục chat...\n")


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    
    print("🔄 Đang load model... (chỉ load 1 lần)")
    model, processor = _load_model_processor()
    print("✅ Model đã sẵn sàng!\n")
    
    # Bắt đầu vòng lặp conversation
    conversation_loop(model, processor)
