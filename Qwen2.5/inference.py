"""
Inference / smoke test for fine-tuned Qwen2.5-Omni-3B TTS checkpoint.

Loads base model, merges (or attaches) the trained LoRA adapters for thinker
and talker, then generates audio for a Vietnamese prompt and writes a WAV.

Usage:
  python inference.py --ckpt checkpoints/qwen25omni-ly-tts/epoch_3 \\
                      --text "Xin chào, đây là giọng đọc tiếng Việt." \\
                      --out out.wav
"""

import argparse
from pathlib import Path

import torch
import soundfile as sf

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from peft import PeftModel


SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="Qwen/Qwen2.5-Omni-3B")
    p.add_argument("--ckpt", type=Path, required=True,
                   help="Checkpoint dir containing thinker_lora/ and/or talker_lora/")
    p.add_argument("--text", default="Xin chào, đây là một bài kiểm tra giọng nói tiếng Việt.")
    p.add_argument("--out", type=Path, default=Path("out.wav"))
    p.add_argument("--speaker", default="Chelsie", choices=["Chelsie", "Ethan"])
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print("Loading base model...")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="sdpa",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(args.base_model)

    # Attach LoRA adapters
    thinker_adapter = args.ckpt / "thinker_lora"
    if thinker_adapter.exists():
        print(f"Attaching thinker LoRA from {thinker_adapter}")
        model.thinker = PeftModel.from_pretrained(model.thinker, thinker_adapter)
        model.thinker = model.thinker.merge_and_unload()

    talker_adapter = args.ckpt / "talker_lora"
    if talker_adapter.exists():
        print(f"Attaching talker LoRA from {talker_adapter}")
        target = model.talker.model if hasattr(model.talker, "model") else model.talker
        attached = PeftModel.from_pretrained(target, talker_adapter)
        attached = attached.merge_and_unload()
        if hasattr(model.talker, "model"):
            model.talker.model = attached
        else:
            model.talker = attached

    model.eval()

    # Build the chat
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": args.text}]},
    ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items() if hasattr(v, "to")}

    print("Generating...")
    with torch.no_grad():
        text_ids, audio = model.generate(
            **inputs,
            speaker=args.speaker,
            thinker_do_sample=False,
            thinker_max_new_tokens=512,
            return_audio=True,
            use_audio_in_video=False,
        )

    response = processor.batch_decode(
        text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print(f"Thinker text response: {response}")

    if audio is None:
        print("No audio returned. Check that model.disable_talker() was not called and the talker is loaded.")
        return

    wav = audio.reshape(-1).detach().cpu().numpy()
    sf.write(args.out, wav, samplerate=24000)
    print(f"Wrote {args.out} ({len(wav)/24000:.2f}s)")


if __name__ == "__main__":
    main()
