# adapters/fingpt_local.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from dotenv import load_dotenv

# Load .env if present (optional)
load_dotenv()

# --- Set your paths here or via env vars ---
BASE_MODEL_DIR = os.getenv("FINGPT_BASE_DIR", r"C:\hf\models\llama2-7b-chat")
LORA_DIR       = os.getenv("FINGPT_LORA_DIR", r"C:\hf\models\fingpt_adapter")  # or HF repo id like "FinGPT/…"

# Singletons
_tokenizer = None
_model = None


def _ensure_paths():
    if not os.path.isfile(os.path.join(BASE_MODEL_DIR, "config.json")):
        raise FileNotFoundError(
            f"Base model not found at {BASE_MODEL_DIR} (config.json missing). "
            "Set FINGPT_BASE_DIR or place the model there."
        )
    if os.path.isdir(LORA_DIR):
        if not any(os.path.isfile(os.path.join(LORA_DIR, f))
                   for f in ("adapter_config.json", "adapter_model.bin", "adapter_model.safetensors")):
            raise FileNotFoundError(
                f"LoRA folder {LORA_DIR} has no adapter files. "
                "Download the FinGPT adapter or point FINGPT_LORA_DIR to its HF repo id."
            )
    # If LORA_DIR is an HF id (string), that's fine — no local check.


def _load_once():
    global _tokenizer, _model
    if _model is not None:
        return

    _ensure_paths()

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32

    # Tokenizer (local only)
    _tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_DIR, use_fast=True, local_files_only=True
    )
    if _tokenizer.pad_token_id is None and _tokenizer.eos_token_id is not None:
        _tokenizer.pad_token = _tokenizer.eos_token

    # Base model (local only). Use GPU if available.
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        local_files_only=True,
        dtype=dtype,                 # <— modern kwarg
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_safetensors=True,
    )
    if use_cuda:
        base = base.to("cuda")

    # Apply LoRA (local folder OR HF repo id)
    if os.path.isdir(LORA_DIR):
        model = PeftModel.from_pretrained(base, LORA_DIR, local_files_only=True)
    else:
        model = PeftModel.from_pretrained(base, LORA_DIR)

    # Optional: merge LoRA to slightly reduce VRAM; ignore if not supported
    try:
        model = model.merge_and_unload()
    except Exception:
        pass

    model.eval()
    _model = model


def your_fingpt_analyze_function(
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.3,
    top_p: float = 0.9,
) -> str:
    """Generate a short continuation with FinGPT-adapted LLaMA2."""
    _load_once()

    inputs = _tokenizer(prompt, return_tensors="pt")
    # Move tensors to same device as model
    device = next(_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min(16, max_new_tokens),   # force a bit of continuation
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            no_repeat_ngram_size=3,
            pad_token_id=_tokenizer.pad_token_id,
            eos_token_id=_tokenizer.eos_token_id,
            use_cache=True,
        )

    # Decode only the continuation (exclude the prompt)
    cont_ids = out[0][input_len:]
    text = _tokenizer.decode(cont_ids, skip_special_tokens=True).strip()
    return text or _tokenizer.decode(out[0], skip_special_tokens=True)


# Tiny CLI: python -m adapters.fingpt_local "your prompt"
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("prompt", nargs="?", default="Give a one-sentence market outlook for the Dow Jones today.")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--top-p", type=float, default=0.9)
    a = p.parse_args()
    print(
        your_fingpt_analyze_function(
            a.prompt, max_new_tokens=a.max_new_tokens, temperature=a.temperature, top_p=a.top_p
        )
    )
