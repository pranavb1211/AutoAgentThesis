import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import snapshot_download

# A hint to your local cache root (what you showed in the screenshot)
LOCAL_HINT = r"C:\hf\models\llama2-7b-chat"
HF_ID = "meta-llama/Llama-2-7b-chat-hf"
ADAPTER = "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora"

_tokenizer = None
_model = None

def _find_local_snapshot() -> tuple[str, bool]:
    """Return (path, is_local). Tries:
       1) LOCAL_HINT/snapshots/<hash>
       2) Already-cached snapshot via snapshot_download(local_files_only=True)
       3) Fallback to HF model id (network)"""
    # If user pointed to the snapshot already
    if os.path.isfile(os.path.join(LOCAL_HINT, "config.json")):
        return LOCAL_HINT, True

    # If user pointed to the model root: pick a snapshot dir with config.json
    snap_root = os.path.join(LOCAL_HINT, "snapshots")
    if os.path.isdir(snap_root):
        candidates = [
            os.path.join(snap_root, d)
            for d in os.listdir(snap_root)
            if os.path.isfile(os.path.join(snap_root, d, "config.json"))
        ]
        if candidates:
            # newest snapshot
            best = max(candidates, key=os.path.getmtime)
            return best, True

    # Ask huggingface_hub where the cached snapshot is (no network)
    try:
        cached = snapshot_download(HF_ID, local_files_only=True)
        if os.path.isfile(os.path.join(cached, "config.json")):
            return cached, True
    except Exception:
        pass

    # No local snapshot found: use HF id (will download)
    return HF_ID, False

def _load_once():
    global _tokenizer, _model
    if _model is not None:
        return

    base_path, is_local = _find_local_snapshot()
    print(f"[FinGPT] Using {'LOCAL' if is_local else 'HF'} base: {base_path}")

    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,  
        local_files_only=is_local,
    )
    _tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=is_local)
    _model = PeftModel.from_pretrained(base, ADAPTER).eval()

def your_fingpt_analyze_function(prompt: str,
                                 max_new_tokens: int = 128,
                                 temperature: float = 0.2,
                                 top_p: float = 0.9) -> str:
    _load_once()
    inputs = _tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=getattr(_tokenizer, "eos_token_id", None),
        )
    return _tokenizer.decode(out_ids[0], skip_special_tokens=True)
