# adapters/fingpt_local.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
from dotenv import load_dotenv

# Load .env if present (optional)
load_dotenv()

# --- Set your paths here or via env vars ---
BASE_MODEL_DIR = os.getenv("FINGPT_BASE_DIR", r"C:\hf\models\llama2-7b-chat")
LORA_DIR       = os.getenv("FINGPT_LORA_DIR", r"C:\hf\models\fingpt_adapter")

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
    # If LORA_DIR is an HF repo id string, skip local check.


def _load_once():
    """Load tokenizer + base + LoRA once into globals."""
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

    # Base model (local only)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        local_files_only=True,
        dtype=dtype,
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

    # Optional merge to reduce VRAM; skip if not supported
    try:
        model = model.merge_and_unload()
    except Exception:
        pass

    model.eval()
    _model = model


class _BraceStop(StoppingCriteria):
    """Stop when we've produced a balanced JSON object after the anchored '{'."""
    def __init__(self, tokenizer, start_len: int):
        self.tok = tokenizer
        self.start_len = start_len  # input length (tokens) before generation

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # Decode only the generated part
        gen_ids = input_ids[0, self.start_len:]
        text = self.tok.decode(gen_ids, skip_special_tokens=True)
        opens = text.count("{")
        closes = text.count("}")
        # When the first JSON object is fully closed, closes >= opens and > 0
        return closes >= opens and closes > 0


def your_fingpt_analyze_function(
    prompt: str,
    max_new_tokens: int = 360,
    do_sample: bool = False,   # greedy first for stable schema
    temperature: float = 0.2,  # used only if do_sample=True (or in fallback)
    top_p: float = 1.0,
) -> str:
    """
    Generate continuation with FinGPT-adapted LLaMA2, anchored to JSON.
    - Greedy first (stable).
    - Balanced-brace stopping to avoid mid-object truncation.
    - Automatic fallback to light sampling with bigger budget if incomplete.
    """
    _load_once()

    # Anchor the reply to start inside JSON
    anchored_prompt = (
        prompt.strip()
        + "\n\nOutput VALID JSON (double-quoted keys and string values). "
          "Begin exactly with the brace below and continue the JSON:\n{"
    )

    inputs = _tokenizer(anchored_prompt, return_tensors="pt")
    device = next(_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    stop = StoppingCriteriaList([_BraceStop(_tokenizer, input_len)])

    def _gen(sample: bool, maxtoks: int):
        gen_kwargs = dict(
            max_new_tokens=maxtoks,
            min_new_tokens=min(96, maxtoks),
            no_repeat_ngram_size=3,
            pad_token_id=_tokenizer.pad_token_id,
            eos_token_id=_tokenizer.eos_token_id,
            use_cache=True,
            stopping_criteria=stop,
        )
        if sample:
            gen_kwargs.update(dict(do_sample=True, temperature=temperature, top_p=top_p))
        else:
            gen_kwargs.update(dict(do_sample=False))
        with torch.no_grad():
            return _model.generate(**inputs, **gen_kwargs)

    # Pass 1: greedy
    out = _gen(sample=False, maxtoks=max_new_tokens)
    cont_ids = out[0][input_len:]
    cont_text = _tokenizer.decode(cont_ids, skip_special_tokens=True).strip()
    text = "{" + cont_text  # reattach the anchored brace

    # Trim to the first complete JSON object
    s = text.find("{")
    e = text.rfind("}")
    complete = (s != -1 and e != -1 and e > s)
    has_verdict = ("\"verdict\"" in text or "'verdict'" in text)

    # Fallback if needed
    if not complete or not has_verdict:
        out = _gen(sample=True, maxtoks=max_new_tokens + 200)
        cont_ids = out[0][input_len:]
        cont_text = _tokenizer.decode(cont_ids, skip_special_tokens=True).strip()
        text = "{" + cont_text
        s = text.find("{")
        e = text.rfind("}")

    # Final hard trim
    if s != -1 and e != -1 and e > s:
        text = text[s:e+1]

    return text


# Tiny CLI: python -m adapters.fingpt_local "your prompt"
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("prompt", nargs="?", default="Give a one-sentence market outlook for the Dow Jones today.")
    p.add_argument("--max-new-tokens", type=int, default=360)
    p.add_argument("--do-sample", action="store_true", help="Use sampling instead of greedy decoding")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=1.0)
    a = p.parse_args()
    print(
        your_fingpt_analyze_function(
            a.prompt, max_new_tokens=a.max_new_tokens,
            do_sample=a.do_sample, temperature=a.temperature, top_p=a.top_p
        )
    )
