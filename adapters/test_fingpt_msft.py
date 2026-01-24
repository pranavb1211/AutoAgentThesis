# test_fingpt_msft.py
"""
Inference-focused experiment runner for local FinGPT (LLaMA2-chat + LoRA).

This mimics your app.py scenario:
- context EXACTLY:
  NEWS_JSON:\n<news-json-array>\n\nFIN_JSON:\n<financials-json-object>
- prompt template is the SAME as app.py
- FinGPT generation logic mimics adapters/fingpt_local.py:
  - anchored "{"
  - brace-balanced stopping
  - greedy first, optional fallback sampling

Unlike your earlier version, this file DOES NOT care about JSON validity.
It logs inference metrics:
- generate_time_sec (per run)
- tokens_generated
- chars_generated
- field_coverage_rate (presence of: ticker/horizon/outlook/confidence/rationale/verdict)

Run:
  python test_fingpt_msft.py
  python test_fingpt_msft.py --runs 10 --do-sample --temperature 0.4 --top-p 0.95
  python test_fingpt_msft.py --grid
  python test_fingpt_msft.py --max-new-tokens 256 --runs 3
"""

import os
import json
import time
import argparse
from typing import Dict, Any, Tuple, List

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
from peft import PeftModel

# ----------------------------
# ENV + PATHS
# ----------------------------
load_dotenv()
BASE_MODEL_DIR = os.getenv("FINGPT_BASE_DIR", r"C:\hf\models\llama2-7b-chat")
LORA_DIR = os.getenv("FINGPT_LORA_DIR", r"C:\hf\models\fingpt_adapter")

_tokenizer = None
_model = None


# ----------------------------
# INPUT DATA (your 2025-11-05 snapshot)
# ----------------------------
MSFT_NEWS = [
    {
        "date": "2025-09-15",
        "headline": "Microsoft announces quarterly dividend increase",
        "summary": "Microsoft increased its quarterly dividend by 10% to $0.91 per share, payable December 11, 2025, reflecting positive financial stability. Shareholders will vote on business resolutions during the annual shareholder meeting scheduled for December 5, 2025.",
        "impact_direction": "positive",
        "sources": [
            "https://news.microsoft.com/source/2025/09/15/microsoft-announces-quarterly-dividend-increase-6/"
        ],
    },
    {
        "date": "2025-10-31",
        "headline": "Microsoft stock falls despite strong earnings due to AI spending concerns and analyst downgrade",
        "summary": "Microsoft exceeded Q1 revenue and EPS expectations with 40% growth in Azure and positive announcements regarding its OpenAI partnership. However, renewed concerns over heavy AI spending and an analyst downgrade put pressure on the stock, causing volatility.",
        "impact_direction": "negative",
        "sources": [
            "https://www.marketbeat.com/stocks/NASDAQ/MSFT/news/"
        ],
    },
    {
        "date": "2025-11-02",
        "headline": "Microsoft to resume hiring after AI-driven productivity shift",
        "summary": "Microsoft plans to restart hiring after previously slowing recruitment citing growth in AI productivity, offering relief to workforce concerns and potentially signaling improvement in business confidence. The company remains focused on leveraging AI for growth.",
        "impact_direction": "positive",
        "sources": [
            "https://www.msn.com/en-us/money/stocks/microsoft-msft-to-resume-hiring-after-ai-driven-productivity-shift/ar-AA1iGCVb"
        ],
    },
    {
        "date": "2025-11-02",
        "headline": "Microsoft maintains buy ratings from analysts after strong Q1 results",
        "summary": "Analysts, including Truist, continue to maintain buy ratings on MSFT following a solid fiscal Q1 performance. Analysts cite robust demand for its cloud business and AI advancements, affirming confidence despite margin pressures.",
        "impact_direction": "positive",
        "sources": [
            "https://www.insidermonkey.com/blog/microsoft-msft-maintains-buy-rating-at-truist-after-solid-fiscal-q1-results-111111.htm"
        ],
    },
]

MSFT_FIN = {
    "ticker": "MSFT",
    "pe_ttm": 36.633194,
    "pe_fwd": 34.403347,
    "market_cap": 3822694957056,
    "sma10": 525.4780029296875,
    "sma20": 520.6839981079102,
    "rsi14": 50.749770177593746,
}


# ----------------------------
# PROMPT BUILDERS (match app.py)
# ----------------------------
def build_context(news: list, fin: dict) -> str:
    news_json = json.dumps(news, ensure_ascii=False, indent=2)
    fin_json = json.dumps(fin, ensure_ascii=False, indent=2)
    return f"NEWS_JSON:\n{news_json}\n\nFIN_JSON:\n{fin_json}"


def build_prompt(stock_name: str, horizon_days: int, context: str) -> str:
    # This mirrors what you showed from app.py
    return f"""
You are a short-horizon equity trend forecaster.
Read the JSON blocks and produce a STRICT JSON response with this schema:

{{
  "ticker": "{stock_name}",
  "horizon_days": {horizon_days},
  "outlook": "bullish" | "bearish" | "neutral",
  "confidence": 0.0-1.0,
  "rationale": "2-3 short sentences using facts from the JSON only",
  "verdict": "Buy" | "Hold" | "Sell"
}}

Rules:
- Output valid JSON with double quotes around all keys and string values.
- Confidence must be a floating-point number between 0.0 and 1.0.
- Base your reasoning ONLY on the provided JSON.
- No extra fields. No prose outside JSON. Be concise.

Context:
{context}
""".strip()


# ----------------------------
# MODEL LOADING (same idea as fingpt_local)
# ----------------------------
def _ensure_paths():
    if not os.path.isfile(os.path.join(BASE_MODEL_DIR, "config.json")):
        raise FileNotFoundError(
            f"Base model not found at {BASE_MODEL_DIR} (config.json missing). "
            "Set FINGPT_BASE_DIR or place the model there."
        )
    if os.path.isdir(LORA_DIR):
        ok = any(
            os.path.isfile(os.path.join(LORA_DIR, f))
            for f in ("adapter_config.json", "adapter_model.bin", "adapter_model.safetensors")
        )
        if not ok:
            raise FileNotFoundError(
                f"LoRA folder {LORA_DIR} has no adapter files. "
                "Download the FinGPT adapter or point FINGPT_LORA_DIR to its HF repo id."
            )
    # If LORA_DIR is an HF repo id string, skip local check.


def _load_once():
    global _tokenizer, _model
    if _model is not None and _tokenizer is not None:
        return

    _ensure_paths()

    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        raise RuntimeError("[FinGPT] CUDA NOT AVAILABLE — your setup is likely CPU-only.")

    dtype = torch.float16

    print(f"[FinGPT] BASE_MODEL_DIR = {BASE_MODEL_DIR}", flush=True)
    print(f"[FinGPT] LORA_DIR       = {LORA_DIR}", flush=True)
    print(f"[FinGPT] device        = cuda | dtype={dtype}", flush=True)

    # Tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_DIR,
        use_fast=True,
        local_files_only=True,
    )
    if _tokenizer.pad_token_id is None and _tokenizer.eos_token_id is not None:
        _tokenizer.pad_token = _tokenizer.eos_token

    # Base model
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        local_files_only=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_safetensors=True,
    ).to("cuda")

    # Apply LoRA
    if os.path.isdir(LORA_DIR):
        model = PeftModel.from_pretrained(base, LORA_DIR, local_files_only=True)
    else:
        model = PeftModel.from_pretrained(base, LORA_DIR)

    try:
        model = model.merge_and_unload()
    except Exception:
        pass

    model.eval()
    _model = model


# ----------------------------
# BRACE STOPPING (same as fingpt_local)
# ----------------------------
class _BraceStop(StoppingCriteria):
    """Stop when we've produced a balanced JSON object after the anchored '{'."""

    def __init__(self, tokenizer, start_len: int):
        self.tok = tokenizer
        self.start_len = start_len

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        gen_ids = input_ids[0, self.start_len:]
        text = self.tok.decode(gen_ids, skip_special_tokens=True)
        opens = text.count("{")
        closes = text.count("}")
        return closes >= opens and closes > 0


# ----------------------------
# INFERENCE (anchored + brace stop + greedy first + optional fallback)
# ----------------------------
def your_fingpt_analyze_function(
    prompt: str,
    max_new_tokens: int = 360,
    do_sample: bool = False,       # if True: allow sampling in fallback (and/or in pass2)
    temperature: float = 0.2,      # used only when sampling is enabled
    top_p: float = 1.0,            # used only when sampling is enabled
) -> Tuple[str, float, int, bool]:
    """
    Returns:
      text: str
      gen_time_sec: float  (time spent inside _model.generate for the last pass)
      tokens_generated: int
      used_fallback: bool
    """
    _load_once()

    anchored_prompt = (
        prompt.strip()
        + "\n\nOutput VALID JSON (double-quoted keys and string values). "
          "Begin exactly with the brace below and continue the JSON:\n{"
    )

    inputs = _tokenizer(anchored_prompt, return_tensors="pt")
    device = next(_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = int(inputs["input_ids"].shape[1])

    stop = StoppingCriteriaList([_BraceStop(_tokenizer, input_len)])

    def _gen(sample: bool, maxtoks: int) -> Tuple[torch.Tensor, float]:
        gen_kwargs = dict(
            max_new_tokens=int(maxtoks),
            min_new_tokens=int(min(96, maxtoks)),
            no_repeat_ngram_size=3,
            pad_token_id=_tokenizer.pad_token_id,
            eos_token_id=_tokenizer.eos_token_id,
            use_cache=True,
            stopping_criteria=stop,
        )
        if sample:
            gen_kwargs.update(
                dict(
                    do_sample=True,
                    temperature=float(temperature),
                    top_p=float(top_p),
                )
            )
        else:
            gen_kwargs.update(dict(do_sample=False))

        # time JUST the generate call
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = _model.generate(**inputs, **gen_kwargs)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        return out, dt

    # Pass 1: greedy (exactly like thesis/app default)
    used_fallback = False
    out, dt = _gen(sample=False, maxtoks=max_new_tokens)

    gen_ids = out[0][input_len:]
    text = "{" + _tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # heuristic completion check (WITHOUT parsing)
    s = text.find("{")
    e = text.rfind("}")
    complete = (s != -1 and e != -1 and e > s)
    has_verdict = ("verdict" in text.lower())

    # Pass 2: fallback sampling (only if you allow it)
    if (not complete or not has_verdict) and do_sample:
        used_fallback = True
        out, dt = _gen(sample=True, maxtoks=max_new_tokens + 200)
        gen_ids = out[0][input_len:]
        text = "{" + _tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        s = text.find("{")
        e = text.rfind("}")

    # hard trim to first object-ish chunk
    if s != -1 and e != -1 and e > s:
        text = text[s : e + 1]

    tokens_generated = int(gen_ids.shape[0])
    return text, dt, tokens_generated, used_fallback


# ----------------------------
# EVALUATION THAT DOES NOT PARSE JSON
# ----------------------------
FIELDS = ["ticker", "horizon_days", "outlook", "confidence", "rationale", "verdict"]


def field_coverage(text: str) -> Tuple[float, int, List[str]]:
    t = text.lower()
    hits = [f for f in FIELDS if f in t]
    return (len(hits) / len(FIELDS)), len(hits), hits


def summarize_runs(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    times = [r["gen_time_sec"] for r in results]
    tokens = [r["tokens_generated"] for r in results]
    chars = [r["chars"] for r in results]
    covs = [r["field_coverage"] for r in results]
    fallbacks = sum(1 for r in results if r["used_fallback"])

    times_sorted = sorted(times)
    p95 = times_sorted[max(0, int(0.95 * (len(times_sorted) - 1)))]

    return {
        "runs": len(results),
        "avg_time_sec": sum(times) / len(times),
        "p95_time_sec": p95,
        "avg_tokens": sum(tokens) / len(tokens),
        "avg_chars": sum(chars) / len(chars),
        "avg_field_coverage": sum(covs) / len(covs),
        "fallback_rate_pct": 100.0 * fallbacks / len(results),
    }


# ----------------------------
# EXPERIMENT RUNNER
# ----------------------------
def run_experiment(
    runs: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    print_prompt_preview: bool,
) -> None:
    stock_name = "MSFT"
    horizon_days = 5

    context = build_context(MSFT_NEWS, MSFT_FIN)
    prompt = build_prompt(stock_name, horizon_days, context)

    if print_prompt_preview:
        print("\n==============================")
        print("PROMPT (first 1400 chars)")
        print("==============================")
        print(prompt[:1400] + ("..." if len(prompt) > 1400 else ""))

    results: List[Dict[str, Any]] = []

    for i in range(runs):
        print("\n==============================")
        print(f"RUN {i+1}/{runs}")
        print("==============================")

        text, dt, toks, used_fb = your_fingpt_analyze_function(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

        cov_rate, cov_hits, cov_list = field_coverage(text)

        row = {
            "run": i + 1,
            "gen_time_sec": dt,
            "tokens_generated": toks,
            "chars": len(text),
            "field_coverage": cov_rate,
            "field_hits": cov_hits,
            "fields_found": cov_list,
            "used_fallback": used_fb,
            "raw_output": text,
        }
        results.append(row)

        print(f"[time] {dt:.2f}s")
        print(f"[tokens] {toks} | [chars] {len(text)}")
        print(f"[field coverage] {cov_hits}/6 = {cov_rate:.2f}  found={cov_list}")
        if used_fb:
            print("[fallback] YES (sampling pass was used)")

        print("\n--- RAW OUTPUT (first 1200 chars) ---")
        print(text[:1200] + ("..." if len(text) > 1200 else ""))

    summary = summarize_runs(results)

    print("\n==============================")
    print("SUMMARY (inference-focused)")
    print("==============================")
    print(json.dumps(summary, indent=2))


def run_grid() -> None:
    """
    A simple grid that matches what you'd write in experiments:
    - E1: Greedy baseline (app-like)
    - E2: Sampling mild
    - E3: Sampling hotter
    - E4: Token budget sensitivity (greedy)
    """
    grids = [
        ("E1_greedy_baseline", dict(runs=3, max_new_tokens=360, do_sample=False, temperature=0.2, top_p=1.0)),
        ("E2_sampling_mild",   dict(runs=5, max_new_tokens=360, do_sample=True,  temperature=0.2, top_p=0.95)),
        ("E3_sampling_hot",    dict(runs=5, max_new_tokens=360, do_sample=True,  temperature=0.6, top_p=0.90)),
        ("E4_tokens_128",      dict(runs=3, max_new_tokens=128, do_sample=False, temperature=0.2, top_p=1.0)),
        ("E4_tokens_256",      dict(runs=3, max_new_tokens=256, do_sample=False, temperature=0.2, top_p=1.0)),
        ("E4_tokens_480",      dict(runs=3, max_new_tokens=480, do_sample=False, temperature=0.2, top_p=1.0)),
    ]

    for name, cfg in grids:
        print("\n\n============================================================")
        print(f"GRID EXP: {name}")
        print("============================================================")
        run_experiment(
            runs=cfg["runs"],
            max_new_tokens=cfg["max_new_tokens"],
            do_sample=cfg["do_sample"],
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            print_prompt_preview=False,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=360)
    ap.add_argument("--do-sample", action="store_true", help="Enable fallback sampling if greedy looks incomplete")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--show-prompt", action="store_true")
    ap.add_argument("--grid", action="store_true", help="Run a small experiment grid (E1..E4)")
    args = ap.parse_args()

    print("\n==============================")
    print("CONFIG")
    print("==============================")
    print(
        json.dumps(
            {
                "runs": args.runs,
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.do_sample,
                "temperature": args.temperature,
                "top_p": args.top_p,
            },
            indent=2,
        )
    )

    if args.grid:
        run_grid()
        return

    run_experiment(
        runs=args.runs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        print_prompt_preview=args.show_prompt,
    )


if __name__ == "__main__":
    main()
