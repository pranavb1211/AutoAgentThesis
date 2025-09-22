from adapters.fingpt_local import your_fingpt_analyze_function

ticker = "AAPL"
horizon = 5
context = """
NEWS_JSON:
[{"date":"2024-05-01","headline":"Apple beats earnings","summary":"EPS beat; services growth strong","impact_direction":"positive","sources":["https://..."]}]

FIN_JSON:
{"ticker":"AAPL","pe_ttm":28.7,"pe_fwd":26.2,"market_cap":2.9e12,"sma10":170.2,"sma20":168.9,"rsi14":58.4}
"""

prompt = f"""
You are a short-horizon equity trend forecaster.
Read the JSON blocks and produce a STRICT JSON response with this schema:

{{
  "ticker": "{ticker}",
  "horizon_days": {horizon},
  "outlook": "bullish" | "bearish" | "neutral",
  "confidence": 0.0-1.0,
  "rationale": "2-3 short sentences using facts from the JSON only",
  "verdict": "Buy" | "Hold" | "Sell"
}}

Rules:
- Base your reasoning ONLY on the provided JSON.
- No extra fields. No prose outside JSON. Be concise.

Context:
{context}
""".strip()

# Greedy = most stable for strict schema; don't pass temperature/top_p
print(your_fingpt_analyze_function(prompt, max_new_tokens=360, do_sample=False))
# For experimentation:
# print(your_fingpt_analyze_function(prompt, max_new_tokens=320, do_sample=True, temperature=0.2, top_p=1.0))
