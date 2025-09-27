# app.py
import os
import asyncio
import math
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import yfinance as yf

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.conditions import TextMentionTermination

# Azure AI Studio Bing tool + project access
from azure.identity.aio import DefaultAzureCredential
from azure.ai.projects.aio import AIProjectClient
from azure.ai.agents.models import BingGroundingTool
from autogen_ext.agents.azure._azure_ai_agent import AzureAIAgent

# Local FinGPT adapter (sync function)
from adapters.fingpt_local import your_fingpt_analyze_function


# ============ Bootstrap ============
print("[INIT] Loading environment variables...")
load_dotenv()
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

# ============ JSONL Logger ============
PREDICTION_LOG = Path("predictions.jsonl")

def log_prediction(*, ticker, as_of_date, horizon_days, decision_text,
                   news=None, financials=None, grounded=True, mode="live"):
    rec = {
        "timestamp": datetime.utcnow().isoformat(),
        "mode": mode,
        "run_id": str(uuid.uuid4()),
        "ticker": ticker,
        "as_of_date": as_of_date,
        "horizon_days": int(horizon_days),
        "decision_text": decision_text,
        "grounded": bool(grounded),
        "news": news or "",
        "financials": financials or "",
    }
    PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with PREDICTION_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ============ Agents + Tools ============
async def create_news_agent(stock_symbol: str) -> AzureAIAgent:
    """
    Creates a NewsAnalyzer AzureAIAgent that MUST use BingGroundingTool
    with a freshness window covering the last 90 days, returning a strict JSON array.
    """
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=90)
    date_range_str = f"{start_date}..{end_date}"

    print(f"[BING] Setting up NewsAnalyzer for {stock_symbol}...")
    credential = DefaultAzureCredential()
    project_client = AIProjectClient(
        credential=credential,
        endpoint=os.getenv("AZURE_PROJECT_ENDPOINT"),
    )
    conn = await project_client.connections.get(name=os.getenv("BING_CONNECTION_NAME"))
    bing_tool = BingGroundingTool(conn.id, freshness=date_range_str)

    agent = AzureAIAgent(
        name="NewsAnalyzer",
        description=f"Summarizes recent {stock_symbol}-related news with citations using Bing search.",
        project_client=project_client,
        deployment_name=os.getenv("MODEL_DEPLOYMENT_NAME", "gpt-4o"),
        instructions=(
            f"You are the NewsAnalyzer. You MUST use the Bing search tool provided "
            f"to find the most relevant news about {stock_symbol} from the last 90 days. "
            "Do not answer from memory.\n"
            "Process:\n"
            "1) Use the Bing search tool to fetch recent articles.\n"
            "2) Keep items likely to move the stock (earnings, analyst changes, M&A, guidance, legal/regulatory).\n"
            "3) Remove duplicates/low-impact items.\n"
            "4) Return a STRICT JSON array (and nothing else) where each item is:\n"
            "   {\"date\": \"YYYY-MM-DD\", \"headline\": str, \"summary\": str, "
            "    \"impact_direction\": \"positive\"|\"negative\"|\"neutral\", "
            "    \"sources\": [str_url, ...]}\n"
            "No forecasts or opinions. No prose outside JSON."
        ),
        tools=bing_tool.definitions,
        metadata={"source": "AzureAIAgent"},
    )
    setattr(agent, "_az_credential", credential)
    return agent


# ---- Minimal financials: only the six fields you want ----
def fetch_min_fin_json(stock_symbol: str) -> str:
    t = yf.Ticker(stock_symbol)
    info = {}
    try:
        info = t.info or {}
    except Exception:
        pass

    hist = t.history(period="1mo")
    sma10 = sma20 = rsi14 = None

    if hist is not None and not hist.empty:
        h = hist.copy()
        # SMA10, SMA20
        h["SMA10"] = h["Close"].rolling(10).mean()
        h["SMA20"] = h["Close"].rolling(20).mean()
        sma10 = float(h["SMA10"].iloc[-1]) if not math.isnan(h["SMA10"].iloc[-1]) else None
        sma20 = float(h["SMA20"].iloc[-1]) if not math.isnan(h["SMA20"].iloc[-1]) else None

        # RSI14
        delta = h["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        roll_up = gain.rolling(14).mean()
        roll_down = loss.rolling(14).mean()
        rs = roll_up / roll_down.replace(0, np.nan)
        rsi_val = 100 - (100 / (1 + rs.iloc[-1])) if rs.iloc[-1] is not None else None
        if rsi_val is not None and not (isinstance(rsi_val, float) and math.isnan(rsi_val)):
            rsi14 = float(rsi_val)

    fin = {
        "ticker": stock_symbol,
        "pe_ttm": info.get("trailingPE"),
        "pe_fwd": info.get("forwardPE"),
        "market_cap": info.get("marketCap"),
        "sma10": sma10,
        "sma20": sma20,
        "rsi14": rsi14,
    }
    return json.dumps(fin, ensure_ascii=False)


async def yfinance_agent_tool(ticker: str) -> str:
    # returns a JSON string with only the agreed fields
    return fetch_min_fin_json(ticker)


# ---- FinGPT tool runner (async wrapper around local sync function) ----
async def fingpt_analysis_tool_runner(stock_name: str, context: str, horizon_days: int) -> str:
    prompt = f"""
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

    # Greedy (stable schema), ample budget (your fingpt_local anchors/brace-stops)
    return your_fingpt_analyze_function(
        prompt,
        max_new_tokens=360,
        do_sample=False,   # IMPORTANT: greedy
    )


# ---- Financials Agent ----
def make_financials_agent(model_client: AzureOpenAIChatCompletionClient) -> AssistantAgent:
    return AssistantAgent(
        name="financials_agent",
        model_client=model_client,
        tools=[yfinance_agent_tool],
        system_message=(
            "You are the Financials Agent. You MUST use the provided `yfinance_agent_tool` "
            "to fetch metrics for the given stock. Do not answer from memory. "
            "Call the tool exactly once using the ticker provided.\n\n"
            "Return ONLY a JSON object with these fields and nothing else:\n"
            "{\n"
            "  \"ticker\": str,\n"
            "  \"pe_ttm\": number|null,\n"
            "  \"pe_fwd\": number|null,\n"
            "  \"market_cap\": number|null,\n"
            "  \"sma10\": number|null,\n"
            "  \"sma20\": number|null,\n"
            "  \"rsi14\": number|null\n"
            "}\n"
            "No prose. No extra keys."
        ),
    )


# ---- SummaryCombiner (wires in FinGPT tool) ----
def make_summary_combiner(model_client: AzureOpenAIChatCompletionClient, horizon_days: int) -> AssistantAgent:
    async def fingpt_tool(stock_name: str, context: str) -> str:
        return await fingpt_analysis_tool_runner(stock_name, context, horizon_days)

    return AssistantAgent(
        name="SummaryCombiner",
        model_client=model_client,
        tools=[fingpt_tool],
        system_message=(
            "You are the SummaryCombiner.\n"
            "1) Find the most recent full message from 'NewsAnalyzer' (STRICT JSON array) "
            "   and the most recent full message from 'financials_agent' (STRICT JSON object).\n"
            "2) Do NOT edit, reformat, or summarize them.\n"
            "3) Build the context EXACTLY as two labeled blocks and pass them verbatim:\n"
            "NEWS_JSON:\\n<news-json-array>\\n\\nFIN_JSON:\\n<financials-json-object>\n"
            "4) Call `fingpt_tool(stock_name=<TICKER>, context=<that context>)`.\n"
            "Return ONLY the tool output."
        ),
    )


# ---- Decision Agent ----
def make_decision_agent(model_client: AzureOpenAIChatCompletionClient) -> AssistantAgent:
    return AssistantAgent(
        name="DecisionAgent",
        model_client=model_client,
        system_message=(
            "You are the Decision Agent. After reviewing all information from the other agents — "
            "including financial metrics, news, trends, and the FinGPT tool result — "
            "decide if investing in the stock is advisable. "
            "Base your decision on recent performance, news impact, market sentiment, and financial health. "
            "Finish your response with: 'Decision Made'."
        ),
    )


# ---- Round-robin with logging ----
class LoggingRoundRobinChat(RoundRobinGroupChat):
    async def on_message(self, message, sender, receiver):
        try:
            sname = getattr(sender, "name", "")
            content = getattr(message, "content", None)
            if sname == "NewsAnalyzer":
                print("\n[LOG] NewsAnalyzer returned:\n")
                print(content if content is not None else message)
                print("\n[END LOG]\n")
            elif sname == "financials_agent":
                print("\n[LOG] financials_agent returned:\n")
                print(content if content is not None else message)
                print("\n[END LOG]\n")
        except Exception as e:
            print(f"[WARN] Logging failed in on_message: {e}")
        return await super().on_message(message, sender, receiver)


# ============ Orchestration ============
async def run_app(stock_symbol: str, horizon_days: int):
    print("[INIT] Setting up Azure OpenAI client...")
    model_client = AzureOpenAIChatCompletionClient(
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        azure_deployment=os.getenv("MODEL_DEPLOYMENT_NAME", "gpt-4o"),
        model="gpt-4o-2024-11-20",
        api_version=os.getenv("MODEL_API_VERSION"),
    )

    # Build agents
    print("[MAIN] Creating NewsAgent...")
    news_agent = await create_news_agent(stock_symbol)

    print("[MAIN] Initializing team chat...")
    financials_agent = make_financials_agent(model_client)
    summary_combiner = make_summary_combiner(model_client, horizon_days)
    decision_agent = make_decision_agent(model_client)

    text_termination = TextMentionTermination("Decision Made")
    team = LoggingRoundRobinChat(
        participants=[news_agent, financials_agent, summary_combiner, decision_agent],
        termination_condition=text_termination,
    )

    print("[MAIN] Sending first task to agents...")
    task = TextMessage(
        content=(
            f"Analyze stock {stock_symbol}. "
            "All agents should use this ticker. "
            "NewsAnalyzer provides recent news; financials_agent provides minimal metrics; "
            "SummaryCombiner will merge both and call the FinGPT tool for the forecast."
        ),
        source="user",
    )

    try:
        result = await team.run(task=task)

        print("\n🔍 Final Decision:\n")
        for msg in result.messages:
            print(f"{msg.source}: {msg.content}")

        # Persist record
        final = next((m for m in reversed(result.messages) if m.source == "DecisionAgent"), None)
        news_msg = next((m for m in reversed(result.messages) if m.source == "NewsAnalyzer"), None)
        fin_msg  = next((m for m in reversed(result.messages) if m.source == "financials_agent"), None)

        log_prediction(
            ticker=stock_symbol,
            as_of_date=datetime.today().strftime("%Y-%m-%d"),
            horizon_days=horizon_days,
            decision_text=final.content if final else "",
            news=news_msg.content if news_msg else "",
            financials=fin_msg.content if fin_msg else "",
            grounded=True,
            mode="live",
        )
        print(f"[LOG] Prediction appended to {PREDICTION_LOG}")

        print("\n=== DEBUG: Agent outputs before SummaryCombiner (recap) ===")
        for msg in result.messages:
            if msg.source in ("NewsAnalyzer", "financials_agent"):
                print(f"\n--- {msg.source} ---\n{msg.content}\n")

    finally:
        # Clean up Azure OpenAI aiohttp session
        try:
            inner = getattr(model_client, "_client", None)
            if inner is not None and hasattr(inner, "close"):
                await inner.close()
        except Exception:
            pass

        # Clean up Azure AI Project + credential used by NewsAnalyzer
        try:
            pc = getattr(news_agent, "project_client", None)
            if pc is not None and hasattr(pc, "close"):
                await pc.close()
        except Exception:
            pass
        try:
            cred = getattr(news_agent, "_az_credential", None)
            if cred is not None and hasattr(cred, "close"):
                await cred.close()
        except Exception:
            pass


# ============ Entry Point ============
if __name__ == "__main__":
    stock_symbol = input("Enter the stock symbol (e.g., AAPL, MSFT, INFY): ").strip().upper()
    try:
        horizon_days = int(input("Enter forecast horizon in days (1–30): ").strip())
    except ValueError:
        horizon_days = 30
    if not (1 <= horizon_days <= 30):
        print("[WARN] Invalid horizon; defaulting to 30 days.")
        horizon_days = 30

    print(f"[START] Running analysis for {stock_symbol} with {horizon_days}-day horizon...\n")
    asyncio.run(run_app(stock_symbol, horizon_days))
    print("\n[END] Analysis complete.")
