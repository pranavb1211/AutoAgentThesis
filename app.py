# app.py
import os
import asyncio
import math
import json
import uuid
import time
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
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

# MCP Runner
from mcprunner import MCPStdIOClient


# ============ Bootstrap ============
print("[INIT] Loading environment variables...")
load_dotenv()
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")


# ============ JSONL Loggers ============
PREDICTION_LOG = Path("predictions.jsonl")
SYSTEM_TIMINGS_LOG = Path("system_timings.jsonl")


def log_prediction(
    *,
    ticker,
    as_of_date,
    horizon_days,
    decision_text,
    news=None,
    financials=None,
    grounded=True,
    mode="live",
):
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


def append_system_timings(timings: dict):
    SYSTEM_TIMINGS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with SYSTEM_TIMINGS_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(timings, ensure_ascii=False) + "\n")


# ============ Timing instrumentation ============
def _wrap_agent_timing(agent, timings: dict, key: str):
    """
    Robustly wraps agent.on_messages (and optionally on_messages_stream) to measure
    time spent inside the agent turn.

    This DOES NOT rely on RoundRobin hooks, so you get stage_* even if on_message isn't called.
    """
    # on_messages (most common in autogen-agentchat)
    if hasattr(agent, "on_messages"):
        orig = agent.on_messages

        async def timed_on_messages(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return await orig(*args, **kwargs)
            finally:
                dt = time.perf_counter() - t0
                # If multiple calls happen (rare), keep the first stage time and also keep total
                if timings.get(key) is None:
                    timings[key] = dt
                timings[key + "_total"] = timings.get(key + "_total", 0.0) + dt

        agent.on_messages = timed_on_messages  # monkeypatch

    # on_messages_stream (some versions may stream)
    if hasattr(agent, "on_messages_stream"):
        origs = agent.on_messages_stream

        async def timed_on_messages_stream(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                async for chunk in origs(*args, **kwargs):
                    yield chunk
            finally:
                dt = time.perf_counter() - t0
                if timings.get(key) is None:
                    timings[key] = dt
                timings[key + "_total"] = timings.get(key + "_total", 0.0) + dt

        agent.on_messages_stream = timed_on_messages_stream  # monkeypatch


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

    # store for cleanup
    setattr(agent, "_az_credential", credential)
    setattr(agent, "_az_project_client", project_client)
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
    t0 = time.perf_counter()
    out = fetch_min_fin_json(ticker)
    dt = time.perf_counter() - t0
    print(f"[TIME] yfinance_agent_tool_sec = {dt:.2f}s")
    return out


# ---- FinGPT tool runner (async wrapper around local sync function) ----
async def fingpt_analysis_tool_runner(stock_name: str, context: str, horizon_days: int) -> str:
    # IMPORTANT: You said you're moving away from strict JSON.
    # Use the same lightweight tag contract you benchmarked (stable and easy).
    prompt = f"""
You are a short-horizon equity trend forecaster.

Read the JSON blocks in the context (NEWS_JSON and FIN_JSON).
Then output a short forecast using ONLY facts from those JSON blocks.

Output MUST include ALL fields using EXACT labels:

TICKER: {stock_name}
HORIZON_DAYS: {horizon_days}
OUTLOOK: bullish|bearish|neutral
CONFIDENCE: <number between 0.0 and 1.0>
VERDICT: Buy|Hold|Sell
RATIONALE: <2-3 short sentences using facts from the JSON only>

Rules:
- Use ONLY the provided JSON content. Do not invent facts.
- Do NOT output JSON.
- Do NOT use markdown or bullet points.

Context:
{context}
""".strip()

    # Greedy (deterministic). DO NOT pass temperature=0.0 anywhere.
    return your_fingpt_analyze_function(
        prompt,
        max_new_tokens=220,
        do_sample=False,   # greedy
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
def make_summary_combiner(model_client: AzureOpenAIChatCompletionClient, horizon_days: int, timings: dict) -> AssistantAgent:
    async def fingpt_tool(stock_name: str, context: str) -> str:
        t0 = time.perf_counter()
        out = await fingpt_analysis_tool_runner(stock_name, context, horizon_days)
        dt = time.perf_counter() - t0
        timings["stage_fingpt_sec"] = dt
        print(f"[TIME] fingpt_infer_sec = {dt:.2f}s")
        return out

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


# ============ Orchestration ============
async def run_app(stock_symbol: str, horizon_days: int):
    run_id = str(uuid.uuid4())
    timings = {
        "run_id": run_id,
        "ticker": stock_symbol,
        "horizon_days": int(horizon_days),
        "pipeline_total_sec": None,
        "stage_news_sec": None,
        "stage_financials_sec": None,
        "stage_fingpt_sec": None,     # set inside FinGPT tool wrapper (SummaryCombiner)
        "stage_decision_sec": None,
        # optional totals if agent gets called multiple times
        "stage_news_sec_total": 0.0,
        "stage_financials_sec_total": 0.0,
        "stage_decision_sec_total": 0.0,
    }

    t_pipeline = time.perf_counter()

    print("[INIT] Setting up Azure OpenAI client...")
    model_client = AzureOpenAIChatCompletionClient(
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        azure_deployment=os.getenv("MODEL_DEPLOYMENT_NAME", "gpt-4o"),
        model="gpt-4o-2024-11-20",
        api_version=os.getenv("MODEL_API_VERSION"),
    )

    # Build agents
    print("[MAIN] Creating NewsAgent...")
    t0 = time.perf_counter()
    news_agent = await create_news_agent(stock_symbol)
    dt = time.perf_counter() - t0
    print(f"[TIME] news_agent_setup_sec = {dt:.2f}s")

    print("[MAIN] Initializing team chat...")
    financials_agent = make_financials_agent(model_client)
    summary_combiner = make_summary_combiner(model_client, horizon_days, timings)
    decision_agent = make_decision_agent(model_client)

    # --- THIS is the fix for your stage_* being 0/null:
    # Instrument agents directly (does not rely on RoundRobin hooks).
    _wrap_agent_timing(news_agent, timings, "stage_news_sec")
    _wrap_agent_timing(financials_agent, timings, "stage_financials_sec")
    _wrap_agent_timing(decision_agent, timings, "stage_decision_sec")

    text_termination = TextMentionTermination("Decision Made")
    team = RoundRobinGroupChat(
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

    result = None
    try:
        result = await team.run(task=task)
        timings["pipeline_total_sec"] = time.perf_counter() - t_pipeline

        print("\n🔍 Final Decision:\n")
        for msg in result.messages:
            print(f"{msg.source}: {msg.content}")

        # Persist record
        final = next((m for m in reversed(result.messages) if m.source == "DecisionAgent"), None)
        news_msg = next((m for m in reversed(result.messages) if m.source == "NewsAnalyzer"), None)
        fin_msg = next((m for m in reversed(result.messages) if m.source == "financials_agent"), None)

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

        # Write timings JSONL + print
        append_system_timings(timings)
        print("\n=== SYSTEM TIMINGS (sec) ===")
        print(json.dumps(timings, indent=2))
        print("=== END TIMINGS ===\n")

        # ======================================================
        #  Slack MCP Notification
        # ======================================================
        try:
            REGISTRY_PATH = r"C:\mcpCnfig\mcpServerList.json"
            SLACK_CHANNEL_ID = "C09D8UXSDLL"

            print("\n[SLACK] Sending DecisionAgent output to Slack...")
            mcp_client = MCPStdIOClient(REGISTRY_PATH)
            await mcp_client.ensure_started("slack")

            summary_text = final.content if final else ""
            if not summary_text:
                msg = next((m for m in reversed(result.messages) if m.source == "SummaryCombiner"), None)
                if msg:
                    summary_text = msg.content

            if not summary_text:
                summary_text = f"No summary found for {stock_symbol}"

            payload = f"📊 Summary for {stock_symbol} ({horizon_days}-day horizon):\n\n{summary_text}"

            t0 = time.perf_counter()
            resp = await mcp_client.call(
                "slack",
                "conversations_add_message",
                {"channel_id": SLACK_CHANNEL_ID, "payload": payload},
            )
            dt = time.perf_counter() - t0
            print(f"[TIME] slack_post_sec = {dt:.2f}s")
            print("[SLACK] Response:", resp)

            await mcp_client.close()
            print("[SLACK] ✅ Notification sent.")
        except Exception as e:
            print(f"[SLACK] ❌ Failed to post to Slack: {e}")

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
            pc = getattr(news_agent, "_az_project_client", None)
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
