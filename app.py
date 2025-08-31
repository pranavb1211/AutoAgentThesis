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

# Your local FinGPT adapter (you already have this)
from adapters.fingpt_local import your_fingpt_analyze_function


# =========================
# Bootstrap
# =========================
print("[INIT] Loading environment variables...")
load_dotenv()

print("[INIT] Setting up Azure OpenAI client...")
model_client = AzureOpenAIChatCompletionClient(
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    azure_deployment=os.getenv("MODEL_DEPLOYMENT_NAME"),
    model="gpt-4o-2024-11-20",
    api_version=os.getenv("MODEL_API_VERSION"),
)

# =========================
# JSONL Logger (for backtesting)
# =========================
PREDICTION_LOG = Path("predictions.jsonl")

def log_prediction(*, ticker, as_of_date, horizon_days, decision_text,
                   news=None, financials=None, grounded=True, mode="live"):
    rec = {
        "timestamp": datetime.utcnow().isoformat(),
        "mode": mode,                   # "live" for app.py
        "run_id": str(uuid.uuid4()),    # unique run ID
        "ticker": ticker,
        "as_of_date": as_of_date,       # YYYY-MM-DD
        "horizon_days": int(horizon_days),
        "decision_text": decision_text,
        "grounded": bool(grounded),
        "news": news or "",
        #"financials": financials or "",
    }
    PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with PREDICTION_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# =========================
# Agents + Tools
# =========================
async def create_news_agent(stock_symbol: str):
    """
    Creates a NewsAnalyzer AzureAIAgent that MUST use BingGroundingTool with a freshness
    window covering the last 90 days (live mode).
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

    return AzureAIAgent(
        name="NewsAnalyzer",
        description=f"Summarizes recent {stock_symbol}-related news with citations using Bing search.",
        project_client=project_client,
        deployment_name="gpt-4o",
        instructions=(
            f"You are the NewsAnalyzer. You MUST use the Bing search tool provided "
            f"to find the most relevant news about {stock_symbol} from the last 90 days. "
            "Do not answer from memory. "
            "Follow this process:\n"
            "1. Use the Bing search tool with appropriate keywords to fetch recent articles.\n"
            "2. Select only items likely to influence the stock price: earnings, analyst changes, M&A, etc.\n"
            "3. Remove duplicates and low-impact news.\n"
            "4. Return the final set as a JSON array with: date, headline, summary, impact_direction, sources.\n"
            "No forecasts or opinions."
        ),
        tools=bing_tool.definitions,
        metadata={"source": "AzureAIAgent"},
    )


def fetch_yfinance_data(stock_symbol: str) -> str:
    """
    Fetch richer yfinance data with safe fallbacks:
    - Meta: name, ticker, exchange, sector, industry, currency
    - Valuation/Risk: market cap, trailing/forward PE, EPS (ttm), dividend yield, beta, 52w high/low
    - Technical snapshot: SMA10, SMA20, RSI14 (latest)
    - Price history: last 30 trading days with Open/Close/Volume + SMA10/SMA20/RSI14 columns
    """
    print(f"[YFINANCE] Fetching market data for {stock_symbol}...")
    try:
        def fmt_num(n, digits=2):
            if n is None:
                return "N/A"
            try:
                if isinstance(n, (int, np.integer)) or (isinstance(n, float) and float(n).is_integer()):
                    return f"{int(n):,}"
                return f"{float(n):,.{digits}f}"
            except Exception:
                return "N/A"

        def pct(n, digits=2):
            try:
                if n is None:
                    return "N/A"
                return f"{float(n)*100:.{digits}f}%"
            except Exception:
                return "N/A"

        def currency_symbol(code: str | None) -> str:
            if not code:
                return ""
            code = code.upper()
            return {"USD": "$", "INR": "₹", "EUR": "€", "GBP": "£", "JPY": "¥"}.get(code, "")

        ticker = yf.Ticker(stock_symbol)
        # info might sometimes raise; guard it
        try:
            info = ticker.info or {}
        except Exception:
            info = {}

        try:
            fast = getattr(ticker, "fast_info", None) or {}
        except Exception:
            fast = {}

        # Meta
        name = info.get("longName") or info.get("shortName") or stock_symbol
        exchange = info.get("exchange") or info.get("fullExchangeName") or "N/A"
        sector = info.get("sector") or "N/A"
        industry = info.get("industry") or "N/A"
        currency = info.get("currency") or getattr(fast, "currency", None) or "N/A"
        cur_sym = currency_symbol(currency)

        # Valuation / risk
        market_cap = info.get("marketCap") or getattr(fast, "market_cap", None)
        pe_trailing = info.get("trailingPE")
        pe_forward = info.get("forwardPE")
        eps_ttm = info.get("trailingEps")
        div_yield = info.get("dividendYield")
        beta = info.get("beta")
        wk52_high = info.get("fiftyTwoWeekHigh")
        wk52_low = info.get("fiftyTwoWeekLow")

        # 1-month price history
        hist = ticker.history(period="1mo")
        if hist is None or hist.empty:
            hist_txt = "No recent price history available."
            sma10_latest = "N/A"
            sma20_latest = "N/A"
            rsi14_latest = "N/A"
        else:
            h = hist.copy()
            h["SMA10"] = h["Close"].rolling(10).mean()
            h["SMA20"] = h["Close"].rolling(20).mean()

            # RSI14
            delta = h["Close"].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            roll_up = gain.rolling(14).mean()
            roll_down = loss.rolling(14).mean()
            rs = roll_up / roll_down.replace(0, np.nan)
            h["RSI14"] = 100 - (100 / (1 + rs))

            sma10_latest = fmt_num(h["SMA10"].iloc[-1]) if not math.isnan(h["SMA10"].iloc[-1]) else "N/A"
            sma20_latest = fmt_num(h["SMA20"].iloc[-1]) if not math.isnan(h["SMA20"].iloc[-1]) else "N/A"
            rsi_val = h["RSI14"].iloc[-1]
            rsi14_latest = f"{float(rsi_val):.2f}" if isinstance(rsi_val, (float, np.floating)) and not math.isnan(rsi_val) else "N/A"

            # Last ~30 rows with derived columns
            cols = ["Open", "Close", "Volume", "SMA10", "SMA20", "RSI14"]
            subset = h[cols].tail(30).copy()
            subset["Open"] = subset["Open"].map(lambda x: f"{x:.2f}")
            subset["Close"] = subset["Close"].map(lambda x: f"{x:.2f}")
            subset["Volume"] = subset["Volume"].map(lambda x: f"{int(x):,}")
            subset["SMA10"] = subset["SMA10"].map(lambda x: "N/A" if pd.isna(x) else f"{x:.2f}")
            subset["SMA20"] = subset["SMA20"].map(lambda x: "N/A" if pd.isna(x) else f"{x:.2f}")
            subset["RSI14"] = subset["RSI14"].map(lambda x: "N/A" if pd.isna(x) else f"{x:.2f}")
            hist_txt = subset.to_string()

        # Build output
        out = []
        out.append(f"📊 {name} ({stock_symbol})")
        out.append(f"- Exchange: {exchange}")
        out.append(f"- Sector / Industry: {sector} / {industry}")
        out.append(f"- Currency: {currency}")

        out.append("\n💰 Valuation & Risk")
        out.append(f"- Market Cap: {cur_sym}{fmt_num(market_cap)}")
        out.append(f"- P/E (TTM): {fmt_num(pe_trailing)}")
        out.append(f"- P/E (Fwd): {fmt_num(pe_forward)}")
        out.append(f"- EPS (TTM): {fmt_num(eps_ttm)}")
        out.append(f"- Dividend Yield: {pct(div_yield)}")
        out.append(f"- Beta: {fmt_num(beta)}")
        out.append(f"- 52W Range: {cur_sym}{fmt_num(wk52_low)} — {cur_sym}{fmt_num(wk52_high)}")

        out.append("\n🧭 Technical Snapshot (latest)")
        out.append(f"- SMA10: {cur_sym}{sma10_latest}")
        out.append(f"- SMA20: {cur_sym}{sma20_latest}")
        out.append(f"- RSI14: {rsi14_latest}")

        out.append("\n📈 Price History (Last ~30 trading days)")
        out.append(hist_txt)

        return "\n".join(out)

    except Exception as e:
        return f"❌ Error fetching yfinance data for {stock_symbol}: {e}"


async def yfinance_agent_tool(stock_name: str) -> str:
    return fetch_yfinance_data(stock_name)


# ---- FinGPT core tool (async) ----
async def fingpt_analysis_tool(stock_name: str, context: str = "", horizon_days: int = 30) -> str:
    print(f"[FinGPT] Running FinGPT forecast for {stock_name} over {horizon_days} days...")
    fingpt_input = f"""
    Stock: {stock_name}

    Context data:
    {context}

    Task: Perform a short-term trend forecast for the stock over the next {horizon_days} day(s) if I invest right now.
    Use technical analysis and market patterns.
    Answer with a clear forecast (Bullish / Bearish / Neutral) and reasoning.
    """
    return your_fingpt_analyze_function(fingpt_input)


# ---- Financials Agent (global; tool is async function above) ----
financials_agent_assistant = AssistantAgent(
    name="financials_agent",
    model_client=model_client,
    tools=[yfinance_agent_tool],
    system_message=(
        "You are the Financials Agent. You MUST use the provided `yfinance_agent_tool` "
        "to fetch all financial metrics for the given stock. Do not answer from memory. "
        "Call the tool exactly once using the ticker provided.\n\n"
        "Retrieve ONLY:\n"
        "- P/E ratio\n"
        "- Market cap\n"
        "- Revenue (last available)\n"
        "- Simple moving averages (if possible)\n"
        "- Price history for the last 30 days\n"
        "Format as plain text with clear labels.\n"
        "Do NOT give forecasts or commentary."
    )
)


# ---- SummaryCombiner factory (captures horizon_days) ----
def make_summary_combiner(horizon_days: int) -> AssistantAgent:
    async def call_fingpt(stock_name: str, context: str):
        return await fingpt_analysis_tool(stock_name, context, horizon_days)

    return AssistantAgent(
        name="SummaryCombiner",
        model_client=model_client,
        tools=[call_fingpt],
        system_message=(
            "You are the SummaryCombiner. Before taking any action, read the chat history "
            "and locate the most recent full message from 'NewsAnalyzer' and the most recent full message "
            "from 'financials_agent'.\n\n"
            "Step 1: Extract their entire contents without omission.\n"
            "Step 2: Build the context as:\n"
            "'NEWS:\n<NewsAnalyzer content>\n\nFINANCIALS:\n<financials_agent content>'\n"
            f"Step 3: Call `fingpt_analysis_tool(stock_name=<TICKER>, context=<that context>, horizon_days={horizon_days})`.\n"
            "Do not forecast yourself."
        ),
    )


decision_agent = AssistantAgent(
    name="DecisionAgent",
    model_client=model_client,
    system_message=(
        "You are the Decision Agent. After reviewing all information from the other agents — including financial metrics, "
        "news, trends, and sentiment — you must decide whether investing in the stock is advisable. "
        "Base your decision on recent performance, news impact, market sentiment, and financial health. "
        "Finish your response with: 'Decision Made'."
    ),
)


# ---- Logging subclass (so you can see intermediate outputs live) ----
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


# =========================
# Main Orchestration
# =========================
async def main(stock_symbol: str, horizon_days: int):
    print("[MAIN] Creating NewsAgent...")
    news_agent = await create_news_agent(stock_symbol)

    print("[MAIN] Initializing team chat...")
    text_termination = TextMentionTermination("Decision Made")
    summary_combiner_agent = make_summary_combiner(horizon_days)

    team = LoggingRoundRobinChat(
        participants=[news_agent, financials_agent_assistant, summary_combiner_agent, decision_agent],
        termination_condition=text_termination,
    )

    print("[MAIN] Sending first task to agents...")
    task = TextMessage(
        content=(
            f"Analyze stock {stock_symbol}. "
            "All agents should use this ticker. "
            "NewsAnalyzer provides recent news; financials_agent provides metrics; "
            "SummaryCombiner will merge and call the FinGPT tool."
        ),
        source="user",
    )

    result = await team.run(task=task)

    print("\n🔍 Final Decision:\n")
    for msg in result.messages:
        print(f"{msg.source}: {msg.content}")

    # ---- Persist a prediction record for backtesting ----
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

    # Optional recap:
    print("\n=== DEBUG: Agent outputs before SummaryCombiner (recap) ===")
    for msg in result.messages:
        if msg.source in ("NewsAnalyzer", "financials_agent"):
            print(f"\n--- {msg.source} ---\n{msg.content}\n")


# =========================
# Entry Point
# =========================
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
    asyncio.run(main(stock_symbol, horizon_days))
    print("\n[END] Analysis complete.")
