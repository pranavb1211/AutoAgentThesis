import os
import asyncio
from dotenv import load_dotenv
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
from adapters.fingpt_local import your_fingpt_analyze_function


# Load .env variables
print("[INIT] Loading environment variables...")
load_dotenv()

# Setup base model client for non-Bing agents
print("[INIT] Setting up Azure OpenAI client...")
model_client = AzureOpenAIChatCompletionClient(
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    azure_deployment=os.getenv("MODEL_DEPLOYMENT_NAME"),
    model="gpt-4o",
    api_version=os.getenv("MODEL_API_VERSION")
)

async def create_news_agent(stock_symbol: str):
    print(f"[BING] Setting up NewsAnalyzer for {stock_symbol}...")
    credential = DefaultAzureCredential()
    project_client = AIProjectClient(
        credential=credential,
        endpoint=os.getenv("AZURE_PROJECT_ENDPOINT")
    )
    conn = await project_client.connections.get(name=os.getenv("BING_CONNECTION_NAME"))
    bing_tool = BingGroundingTool(conn.id)

    return AzureAIAgent(
        name="NewsAnalyzer",
        description=f"Summarizes recent {stock_symbol}-related news with citations using Bing search.",
        project_client=project_client,
        deployment_name="gpt-4o",
        instructions=(
            f"Find the most relevant news about {stock_symbol} from the past month. "
            "Focus only on events likely to influence the stock price: "
            "earnings, analyst rating changes, major partnerships, M&A, "
            "regulatory changes, and macroeconomic factors. "
            "Avoid repetitive items and exclude minor PR updates. "
            "Summarize up to 10 distinct events. "
            "Do NOT provide forecasts or opinions."
        ),
        tools=bing_tool.definitions,
        metadata={"source": "AzureAIAgent"},
    )


def fetch_yfinance_data(stock_symbol: str) -> str:
    print(f"[YFINANCE] Fetching market data for {stock_symbol}...")
    try:
        ticker = yf.Ticker(stock_symbol)
        info = ticker.info
        financials = ticker.financials

        market_cap = info.get("marketCap", "N/A")
        pe_ratio = info.get("trailingPE", "N/A")
        name = info.get("longName", stock_symbol)

        revenue = "N/A"
        if hasattr(financials, "index") and "Total Revenue" in financials.index:
            revenue = financials.loc["Total Revenue"].iloc[0]

        history = ticker.history(period="50d").tail(30)

        return f"""
📊 {name} ({stock_symbol})
- Market Cap: ₹{market_cap:,}
- P/E Ratio: {pe_ratio}
- Revenue (Last): ₹{revenue:,}

📈 Price History (Last 30 Days):
{history[["Open", "Close", "Volume"]].to_string()}
"""
    except Exception as e:
        return f"❌ Error fetching yfinance data for {stock_symbol}: {e}"

async def yfinance_agent_tool(stock_name: str) -> str:
    return fetch_yfinance_data(stock_name)


# core FinGPT analysis tool
async def fingpt_analysis_tool(stock_name: str, context: str = "") -> str:
    print(f"[FinGPT] Running FinGPT forecast for {stock_name}...")
    fingpt_input = f"""
    Stock: {stock_name}

    Context data:
    {context}

    Task: Perform a short-term trend forecast for the stock over the next one month if I invest right now.
    Use technical analysis and market patterns.
    Answer with a clear forecast (Bullish / Bearish / Neutral) and reasoning.
    """
    return your_fingpt_analyze_function(fingpt_input)


# Agents

financials_agent_assistant = AssistantAgent(
    name="financials_agent",
    model_client=model_client,
    tools=[yfinance_agent_tool],
    system_message=(
        "You are the Financials Agent. Retrieve ONLY structured financial metrics "
        "for the given stock, including P/E ratio, revenue, market cap, SMA values if possible, "
        "and price history for past 50 days. Format as plain text with clear labels. "
        "Do NOT give forecasts or commentary."
    )
)

summary_combiner_agent = AssistantAgent(
    name="SummaryCombiner",
    model_client=model_client,
    tools=[fingpt_analysis_tool],
    system_message=(
        "You are the Summary Combiner agent. "
        "You receive prior messages containing financial metrics and summarized news. "
        "Combine them into a unified context and feed it into the FinGPT analysis tool "
        "for a short-term trend forecast. "
        "Do not create your own forecast — always call the tool."
    )
)

decision_agent = AssistantAgent(
    name="DecisionAgent",
    model_client=model_client,
    system_message=(
        "You are the Decision Agent. After reviewing all information from the other agents — including financial metrics, "
        "news, trends, and sentiment — you must decide whether investing in the stock is advisable. "
        "Base your decision on recent performance, news impact, market sentiment, and financial health. "
        "Finish your response with: 'Decision Made'."
    )
)


# ---- NEW: logging subclass ----
class LoggingRoundRobinChat(RoundRobinGroupChat):
    async def on_message(self, message, sender, receiver):
        """
        Called when a message is delivered from `sender` to `receiver`.
        We log specific agent outputs immediately so it's visible *before* the next agent acts.
        """
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
        # Important: continue normal delivery
        return await super().on_message(message, sender, receiver)


# Main async run
async def main(stock_symbol: str):
    print("[MAIN] Creating NewsAgent...")
    news_agent = await create_news_agent(stock_symbol)

    print("[MAIN] Initializing team chat...")
    text_termination = TextMentionTermination("Decision Made")
    # ---- use the logging team here ----
    team = LoggingRoundRobinChat(
        participants=[news_agent, financials_agent_assistant, summary_combiner_agent, decision_agent],
        termination_condition=text_termination
    )

    print("[MAIN] Sending first task to agents...")
    task = TextMessage(
        content=f"Analyze stock {stock_symbol}. "
                f"All agents should use this ticker. "
                f"NewsAnalyzer provides recent news; financials_agent provides metrics; "
                f"SummaryCombiner will merge and call the FinGPT tool.",
        source="user"
    )

    result = await team.run(task=task)

    print("\n🔍 Final Decision:\n")
    for msg in result.messages:
        print(f"{msg.source}: {msg.content}")

    # Keep this post-run section if you want a clean recap too
    print("\n=== DEBUG: Agent outputs before SummaryCombiner (recap) ===")
    for msg in result.messages:
        if msg.source in ("NewsAnalyzer", "financials_agent"):
            print(f"\n--- {msg.source} ---\n{msg.content}\n")


if __name__ == "__main__":
    stock_symbol = input("Enter the stock symbol (e.g., AAPL, MSFT, INFY): ").strip().upper()
    print(f"[START] Running analysis for {stock_symbol}...\n")
    asyncio.run(main(stock_symbol))
    print("\n[END] Analysis complete.")
