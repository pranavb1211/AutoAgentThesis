#!/usr/bin/env python3
"""
newsagent.py — Standalone Bing-powered NewsAnalyzer

What this does
--------------
- Loads Azure project + Bing connection from environment
- Creates a single AzureAIAgent ("NewsAnalyzer") with the Bing Grounding Tool
- Runs a one-off prompt to fetch and summarize recent news for a given stock
- Prints the agent's response with citations
- Terminates when the agent outputs the word "DONE"

Prereqs
-------
pip install:
  python-dotenv azure-identity azure-ai-projects autogen-agentchat autogen-ext yfinance

.env required keys:
  AZURE_PROJECT_ENDPOINT=...
  BING_CONNECTION_NAME=...   # Name of your Azure AI Project connection to Bing
  AZURE_ENDPOINT=...         # (Not used directly here but common in your project)
  MODEL_API_VERSION=...      # (Optional; not used directly here)
  MODEL_DEPLOYMENT_NAME=...  # (Optional; not used directly here)

Usage
-----
python newsagent.py --ticker AAPL
python newsagent.py --ticker INFY --freshness Month --count 30 --market en-US

Notes
-----
- The agent will *only* summarize news and include citations. No forecasts/opinions.
- We use a one-participant RoundRobinGroupChat for convenience.
"""

import os
import sys
import asyncio
import argparse
from dotenv import load_dotenv

# Azure AI Studio Bing tool + project access
from azure.identity.aio import DefaultAzureCredential
from azure.ai.projects.aio import AIProjectClient
from azure.ai.agents.models import BingGroundingTool

# Autogen AgentChat wrappers
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.agents.azure._azure_ai_agent import AzureAIAgent


def build_arg_parser():
    p = argparse.ArgumentParser(description="Fetch recent news via Azure Bing Grounding Tool for a given stock ticker.")
    p.add_argument("--ticker", required=True, help="Stock symbol, e.g., AAPL, MSFT, INFY")
    p.add_argument("--freshness", default="Month", choices=["Day", "Week", "Month"],
                   help="Time range filter for Bing results (default: Month)")
    p.add_argument("--count", type=int, default=50, help="Max number of results to retrieve (default: 50)")
    p.add_argument("--market", default="en-US", help="Bing market locale, e.g., en-US")
    p.add_argument("--deployment", default="gpt-4o", help="Azure OpenAI deployment name to use for the agent")
    return p


async def make_news_agent(ticker: str, freshness: str, count: int, market: str, deployment: str) -> AzureAIAgent:
    # Load environment (endpoint + connection name for Bing)
    load_dotenv()
    project_endpoint = os.getenv("AZURE_PROJECT_ENDPOINT")
    if not project_endpoint:
        raise RuntimeError("Missing AZURE_PROJECT_ENDPOINT in environment/.env")

    bing_conn_name = os.getenv("BING_CONNECTION_NAME")
    if not bing_conn_name:
        raise RuntimeError("Missing BING_CONNECTION_NAME in environment/.env")

    # Create Azure AI Project client with default credentials
    credential = DefaultAzureCredential()
    project_client = AIProjectClient(credential=credential, endpoint=project_endpoint)

    # Resolve the Bing connection and instantiate the tool
    conn = await project_client.connections.get(name=bing_conn_name)
    bing_tool = BingGroundingTool(
        connection_id=conn.id,
        count=count,
        freshness=freshness,
        market=market,
    )

    system_instructions = (
        f"You are NewsAnalyzer. Search for the **latest and most impactful** news about {ticker} "
        f"from the selected freshness window. Focus on items that could move the stock: earnings, "
        f"analyst rating changes, major partnerships, M&A, legal/regulatory events, sector/macro shifts. "
        f"Summarize concisely with bullet points and include the source in parentheses after each bullet. "
        f"Do **not** provide forecasts or opinions. When finished, end your response with the word: DONE"
    )

    agent = AzureAIAgent(
        name="NewsAnalyzer",
        description=f"Summarizes recent {ticker}-related news with citations using Bing search.",
        project_client=project_client,
        deployment_name=deployment,
        instructions=system_instructions,
        tools=bing_tool.definitions,
        metadata={"source": "AzureAIAgent"},
    )
    return agent


async def run_news_only(ticker: str, freshness: str, count: int, market: str, deployment: str):
    news_agent = await make_news_agent(ticker, freshness, count, market, deployment)

    # Single-agent "team" with termination on "DONE"
    termination = TextMentionTermination("DONE")
    team = RoundRobinGroupChat(
        participants=[news_agent],
        termination_condition=termination,
        # Optional: set a reasonable max turns fallback to avoid infinite loops
        max_turns=8,
    )

    # Clear, bounded user task
    user_task = TextMessage(
        content=(
            f"Find and summarize the most impactful news about {ticker} in the past {freshness}. "
            f"Use the Bing tool to find sources. Provide a concise bullet list; "
            f"each bullet must end with a citation in parentheses. Finish with 'DONE'."
        ),
        source="user",
    )

    result = await team.run(task=user_task)

    # Print all messages in order (agent messages usually include the content we want)
    print("\n===== NewsAnalyzer Output =====\n")
    for msg in result.messages:
        role = msg.source or "unknown"
        # Only print non-empty content
        if getattr(msg, "content", None):
            print(f"{role}:\n{msg.content}\n")

    print("===== End =====\n")


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        asyncio.run(run_news_only(
            ticker=args.ticker.strip().upper(),
            freshness=args.freshness,
            count=args.count,
            market=args.market,
            deployment=args.deployment,
        ))
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
