# THE LOCATION OF THE ZIP DEPLOYED TO AZURE IS : C:\Users\prana\Desktop\slackListenerAzure\deploy

# ==========================================================
# azure_slacklistener.py
# FastAPI app to receive Slack Event Subscriptions on Azure
# Fetches parent thread message + reply context
# ==========================================================

from fastapi import FastAPI, Request, BackgroundTasks
import httpx
import asyncio
import os
import json

# ==========================================================
# Environment variables (configure in Azure)
# ==========================================================
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")  # xoxb- token from your Slack App
SLACK_API_BASE = "https://slack.com/api"

# ==========================================================
# Initialize FastAPI app
# ==========================================================
app = FastAPI(title="Azure Slack Listener", version="1.0")


# ==========================================================
# Root health check
# ==========================================================
@app.get("/")
async def root():
    """Simple health check for Azure deployment."""
    return {"status": "ok", "message": "Azure Slack listener is running."}


# ==========================================================
# Slack Events endpoint
# ==========================================================
@app.post("/slack/events")
async def slack_events(request: Request, background_tasks: BackgroundTasks):
    """
    Primary endpoint that Slack will POST to.
    Handles:
    - URL verification (challenge)
    - Message events (thread replies)
    """
    try:
        body = await request.json()
    except Exception as e:
        print(f"[ERROR] Failed to parse Slack request: {e}")
        return {"ok": False}

    print(f"[SLACK EVENT RECEIVED]\n{json.dumps(body, indent=2)}", flush=True)

    # 1️⃣ Slack's URL verification (first-time setup)
    if body.get("type") == "url_verification":
        challenge = body.get("challenge")
        print(f"[SLACK] URL verification challenge received: {challenge}")
        return {"challenge": challenge}

    # 2️⃣ Actual event payloads
    event = body.get("event", {})
    if not event:
        return {"ok": True}

    # Ignore bot messages to prevent loops
    if event.get("subtype") == "bot_message" or event.get("bot_id"):
        print("[INFO] Ignoring bot_message to prevent loops.")
        return {"ok": True}

    # Only handle threaded replies (messages with thread_ts)
    if event.get("type") == "message" and event.get("thread_ts"):
        user = event.get("user")
        reply_text = event.get("text")
        channel = event.get("channel")
        thread_ts = event.get("thread_ts")

        print(f"[SLACK MESSAGE] user={user}, text='{reply_text}', channel={channel}, thread_ts={thread_ts}")

        # Process in background (non-blocking)
        background_tasks.add_task(process_message_with_parent, user, reply_text, channel, thread_ts)

    return {"ok": True}


# ==========================================================
# Async background processor
# ==========================================================
async def process_message_with_parent(user: str, reply_text: str, channel: str, thread_ts: str):
    """
    Fetches the parent message using Slack API,
    builds a context payload, and triggers MCP / app.py logic.
    """
    parent_text = await fetch_parent_message(channel, thread_ts)
    print(f"[PARENT CONTEXT] Parent: '{parent_text}' | Reply: '{reply_text}'")

    payload = {
        "user": user,
        "channel": channel,
        "thread_ts": thread_ts,
        "parent_text": parent_text,
        "reply_text": reply_text,
    }

    # 🔜 Forward to your pipeline (GPT / MCP / Alpaca)
    # from app import process_investment_request
    # await process_investment_request(payload)

    # or use MCP runner:
    # from mcprunner import run_mcp_runner
    # await run_mcp_runner(payload)

    print(f"[PIPELINE] Forwarded payload:\n{json.dumps(payload, indent=2)}", flush=True)


# ==========================================================
# Helper: fetch parent message text
# ==========================================================
async def fetch_parent_message(channel: str, thread_ts: str) -> str:
    """Fetches the parent message for a given thread."""
    if not SLACK_BOT_TOKEN:
        print("[ERROR] Missing SLACK_BOT_TOKEN environment variable.")
        return ""

    headers = {
        "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    params = {"channel": channel, "ts": thread_ts, "limit": 1}

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{SLACK_API_BASE}/conversations.replies", params=params, headers=headers)
            data = resp.json()
            if not data.get("ok"):
                print(f"[ERROR] Slack API error: {data}")
                return ""
            messages = data.get("messages", [])
            if messages:
                parent_text = messages[0].get("text", "")
                return parent_text
        except Exception as e:
            print(f"[ERROR] Exception fetching parent message: {e}")
    return ""
