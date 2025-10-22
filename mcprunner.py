# mcprunner.py — Final version sending "payload" field (Go-compatible)
import os
import asyncio
import json
import time
from pathlib import Path
from dotenv import load_dotenv

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient


# ==========================================================
#  Timestamped Logger
# ==========================================================
def log(message: str):
    """Print timestamped log messages (flushes immediately)."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ==========================================================
#  Load Environment & Azure Config
# ==========================================================
log("INIT: Loading environment variables...")
load_dotenv()
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("MODEL_DEPLOYMENT_NAME", "gpt-4o")
AZURE_API_VERSION = os.getenv("MODEL_API_VERSION")

if not all([AZURE_ENDPOINT, AZURE_API_VERSION]):
    raise RuntimeError("❌ Missing Azure credentials in environment! Check .env file.")


# ==========================================================
#  MCP Client (Slack subprocess bridge)
# ==========================================================
class MCPStdIOClient:
    def __init__(self, registry_path: str):
        self.registry_path = Path(registry_path)
        self._sessions = {}

    def _load_conf(self, server_name: str):
        data = json.loads(self.registry_path.read_text(encoding="utf-8"))
        servers = data.get("mcpServers") or data.get("mcp_servers")
        if not servers or server_name not in servers:
            raise ValueError(f"Server '{server_name}' not found in {self.registry_path}")
        return servers[server_name]

    async def ensure_started(self, server_name: str):
        if server_name in self._sessions:
            return self._sessions[server_name][0]
        conf = self._load_conf(server_name)
        params = StdioServerParameters(
            command=conf["command"],
            args=conf.get("args", []),
            cwd=conf.get("workingDirectory"),
            env={**os.environ, **conf.get("env", {})},
        )
        log(f"MCP: Launching {server_name}: {params.command} {' '.join(params.args or [])}")
        cm = stdio_client(params)
        agen = cm.__aenter__()
        read, write = await agen
        session = ClientSession(read, write)
        await session.__aenter__()
        await session.initialize()
        self._sessions[server_name] = (session, cm)
        log(f"MCP: ✅ {server_name} initialized successfully.")
        return session

    async def call(self, server_name: str, tool_name: str, arguments: dict):
        # DEBUG: show what JSON is being sent to Go
        try:
            debug_json = json.dumps({
                "method": "tools/call",
                "params": {
                    "tool_name": tool_name,
                    "arguments": arguments
                }
            }, ensure_ascii=False, indent=2)
            print("\n========== PYTHON → GO (Raw JSON to be sent) ==========")
            print(debug_json)
            print("========================================================\n", flush=True)
        except Exception as e:
            print(f"[DEBUG ERROR] Could not serialize arguments for logging: {e}", flush=True)

        # Normal MCP call
        log(f"MCP: Calling tool '{tool_name}' with {arguments}")
        session, _ = self._sessions.get(server_name, (None, None))
        if not session:
            session = await self.ensure_started(server_name)
        result = await session.call_tool(tool_name, arguments)
        log(f"MCP: Tool '{tool_name}' completed.")
        content = getattr(result, "content", None)
        if isinstance(content, list) and content:
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    return item.get("text")
        return str(result)

    async def close(self):
        for name, (session, cm) in list(self._sessions.items()):
            await session.__aexit__(None, None, None)
            await cm.__aexit__(None, None, None)
            log(f"MCP: Closed session for {name}.")
        self._sessions.clear()
        log("MCP: All sessions closed.")


# ==========================================================
#  Slack Agent (strict JSON output using "payload")
# ==========================================================
def make_slack_agent(model_client: AzureOpenAIChatCompletionClient) -> AssistantAgent:
    """Slack automation agent that emits Go-compatible JSON payloads."""
    return AssistantAgent(
        name="SlackAgent",
        model_client=model_client,
        system_message=(
            "You are a Slack automation agent connected to a Slack MCP server.\n"
            "Return one valid JSON object only (no prose, no explanations).\n"
            "Schema:\n"
            "{\n"
            '  \"tool\": \"conversations_add_message\",\n'
            '  \"args\": {\"channel_id\": str, \"payload\": str}\n'
            "}\n"
            "Example:\n"
            "User: post hello in channel C09D8UXSDLL\n"
            "Assistant:\n"
            "{\n"
            '  \"tool\": \"conversations_add_message\",\n'
            '  \"args\": {\"channel_id\": \"C09D8UXSDLL\", \"payload\": \"hello\"}\n'
            "}\n"
            "Rules:\n"
            "- Output must be valid JSON parsable by json.loads().\n"
            "- Do not escape quotes or wrap strings twice.\n"
            "- Do not include the prefix 'MCP:' or any explanatory text."
        ),
    )


# ==========================================================
#  Azure + Slack Agent Runner
# ==========================================================
REGISTRY_PATH = r"C:\mcpCnfig\slack.json"


async def run_mcp_runner():
    log("INIT: Setting up Azure OpenAI client...")
    model_client = AzureOpenAIChatCompletionClient(
        azure_endpoint=AZURE_ENDPOINT,
        azure_deployment=AZURE_DEPLOYMENT,
        model="gpt-4o-2024-11-20",
        api_version=AZURE_API_VERSION,
    )

    # ---- Connectivity test ----
    try:
        log("TEST: Checking Azure GPT connectivity using team.run() ...")
        test_agent = AssistantAgent(
            name="TestAgent",
            model_client=model_client,
            system_message="You are a simple echo bot. Reply exactly: hello world.",
        )
        termination = TextMentionTermination("hello world")
        test_team = RoundRobinGroupChat(participants=[test_agent], termination_condition=termination)
        test_task = TextMessage(content="Say hello world only.", source="user")
        await asyncio.wait_for(test_team.run(task=test_task), timeout=15)
        log("✅ Azure GPT connectivity confirmed.")
    except Exception as e:
        log(f"❌ Azure connection test failed: {e}")
        return

    # ---- Start Slack MCP ----
    mcp = MCPStdIOClient(REGISTRY_PATH)
    await mcp.ensure_started("slack")
    log("MCP: Slack server started successfully.\n")

    # ---- Create SlackAgent ----
    slack_agent = make_slack_agent(model_client)
    termination = TextMentionTermination("}")  # end of JSON
    team = RoundRobinGroupChat(participants=[slack_agent], termination_condition=termination)

    user_message = TextMessage(
        content="Post 'Hello from Pune via MCP' in channel C09D8UXSDLL",
        source="user",
    )

    log("TEAM: Running team.run() — waiting for GPT-4o response...")
    try:
        result = await asyncio.wait_for(team.run(task=user_message), timeout=30)
        print("\n===================== TEAM MESSAGES =====================")
        if hasattr(result, "messages"):
            for msg in result.messages:
                print(f"{getattr(msg, 'source', 'unknown')}: {getattr(msg, 'content', msg)}")
        else:
            print(result)
        print("==========================================================\n")
        log("TEAM: ✅ team.run() finished.")
    except asyncio.TimeoutError:
        log("⚠ TEAM run timed out — no GPT reply.")
        await mcp.close()
        return
    except Exception as e:
        log(f"❌ TEAM ERROR during run(): {e}")
        await mcp.close()
        return

    # ---- Parse JSON output ----
    slack_reply = None
    if hasattr(result, "messages") and result.messages:
        for m in reversed(result.messages):
            if getattr(m, "source", "") == "SlackAgent":
                slack_reply = getattr(m, "content", None)
                break

    if slack_reply:
        try:
            data = json.loads(slack_reply)
            tool_name = data["tool"]
            args = data["args"]
            log(f"PARSER: Parsed -> tool={tool_name}, args={args}")

            resp = await mcp.call("slack", tool_name, args)
            log(f"MCP RESPONSE: {resp}")
        except Exception as e:
            log(f"❌ Failed to parse or execute SlackAgent output: {e}\nRaw: {slack_reply}")
    else:
        log("TEAM: No SlackAgent message detected.")

    await mcp.close()
    log("CLEANUP: Runner completed.")


# ==========================================================
#  Entry Point
# ==========================================================
if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_mcp_runner())
