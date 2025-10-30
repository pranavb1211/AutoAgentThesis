# ==========================================================
# alpacaMcpExecutor.py — FINAL VERSION
# Works with official Alpaca MCP Server (stdio transport)
# ==========================================================
# The corresponding MCP server must be defined in mcpServerList.json

    #   "alpaca": {
    #     "command": "C:\\Users\\prana\\.local\\bin\\uvx.exe",
    #     "args": ["alpaca-mcp-server", "serve"],
    #     "workingDirectory": "C:\\Users\\prana\\Desktop\\kello_notify\\alpaca-mcp-server",
    #     "env": {
    #       "ALPACA_API_KEY": " api key here ",
    #       "ALPACA_SECRET_KEY": "secret key here",
    #       "ALPACA_PAPER_TRADE": "True"
    #     }
    # }



import os
import asyncio
import json
import time
from pathlib import Path
from dotenv import load_dotenv

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters


# ==========================================================
#  Timestamped Logger
# ==========================================================
def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ==========================================================
#  Load Environment & Config
# ==========================================================
log("INIT: Loading environment variables...")
load_dotenv()
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

REGISTRY_PATH = r"C:\mcpCnfig\mcpServerList.json"


# ==========================================================
#  MCP Client (Reusable for stdio-based servers)
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
        """Ensure the MCP server is running and ready for tool calls."""
        if server_name in self._sessions:
            return self._sessions[server_name][0]

        conf = self._load_conf(server_name)
        params = StdioServerParameters(
            command=conf["command"],
            args=conf.get("args", []),
            cwd=conf.get("workingDirectory"),
            env={**os.environ, **conf.get("env", {})},
        )

        log(f"MCP: Launching {server_name}: {params.command} {' '.join(params.args)}")
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
        """Call a tool from the MCP server."""
        try:
            debug_json = json.dumps(
                {"method": "tools/call", "params": {"tool_name": tool_name, "arguments": arguments}},
                ensure_ascii=False, indent=2,
            )
            print("\n========== PYTHON → MCP (Raw JSON to be sent) ==========")
            print(debug_json)
            print("========================================================\n", flush=True)
        except Exception as e:
            log(f"[DEBUG ERROR] Could not serialize arguments: {e}")

        session, _ = self._sessions.get(server_name, (None, None))
        if not session:
            session = await self.ensure_started(server_name)

        log(f"MCP: Calling '{tool_name}' with {arguments}")
        result = await session.call_tool(tool_name, arguments)
        log(f"MCP: Tool '{tool_name}' completed.")

        # Extract text content
        content = getattr(result, "content", None)
        if isinstance(content, list) and content:
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    return item.get("text")
        return str(result)

    async def close(self):
        """Close all MCP sessions cleanly."""
        for name, (session, cm) in list(self._sessions.items()):
            await session.__aexit__(None, None, None)
            await cm.__aexit__(None, None, None)
            log(f"MCP: Closed session for {name}.")
        self._sessions.clear()
        log("MCP: All sessions closed.")


# ==========================================================
#  Helper: Execute Trade
# ==========================================================
async def execute_trade(symbol: str, qty: int, side: str = "buy"):
    """Place a stock order using Alpaca MCP."""
    log(f"TRADE: {side.upper()} {qty} {symbol}")
    mcp = MCPStdIOClient(REGISTRY_PATH)

    await mcp.ensure_started("alpaca")
    log("MCP: Alpaca MCP server connected.\n")

    order_args = {
        "symbol": symbol,
        "side": side,
        "quantity": qty,
        "order_type": "market",
        "time_in_force": "day"
    }

    try:
        result = await mcp.call("alpaca", "place_stock_order", order_args)
        print("\n[ORDER RESULT]\n", result, "\n")
    except Exception as e:
        log(f"❌ Trade failed: {e}")
    finally:
        await mcp.close()
        log("✅ Trade execution complete.\n")


# ==========================================================
#  Test Routine (account + positions + trade)
# ==========================================================
async def run_alpaca_executor():
    log("INIT: Starting Alpaca MCP Executor...")
    mcp = MCPStdIOClient(REGISTRY_PATH)

    await mcp.ensure_started("alpaca")
    log("MCP: Alpaca server started successfully.\n")

    # ---- Step 1: Fetch account info ----
    try:
        log("STEP 1: Fetching account info...")
        account = await mcp.call("alpaca", "get_account_info", {})
        print("\n[ACCOUNT INFO]\n", account, "\n")
    except Exception as e:
        log(f"❌ Error fetching account info: {e}")

    # ---- Step 2: Fetch positions ----
    try:
        log("STEP 2: Fetching open positions...")
        positions = await mcp.call("alpaca", "get_positions", {})
        print("\n[POSITIONS]\n", positions, "\n")
    except Exception as e:
        log(f"❌ Error fetching positions: {e}")

    # ---- Step 3: Test trade ----
    try:
        log("STEP 3: Placing test market order (AAPL, qty=1)...")
        await execute_trade("AAPL", 1, "buy")
    except Exception as e:
        log(f"❌ Error placing order: {e}")

    await mcp.close()
    log("CLEANUP: ✅ Alpaca MCP Executor finished.\n")


# ==========================================================
#  Entry Point
# ==========================================================
if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_alpaca_executor())
