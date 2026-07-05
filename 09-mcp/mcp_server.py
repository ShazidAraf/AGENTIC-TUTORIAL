# mcp_server.py -- a tiny MCP server exposing two tools
# Topic 9: MCP -- Model Context Protocol (the SERVER half)
# Deps:  pip install "mcp[cli]"
# Run:   this file is launched automatically by mcp_client.py (as a subprocess)

# Step 0: Import FastMCP -- the quick way to stand up an MCP server.
from mcp.server.fastmcp import FastMCP

# Step 1: Create the server with a name. Clients see this name on connect.
mcp = FastMCP("study-tools")


# Step 2: Expose a tool with the @mcp.tool() decorator.
#   MCP builds the input schema from the type hints -- you don't hand-write JSON schema.
@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers."""      # <- the docstring becomes the tool description
    return a + b


# Step 3: A second tool -- a tiny glossary lookup.
@mcp.tool()
def define(term: str) -> str:
    """Define an agentic-AI term from a tiny local glossary."""
    glossary = {
        "react": "Reasoning + Acting loop: thought, action, observation.",
        "mcp": "Open protocol standardizing tool/context access for LLM apps.",
    }
    return glossary.get(term.lower(), "Unknown term.")


# Step 4: Run the server, speaking JSON-RPC over stdio (stdin/stdout).
#   "Build a tool server once, use it from any MCP client" -- USB-C for AI tools.
if __name__ == "__main__":
    mcp.run(transport="stdio")
