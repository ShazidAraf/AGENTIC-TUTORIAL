# mcp_client.py -- connect to the server, list tools, call one
# Topic 9: MCP -- Model Context Protocol (the CLIENT half)
# Deps:  pip install "mcp[cli]"
# Run:   conda activate agentic && python mcp_client.py
#        (this launches mcp_server.py as a subprocess automatically)

# Step 0: Imports. MCP clients are async, so we use asyncio.
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    # Step 1: Describe HOW to start the server -- run `python mcp_server.py`.
    params = StdioServerParameters(command="python", args=["mcp_server.py"])

    # Step 2: Launch the server subprocess and open stdio pipes to it.
    async with stdio_client(params) as (read, write):

        # Step 3: Open an MCP session over those pipes.
        async with ClientSession(read, write) as session:

            # Step 4: Do the MCP handshake (version/capability negotiation).
            await session.initialize()

            # Step 5: RUNTIME DISCOVERY -- ask the server what tools it has.
            #   Note: we never wrote schemas by hand; the client discovers them.
            tools = await session.list_tools()
            print("[STEP 5] Discovered tools:", [t.name for t in tools.tools])

            # Step 6: Call one of the discovered tools by name, with arguments.
            result = await session.call_tool("define", {"term": "mcp"})
            print("[STEP 6] define('mcp') ->", result.content[0].text)

            # Step 7: Call the other tool to show typed args flowing through.
            result2 = await session.call_tool("add", {"a": 17, "b": 23})
            print("[STEP 7] add(17, 23)   ->", result2.content[0].text)

    # NOTE: to plug these into an agent, convert each tool's inputSchema into the
    # tools=[...] list from Topic 2 and route tool_use blocks to session.call_tool(...).
    # That one adapter is the whole integration.


# Step 8: Run the async entry point.
if __name__ == "__main__":
    asyncio.run(main())
