# langgraph_agent.py -- the Topic-1 loop, rebuilt as an explicit graph
# Topic 10: LangGraph -- Agents as State Machines
# Deps:  pip install langgraph langchain langchain-anthropic
# Run:   conda activate agentic && python langgraph_agent.py

# Step 0: Imports.
from typing import Annotated, TypedDict
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver


# Step 1: Define tools with the @tool decorator (schema comes from the signature/docstring).
@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"{city}: 31 C, sunny"


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression like '17*23'."""
    return str(eval(expression, {"__builtins__": {}}))   # demo only!


tools = [get_weather, calculator]

# Step 2: Bind the tools to the model so it knows it may call them.
llm = ChatAnthropic(model="claude-haiku-4-5").bind_tools(tools)


# Step 3: Define the STATE that flows through the graph.
#   add_messages is a REDUCER: node outputs are APPENDED to `messages`, not overwritten.
#   Forgetting this reducer is the classic LangGraph beginner bug.
class State(TypedDict):
    messages: Annotated[list, add_messages]


# Step 4: The "agent" NODE -- one LLM call that reads state and returns new messages.
def agent_node(state: State) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


# Step 5: The conditional EDGE -- the branching logic (Topic 1's `if stop_reason...`)
#   is now a routing function: go to "tools" if the model asked for tools, else END.
def route(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


# Step 6: Assemble the graph: nodes + edges.
g = StateGraph(State)
g.add_node("agent", agent_node)
g.add_node("tools", ToolNode(tools))          # prebuilt node that executes tool calls
g.add_edge(START, "agent")                    # start -> agent
g.add_conditional_edges("agent", route, {"tools": "tools", END: END})  # branch
g.add_edge("tools", "agent")                  # after tools, loop back to agent

# Step 7: Compile with a checkpointer -> persistence / resume for free.
app = g.compile(checkpointer=MemorySaver())


# Step 8: Entry point -- invoke the graph, then reuse the SAME thread_id to resume state.
if __name__ == "__main__":
    cfg = {"configurable": {"thread_id": "demo-1"}}   # conversation id (keys the checkpoint)

    out = app.invoke({"messages": [("user",
        "Weather in Riverside, and what is 250*4?")]}, cfg)
    print("[STEP 8] answer 1:", out["messages"][-1].content)

    # Same thread_id -> the checkpointer restores the prior messages automatically.
    out2 = app.invoke({"messages": [("user",
        "Repeat just the temperature you found.")]}, cfg)
    print("[STEP 8] answer 2:", out2["messages"][-1].content)
