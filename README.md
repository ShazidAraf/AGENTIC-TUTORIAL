# Agentic AI — Hands-On Tutorial

Working through a 12-topic, code-first guide to modern agentic AI using pure Python + the Anthropic SDK. Each topic is a self-contained, heavily-commented, runnable example.

## Setup

```bash
conda create -n agentic python=3.12
conda activate agentic
pip install anthropic pydantic
export ANTHROPIC_API_KEY="sk-ant-..."   # never hard-code the key in source
```

Extra dependencies are installed per topic as needed:

```bash
pip install chromadb sentence-transformers        # Topics 4-5 (memory / RAG)
pip install "mcp[cli]"                             # Topic 9 (MCP)
pip install langgraph langchain langchain-anthropic # Topic 10 (LangGraph)
```

All examples run on `claude-haiku-4-5` (cheapest current model).

## Topics

| # | Folder | Topic |
|---|--------|-------|
| 1 | `01-agent-loop/` | The Agent Loop (ReAct) |
| 2 | `02-tool-calling/` | Native Tool / Function Calling |
| 3 | `03-structured-outputs/` | Structured Outputs (validated JSON) |
| 4 | `04-memory/` | Memory — short-term vs. long-term |
| 5 | `05-agentic-rag/` | Agentic RAG |
| 6 | `06-planning/` | Plan-and-Execute |
| 7 | `07-reflection/` | Reflection & Self-Correction |
| 8 | `08-multi-agent/` | Multi-Agent Orchestration |
| 9 | `09-mcp/` | MCP — Model Context Protocol |
| 10 | `10-langgraph/` | LangGraph — agents as state machines |
| 11 | `11-guardrails/` | Guardrails & Human-in-the-Loop |
| 12 | `12-evaluation/` | Evaluating Agents (incl. LLM-as-Judge) |

## Run an example

```bash
conda activate agentic
python 01-agent-loop/react_agent.py
```
