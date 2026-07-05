# agentic_rag.py -- retrieval as a tool + relevance grading + retry
# Topic 5: Agentic RAG
# Deps:  pip install chromadb
# Run:   conda activate agentic && python agentic_rag.py

# Step 0: Imports.
import anthropic, chromadb

# Step 1: Create the API client (reads ANTHROPIC_API_KEY from the environment).
client = anthropic.Anthropic()

# Step 2: Build a tiny knowledge base (vector store) with 3 documents.
db = chromadb.Client()
kb = db.create_collection("knowledge_base")   # name must be >=3 chars (chromadb rule)
kb.add(
    ids=["d1", "d2", "d3"],
    documents=[
        "LoRA freezes base weights and trains low-rank adapter matrices A and B.",
        "QLoRA quantizes the frozen base model to 4-bit NF4 and trains LoRA adapters on top.",
        "DPO optimizes a policy directly from preference pairs without a reward model.",
    ],
)

# Step 3: Expose RETRIEVAL AS A TOOL.
#   The difference from classic RAG: the agent DECIDES when to search, can rewrite the
#   query, and can search again if results look irrelevant. Retrieval is just a tool now.
SEARCH_TOOL = {
    "name": "search_kb",
    "description": ("Search the ML knowledge base. Rewrite the user's question into a "
                    "short keyword query first. Call again with a DIFFERENT query if "
                    "results look irrelevant."),
    "input_schema": {"type": "object",
                     "properties": {"query": {"type": "string"}},
                     "required": ["query"]},
}

# Step 4: System prompt enforces GROUNDING -- answer only from results, cite doc ids,
#   and say "I don't know" if nothing relevant is found after a few searches.
SYSTEM = ("Answer ONLY from search results. If after 3 searches nothing relevant "
          "is found, say you don't know. Cite the doc ids you used.")


def ask(question: str) -> str:
    # Step 5: Seed the conversation with the user's question.
    messages = [{"role": "user", "content": question}]
    print(f"[STEP 5] QUESTION: {question}")

    # Step 6: The agent loop (same tool_use handshake as Topic 2).
    for i in range(6):
        print(f"\n===== loop iteration {i+1} =====")
        resp = client.messages.create(model="claude-haiku-4-5", max_tokens=600,
                                      system=SYSTEM, tools=[SEARCH_TOOL],
                                      messages=messages)
        print(f"[STEP 6] stop_reason = {resp.stop_reason}")

        # Step 6a: Preserve the assistant turn exactly (keeps tool_use ids intact).
        messages.append({"role": "assistant", "content": resp.content})

        # Step 6b: If the model didn't ask for a tool, it's answering -> return the text.
        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")

        # Step 6c: Run each requested search against the vector DB, pack results with ids.
        results = []
        for b in resp.content:
            if b.type == "tool_use":
                print(f"[STEP 6c] agent searched: {b.input['query']!r}")
                hits = kb.query(query_texts=[b.input["query"]], n_results=2)
                packed = "\n".join(f"[{i}] {d}" for i, d in
                                   zip(hits["ids"][0], hits["documents"][0]))
                print(f"[STEP 6c]   -> hits: {hits['ids'][0]}")
                results.append({"type": "tool_result", "tool_use_id": b.id,
                                "content": packed or "NO RESULTS"})

        # Step 6d: Send all search results back and loop (model may search again or answer).
        messages.append({"role": "user", "content": results})

    # Step 7: Ran out of loop budget.
    return "Budget exhausted."


# Step 8: Entry point.
if __name__ == "__main__":
    print(f"\n[STEP 8] ANSWER:\n{ask('How does QLoRA differ from plain LoRA?')}")
