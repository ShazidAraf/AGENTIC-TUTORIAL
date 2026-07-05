# memory_agent.py -- short-term trimming + long-term vector memory
# Topic 4: Memory -- Short-Term vs. Long-Term
# Deps:  pip install chromadb sentence-transformers
# Run:   conda activate agentic && python memory_agent.py

# Step 0: Imports.
#   - `anthropic` -> the SDK.
#   - `chromadb`  -> a local vector database for long-term (semantic) memory.
import anthropic, chromadb

# Step 1: Create the API client (reads ANTHROPIC_API_KEY from the environment).
client = anthropic.Anthropic()

# Step 2: Create an in-memory vector store + a collection to hold user "memories".
#   Chroma embeds each stored string into a vector so we can later search by MEANING.
db = chromadb.Client()
mem = db.create_collection("user_memory")   # uses a default embedder


# Step 3: WRITE PATH -- store one durable fact as a memory.
def remember(fact: str, i: int) -> None:
    mem.add(documents=[fact], ids=[f"m{i}"])   # id must be unique per memory


# Step 4: READ PATH -- semantic search for the k memories most relevant to `query`.
def recall(query: str, k: int = 2) -> list[str]:
    res = mem.query(query_texts=[query], n_results=k)
    return res["documents"][0] if res["documents"] else []


# Step 5: SHORT-TERM window -- how many recent messages we keep in the prompt.
MAX_TURNS = 6   # long chats get trimmed so context stays small + cheap


def chat(history: list, user_msg: str) -> str:
    # Step 6: READ long-term memory relevant to THIS message, inject into the system prompt.
    facts = recall(user_msg)
    print(f"[STEP 6] recalled facts for '{user_msg[:40]}...': {facts}")
    system = ("You are a helpful assistant.\n"
              "Known facts about the user:\n- " + "\n- ".join(facts)) if facts \
             else "You are a helpful assistant."

    # Step 7: Append the new user message to short-term history.
    history.append({"role": "user", "content": user_msg})

    # Step 8: SHORT-TERM management -- keep only the last MAX_TURNS messages.
    trimmed = history[-MAX_TURNS:]
    print(f"[STEP 8] sending {len(trimmed)} of {len(history)} messages (trimmed to last {MAX_TURNS})")

    # Step 9: Call the model with the injected facts (system) + recent turns (messages).
    resp = client.messages.create(model="claude-haiku-4-5", max_tokens=400,
                                  system=system, messages=trimmed)
    answer = resp.content[0].text

    # Step 10: Record the assistant's reply in short-term history and return it.
    history.append({"role": "assistant", "content": answer})
    return answer


# Step 11: Entry point -- seed some long-term facts, then chat.
if __name__ == "__main__":
    remember("User's name is Shazid.", 1)
    remember("User prefers concise, bullet-point answers.", 2)
    h = []
    print("\n--- turn 1 ---")
    print(chat(h, "Plan me a 3-day study schedule for LangGraph."))
    print("\n--- turn 2 (answered from long-term memory) ---")
    print(chat(h, "What's my name, by the way?"))
