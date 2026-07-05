# multi_agent.py -- supervisor/worker pattern via nested LLM calls
# Topic 8: Multi-Agent Orchestration
# Run:  conda activate agentic && python multi_agent.py

# Step 0: Imports.
import anthropic

# Step 1: Create the API client (reads ANTHROPIC_API_KEY from the environment).
client = anthropic.Anthropic()


# Step 2: A generic WORKER -- a focused LLM call with a narrow role prompt.
#   Each worker has few responsibilities -> a narrow prompt beats one giant confused agent.
def worker(role_prompt: str, task: str) -> str:
    resp = client.messages.create(model="claude-haiku-4-5", max_tokens=500,
                                  system=role_prompt,
                                  messages=[{"role": "user", "content": task}])
    return resp.content[0].text


# Step 3: The specialist roles. Each is a different "system" personality.
WORKERS = {
    "researcher": "You are a research specialist. Return terse factual notes.",
    "analyst":    "You are a quantitative analyst. Show your arithmetic.",
    "writer":     "You are a writer. Turn notes into one polished paragraph.",
}

# Step 4: Expose EACH worker as a TOOL the supervisor can delegate to.
#   Delegation is just tool calling where the "tool" is another agent.
WORKER_TOOLS = [{
    "name": name,
    "description": f"Delegate a subtask to the {name}. {prompt}",
    "input_schema": {"type": "object",
                     "properties": {"task": {"type": "string"}},
                     "required": ["task"]},
} for name, prompt in WORKERS.items()]


def supervise(task: str, max_steps: int = 8) -> str:
    # Step 5: The SUPERVISOR seeds the conversation and is told to delegate then compose.
    messages = [{"role": "user", "content": task}]
    system = ("You are a supervisor. Delegate subtasks to worker tools, "
              "then compose the final answer yourself.")
    print(f"[STEP 5] TASK: {task}")

    # Step 6: Supervisor loop (same tool_use handshake as Topic 2).
    for i in range(max_steps):
        print(f"\n===== supervisor step {i+1} =====")
        resp = client.messages.create(model="claude-haiku-4-5", max_tokens=800,
                                      system=system, tools=WORKER_TOOLS, messages=messages)
        messages.append({"role": "assistant", "content": resp.content})

        # Step 6a: If the supervisor stops calling tools, it has composed the answer.
        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")

        # Step 6b: For each delegation, run the matching worker (a NESTED LLM call).
        results = []
        for b in resp.content:
            if b.type == "tool_use":
                print(f"[STEP 6b] supervisor -> {b.name}: {b.input['task'][:60]}...")
                out = worker(WORKERS[b.name], b.input["task"])   # nested worker call
                results.append({"type": "tool_result",
                                "tool_use_id": b.id, "content": out})

        # Step 6c: Return all worker results to the supervisor and loop.
        messages.append({"role": "user", "content": results})

    # Step 7: Loop budget exhausted.
    return "Budget exhausted."


# Step 8: Entry point.
if __name__ == "__main__":
    print("\n[STEP 8] FINAL:\n" + supervise(
        "Estimate the yearly cost of running a 7B model on one A100 at $2/hr, "
        "8h/day, and summarize for an exec."))
