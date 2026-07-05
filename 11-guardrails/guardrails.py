# guardrails.py -- allow-list + budget + human approval wrapper
# Topic 11: Guardrails & Human-in-the-Loop
# Run:  conda activate agentic && python guardrails.py
#   NOTE: this example is INTERACTIVE -- it will ask you to approve the email at the prompt.

# Step 0: Imports.
import anthropic

# Step 1: Create the API client (reads ANTHROPIC_API_KEY from the environment).
client = anthropic.Anthropic()


# Step 2: The real tool functions (one has a pretend side effect).
def send_email(to: str, body: str) -> str:
    return f"EMAIL SENT to {to}"      # a real, irreversible-ish action

def read_notes(topic: str) -> str:
    return f"Notes about {topic}: ..."   # a safe, read-only action

# Step 3: Guardrail configuration.
REGISTRY = {"send_email": send_email, "read_notes": read_notes}
RISKY = {"send_email"}                 # ACTION GUARD: which tools need human approval
MAX_TOOL_CALLS = 5                     # BUDGET GUARD: cap total tool calls
BANNED_INPUT = ("ignore previous instructions", "system prompt")  # INPUT GUARD phrases


# Step 4: INPUT GUARD -- screen the request BEFORE it ever reaches the LLM.
def input_guard(text: str) -> None:
    if any(p in text.lower() for p in BANNED_INPUT):
        raise PermissionError("Blocked: possible prompt injection.")


# Step 5: The guarded executor -- every tool call passes through these deterministic checks.
def guarded_execute(name: str, args: dict, calls_so_far: int) -> str:
    # 5a: ALLOW-LIST -- refuse tools we didn't explicitly permit.
    if name not in REGISTRY:
        return "DENIED: tool not on allow-list."
    # 5b: BUDGET -- refuse once we exceed the call budget.
    if calls_so_far >= MAX_TOOL_CALLS:
        return "DENIED: tool budget exhausted."
    # 5c: HUMAN-IN-THE-LOOP -- pause for approval on risky/irreversible actions.
    if name in RISKY:
        print(f"\n[APPROVAL NEEDED] {name}({args})")
        if input("Approve? [y/N] ").strip().lower() != "y":
            return "DENIED by human reviewer."
    # 5d: Passed all guards -> actually run the tool.
    return REGISTRY[name](**args)


# Step 6: Tool schemas (same shape as Topic 2).
TOOLS = [
    {"name": "read_notes", "description": "Read study notes on a topic.",
     "input_schema": {"type": "object",
                      "properties": {"topic": {"type": "string"}},
                      "required": ["topic"]}},
    {"name": "send_email", "description": "Send an email.",
     "input_schema": {"type": "object",
                      "properties": {"to": {"type": "string"},
                                     "body": {"type": "string"}},
                      "required": ["to", "body"]}},
]


def run(task: str) -> str:
    # Step 7: Run the INPUT GUARD first, before anything else.
    input_guard(task)

    messages, n_calls = [{"role": "user", "content": task}], 0

    # Step 8: The agent loop -- but every tool call is wrapped by guarded_execute.
    for _ in range(8):
        resp = client.messages.create(model="claude-haiku-4-5",
                                      max_tokens=600, tools=TOOLS, messages=messages)
        messages.append({"role": "assistant", "content": resp.content})

        # 8a: Done when the model stops calling tools.
        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")

        # 8b: Route every requested tool through the guardrails.
        results = []
        for b in resp.content:
            if b.type == "tool_use":
                out = guarded_execute(b.name, b.input, n_calls)   # <- the guard wrapper
                n_calls += 1
                results.append({"type": "tool_result",
                                "tool_use_id": b.id, "content": out})
        messages.append({"role": "user", "content": results})

    return "Budget exhausted."


# Step 9: Entry point -- note tool results are UNTRUSTED input; guards are deterministic.
if __name__ == "__main__":
    print("\n[STEP 9] FINAL:\n" + run(
        "Summarize my notes on MCP, then email the summary to boss@example.com."))
