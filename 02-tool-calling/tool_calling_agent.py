# tool_calling_agent.py -- native tool use (the robust version of Topic 1)
# Topic 2: Native Tool / Function Calling
# Run:  conda activate agentic && python tool_calling_agent.py

# Step 0: Imports.
#   - `json`      -> to build/return structured data as JSON strings.
#   - `anthropic` -> the official Anthropic SDK.
import json, anthropic

# Step 1: Create the API client (reads ANTHROPIC_API_KEY from the environment).
client = anthropic.Anthropic()


# Step 2: Define the tool functions as ORDINARY Python.
#   Same idea as Topic 1: the model never runs these -- our code does, when asked.
def calculator(expression: str) -> str:
    # eval() is UNSAFE -> demo only. In production you'd sandbox this.
    return str(eval(expression, {"__builtins__": {}}))

def get_weather(city: str) -> str:
    # Return JSON so the model gets clean, structured data back (not just prose).
    return json.dumps({"city": city, "temp_c": 31, "sky": "sunny"})

# Step 3: A REGISTRY mapping tool name -> the Python function to run.
#   When the model asks for "calculator", we look it up here and call it.
REGISTRY = {"calculator": calculator, "get_weather": get_weather}

# Step 4: Describe each tool to the model as a JSON SCHEMA.
#   THIS is the big difference from Topic 1: instead of describing tools in prose and
#   parsing text with regex, we hand the API a machine-readable schema. The model then
#   returns a structured `tool_use` block (real JSON), so no regex is ever needed.
#   The "description" is prompt engineering -- it's how the model decides WHEN to call.
TOOL_SCHEMAS = [
    {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression, e.g. '17*23'. Use for ANY math.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},  # one input: expression (a string)
            "required": ["expression"],                          # it's mandatory
        },
    },
    {
        "name": "get_weather",
        "description": "Get current weather for a city name.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},          # one input: city (a string)
            "required": ["city"],
        },
    },
]


def run(task: str, max_steps: int = 8) -> str:
    # Step 5: Seed the conversation with the user's task (same as Topic 1).
    messages = [{"role": "user", "content": task}]
    print(f"\n[STEP 5] TASK: {task}")

    # Step 6: The agent loop, capped at max_steps so it can't run forever.
    for i in range(max_steps):
        print(f"\n===== loop iteration {i+1} =====")

        # Step 6a: Call the model, passing the tool schemas via `tools=`.
        #   Now the model can answer in plain text OR emit structured tool_use blocks.
        resp = client.messages.create(
            model="claude-haiku-4-5", max_tokens=600,   # cheapest current model
            tools=TOOL_SCHEMAS, messages=messages,
        )
        # Print WHY the model stopped, and the block types it returned, so we can see
        # exactly what came back (text? one tool_use? several?).
        print(f"[STEP 6a] stop_reason = {resp.stop_reason}")
        print(f"[STEP 6a] block types  = {[b.type for b in resp.content]}")

        # Step 6b: Preserve the assistant's turn EXACTLY as returned.
        #   resp.content is a list of blocks -- it may mix text AND tool_use blocks.
        messages.append({"role": "assistant", "content": resp.content})

        # Step 6c: TERMINATION CHECK. `stop_reason` tells us WHY the model stopped.
        #   If it's anything other than "tool_use", the model is done -> return its text.
        if resp.stop_reason != "tool_use":
            answer = "".join(b.text for b in resp.content if b.type == "text")
            print(f"[STEP 6c] no more tools -> returning final answer")
            return answer

        # Step 6d: The model wants tools. Loop over ALL blocks (could be several!).
        results = []
        for block in resp.content:
            if block.type == "tool_use":                 # only act on tool_use blocks
                print(f"[STEP 6d] model asked for tool: {block.name}({block.input})")
                fn = REGISTRY[block.name]                 # find the matching Python function
                out = fn(**block.input)                   # block.input is already real JSON (a dict)
                print(f"[STEP 6d]   -> our code ran it, result = {out}")
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,              # MUST echo the block's id, or API rejects it
                    "content": out,                       # the tool's output, sent back to the model
                })

        # Step 6e: Send ALL tool results back in a SINGLE user message, then loop again.
        print(f"[STEP 6e] sending {len(results)} tool result(s) back to the model")
        messages.append({"role": "user", "content": results})

    # Step 7: Ran out of steps without a final answer.
    return "Step budget exhausted."


# Step 8: Entry point -- run one example when the file is executed directly.
if __name__ == "__main__":
    final = run("Weather in Riverside? Also compute (250-17)*4.")
    print(f"\n[STEP 8] FINAL ANSWER:\n{final}")
