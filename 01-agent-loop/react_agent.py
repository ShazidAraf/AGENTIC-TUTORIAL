1111# react_agent.py -- minimal ReAct loop, no framework
# Topic 1: The Agent Loop (ReAct = Reasoning + Acting)
# Run:  conda activate agentic && python react_agent.py

# Step 0: Imports.
#   - `re`        -> regular expressions, used to parse the model's "Action: tool[input]" text.
#   - `anthropic` -> the official Anthropic SDK that talks to the Claude API.
import re, anthropic

# Step 1: Create the API client.
#   The client automatically reads your key from the ANTHROPIC_API_KEY environment variable.
#   We never hard-code the key in source.
client = anthropic.Anthropic()

# Step 2: Define the TOOLS the agent is allowed to use.
#   This is just a dict mapping a tool name -> a plain Python function.
#   KEY INSIGHT: the LLM never runs this code. *Our* Python program runs it when the
#   model asks for it, then pastes the result back into the conversation.
TOOLS = {
    # calculator: evaluates an arithmetic string. eval() is UNSAFE -> demo only, never in production.
    "calculator": lambda expr: str(eval(expr, {"__builtins__": {}})),
    # get_weather: a fake weather API that always returns the same canned string.
    "get_weather": lambda city: f"{city}: 31 C, sunny",
}

# Step 3: Write the SYSTEM prompt.
#   In Topic 1 the tools live ONLY in the prompt text (no native tool calling yet).
#   We teach the model an EXACT output format so our regex can parse it reliably.
SYSTEM = """You solve tasks step by step. You may use tools.
Available tools:
- calculator[expression]   e.g. calculator[2*(3+4)]
- get_weather[city]        e.g. get_weather[Riverside]

Respond in EXACTLY this format:
Thought: <your reasoning>
Action: <tool_name>[<input>]

...or, when you know the answer:
Thought: <your reasoning>
Final: <answer to the user>"""


def run_agent(task: str, max_steps: int = 6) -> str:
    # Step 4: Seed the conversation with the user's task.
    #   `messages` is the growing conversation history sent on every API call.
    messages = [{"role": "user", "content": task}]

    # Step 5: The agent loop. We cap it at max_steps so an impossible task can't loop forever.
    for step in range(max_steps):

        # Step 5a: Ask the model what to do next, given the whole conversation so far.
        resp = client.messages.create(
            model="claude-haiku-4-5", max_tokens=400,  # cheapest current model (PDF used 4-5)
            system=SYSTEM, messages=messages,
        )
        print(f"MSG : {messages}")

        # Step 5b: Pull the plain text out of the model's response and show it for learning.
        text = resp.content[0].text
        print(f"--- step {step+1} ---\n{text}\n")

        # Step 5c: Record the model's turn in the history so the next call sees it.
        messages.append({"role": "assistant", "content": text})

        # Step 5d: TERMINATION CHECK. If the model wrote "Final:", we're done -> return the answer.
        if "Final:" in text:
            return text.split("Final:", 1)[1].strip()

        # Step 5e: Otherwise, try to PARSE an action of the form  Action: tool[input].
        #   NOTE: re.search finds only the FIRST match -> if the model requests two tools
        #   in one turn, we silently drop the second. That inefficiency is exactly what
        #   Topic 2 (native tool calling) fixes.
        m = re.search(r"Action:\s*(\w+)\[(.*?)\]", text, re.S)

        # Step 5f: Handle FORMAT DRIFT. If there's no Action and no Final, the model broke
        #   the format -> tell it explicitly and let it retry on the next loop iteration.
        if not m:
            messages.append({"role": "user",
                "content": "Format error. Use Action: tool[input] or Final:."})
            continue

        # Step 5g: Extract the tool name and its argument from the regex match.
        tool, arg = m.group(1), m.group(2)

        # Step 5h: EXECUTE the tool (if we know it) and capture the result as the "Observation".
        #   If the model named a tool we don't have, we return a helpful error string instead.
        obs = TOOLS[tool](arg) if tool in TOOLS else f"Unknown tool {tool}"

        # Step 5i: Feed the Observation back into the conversation so the model can react to it.
        messages.append({"role": "user", "content": f"Observation: {obs}"})

    # Step 6: If we exhausted the step budget without a Final answer, say so.
    return "Step budget exhausted."


# Step 7: Entry point -- run one example task when the file is executed directly.
if __name__ == "__main__":
    print(run_agent("What is the weather in Riverside, and what is 17*23?"))
