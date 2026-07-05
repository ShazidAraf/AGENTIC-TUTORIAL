# plan_and_execute.py -- explicit plan, then step-by-step execution
# Topic 6: Planning -- Plan-and-Execute & Task Decomposition
# Run:  conda activate agentic && python plan_and_execute.py

# Step 0: Imports.
import anthropic
from pydantic import BaseModel

# Step 1: Create the API client (reads ANTHROPIC_API_KEY from the environment).
client = anthropic.Anthropic()


# Step 2: The plan's SHAPE -- an ordered list of small, self-contained step strings.
#   (Structured output from Topic 3 is reused here to get a clean list back.)
class Plan(BaseModel):
    steps: list[str]


# Step 3: Expose the plan shape as a forced tool (same schema-as-tool trick as Topic 3).
PLAN_TOOL = {"name": "submit_plan",
             "description": "Submit the step-by-step plan.",
             "input_schema": Plan.model_json_schema()}


def make_plan(task: str) -> list[str]:
    # Step 4: PLANNER call -- decompose the task into 3-5 ordered steps.
    resp = client.messages.create(
        model="claude-haiku-4-5", max_tokens=500,
        tools=[PLAN_TOOL], tool_choice={"type": "tool", "name": "submit_plan"},  # force it
        messages=[{"role": "user", "content":
                   f"Break this task into 3-5 minimal ordered steps:\n{task}"}])
    args = next(b.input for b in resp.content if b.type == "tool_use")
    return Plan(**args).steps


def execute_step(step: str, scratchpad: str) -> str:
    # Step 5: EXECUTOR call -- complete ONE step, given the results-so-far scratchpad.
    #   Small, focused calls: cheaper and easier to debug than one giant prompt.
    resp = client.messages.create(
        model="claude-haiku-4-5", max_tokens=600,
        messages=[{"role": "user", "content":
                   f"Results so far:\n{scratchpad or '(none)'}\n\n"
                   f"Complete ONLY this step and report the result:\n{step}"}])
    return resp.content[0].text


def run(task: str) -> str:
    # Step 6: Make the plan up front (this is what makes the agent auditable/steerable).
    steps = make_plan(task)
    print("[STEP 6] PLAN:", *[f"\n   {i+1}. {s}" for i, s in enumerate(steps)])

    # Step 7: Execute steps one by one, accumulating results into a scratchpad.
    scratchpad = ""
    for i, step in enumerate(steps, 1):
        result = execute_step(step, scratchpad)
        scratchpad += f"\nStep {i} ({step}):\n{result}\n"
        print(f"\n[STEP 7] == step {i} done ==")

    # Step 8: Final synthesis pass -- turn the scratchpad into the user-facing answer.
    return execute_step("Write the final answer for the user.", scratchpad)


# Step 9: Entry point.
if __name__ == "__main__":
    print("\n[STEP 9] FINAL:\n" + run(
        "Compare LoRA and QLoRA for fine-tuning a 7B model on one GPU, "
        "and recommend one with a short justification."))
