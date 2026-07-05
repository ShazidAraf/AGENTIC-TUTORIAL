# eval_harness.py -- rule checks + LLM-as-judge over a small eval set
# Topic 12: Evaluating Agents (incl. LLM-as-Judge)
# Run:  conda activate agentic && python eval_harness.py

# Step 0: Imports.
import json, anthropic

# Step 1: Create the API client (reads ANTHROPIC_API_KEY from the environment).
client = anthropic.Anthropic()


# Step 2: The SYSTEM UNDER TEST -- a plain one-shot agent we want to evaluate.
def agent(question: str) -> str:
    resp = client.messages.create(model="claude-haiku-4-5", max_tokens=400,
                                  messages=[{"role": "user", "content": question}])
    return resp.content[0].text


# Step 3: The EVAL SET -- tasks paired with expected properties.
#   Two kinds of scoring: a deterministic `check`, or a fuzzy `rubric` (LLM judge).
EVAL_SET = [
    {"q": "What is 17*23? Answer with just the number.",
     "check": lambda a: "391" in a},                       # deterministic scorer
    {"q": "In one sentence, what does LoRA do?",
     "rubric": "Must mention low-rank matrices/adapters and frozen base weights."},
    {"q": "Name the protocol standardizing LLM tool access.",
     "rubric": "Correct iff it names MCP / Model Context Protocol."},
]

# Step 4: The JUDGE's instructions -- return ONLY strict JSON so we can parse a verdict.
JUDGE_SYSTEM = ("You are a strict grader. Given QUESTION, RUBRIC, ANSWER, "
                'return ONLY JSON: {"pass": true/false, "reason": "..."}')


# Step 5: LLM-AS-JUDGE -- a (separate) model call grades an answer against a rubric.
#   Best practice: judge with a different/stronger model + spot-check against humans.
def llm_judge(q: str, rubric: str, answer: str) -> dict:
    resp = client.messages.create(model="claude-haiku-4-5", max_tokens=200,
                                  system=JUDGE_SYSTEM,
                                  messages=[{"role": "user", "content":
                                    f"QUESTION: {q}\nRUBRIC: {rubric}\nANSWER: {answer}"}])
    # Strip any ```json fences the model may add, then parse to a dict.
    txt = resp.content[0].text.strip().removeprefix("```json").removesuffix("```")
    return json.loads(txt)


# Step 6: Run every case, score it, and report a PASS RATE (regression testing for agents).
def run_evals() -> None:
    passed = 0
    for i, case in enumerate(EVAL_SET, 1):
        ans = agent(case["q"])                     # 6a: run the system under test

        if "check" in case:                        # 6b: deterministic scoring
            ok, why = case["check"](ans), "rule check"
        else:                                      # 6c: LLM-as-judge scoring
            verdict = llm_judge(case["q"], case["rubric"], ans)
            ok, why = verdict["pass"], verdict["reason"]

        passed += ok
        print(f"[{i}] {'PASS' if ok else 'FAIL'} - {why}")

    # Step 7: The headline metric -- track this per change (prompt / model / tools).
    print(f"\n[STEP 7] Score: {passed}/{len(EVAL_SET)}")


# Step 8: Entry point.
if __name__ == "__main__":
    run_evals()
