# reflection_agent.py -- generate code, run tests, feed failures back
# Topic 7: Reflection & Self-Correction
# Run:  conda activate agentic && python reflection_agent.py

# Step 0: Imports.
#   subprocess/sys/tempfile let us actually RUN the model's code against real tests.
import anthropic, subprocess, sys, tempfile, textwrap, re

# Step 1: Create the API client (reads ANTHROPIC_API_KEY from the environment).
client = anthropic.Anthropic()

# Step 2: The GROUND-TRUTH critic -- a set of unit tests the code must pass.
#   External feedback (tests) is a far stronger signal than the model critiquing itself.
TESTS = textwrap.dedent("""
    from solution import slugify
    assert slugify("Hello, World!") == "hello-world"
    assert slugify("  many   spaces  ") == "many-spaces"
    assert slugify("Already-Good") == "already-good"
    print("ALL TESTS PASSED")
""")


def run_tests(code: str) -> str:
    # Step 3: Write the model's code + the tests to a temp dir and execute them.
    #   Returns combined stdout+stderr -- i.e. the exact feedback to hand back.
    with tempfile.TemporaryDirectory() as d:
        open(f"{d}/solution.py", "w").write(code)
        open(f"{d}/test_it.py", "w").write(TESTS)
        p = subprocess.run([sys.executable, "test_it.py"], cwd=d,
                           capture_output=True, text=True, timeout=10)
    return p.stdout + p.stderr


def extract_code(text: str) -> str:
    # Step 4: Pull the python code out of a ```python ...``` block (or return raw text).
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    return m.group(1) if m else text


def solve(max_rounds: int = 3) -> str:
    # Step 5: The generation prompt.
    prompt = ("Write a Python function slugify(s) that lowercases, trims, and "
              "converts runs of non-alphanumeric chars to single hyphens. "
              "Return ONLY a python code block.")
    messages = [{"role": "user", "content": prompt}]

    # Step 6: The reflection loop: generate -> test -> feed failures back -> revise.
    for r in range(max_rounds):
        print(f"\n===== round {r+1} =====")

        # Step 6a: GENERATOR -- ask the model for code.
        resp = client.messages.create(model="claude-haiku-4-5",
                                      max_tokens=700, messages=messages)
        code = extract_code(resp.content[0].text)

        # Step 6b: CRITIC -- run the real tests against the generated code.
        feedback = run_tests(code)
        print(f"[STEP 6b] test output:\n{feedback.strip()}")

        # Step 6c: STOP CONDITION -- tests pass -> return the working code.
        if "ALL TESTS PASSED" in feedback:
            print("[STEP 6c] tests passed -> done")
            return code

        # Step 6d: REVISE -- hand back the EXACT error text (vague critiques are useless).
        messages.append({"role": "assistant", "content": resp.content[0].text})
        messages.append({"role": "user", "content":
            f"Your code failed these tests:\n{feedback}\nFix it. Code block only."})

    # Step 7: Bounded -- never loop forever; fail loudly after max_rounds.
    raise RuntimeError("Failed after retries.")


# Step 8: Entry point.
if __name__ == "__main__":
    print("\n[STEP 8] FINAL CODE:\n" + solve())
