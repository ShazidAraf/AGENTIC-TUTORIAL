# structured_output.py -- forced tool call + Pydantic validation + retry
# Topic 3: Structured Outputs (Validated JSON)
# Run:  conda activate agentic && python structured_output.py

# Step 0: Imports.
#   - `anthropic`           -> the SDK.
#   - pydantic BaseModel... -> declare a typed shape and validate data against it.
import anthropic
from pydantic import BaseModel, Field, ValidationError

# Step 1: Create the API client (reads ANTHROPIC_API_KEY from the environment).
client = anthropic.Anthropic()


# Step 2: Declare the OUTPUT SHAPE we want as a typed Pydantic model.
#   Downstream code will consume t.priority / t.customer -- not regexes over prose.
class Ticket(BaseModel):
    customer: str
    product: str
    # `pattern` constrains priority to exactly one of these three values.
    priority: str = Field(pattern="^(low|medium|high)$")
    # `max_length` caps the summary so the model can't ramble.
    summary: str = Field(max_length=200)


# Step 3: Turn the Pydantic model into a tool whose input_schema IS our desired shape.
#   THE TRICK: we don't really want the model to "use a tool" -- we want its ARGUMENTS,
#   because those arguments are forced to match our JSON schema.
EXTRACT_TOOL = {
    "name": "record_ticket",
    "description": "Record the extracted support ticket.",
    "input_schema": Ticket.model_json_schema(),   # Pydantic -> JSON Schema, for free
}


def extract_ticket(email: str, max_retries: int = 2) -> Ticket:
    # Step 4: Seed the conversation with the raw text to extract from.
    messages = [{"role": "user",
                 "content": f"Extract a support ticket from this email:\n\n{email}"}]

    # Step 5: Try up to (max_retries + 1) times -- a tiny self-correction loop.
    for attempt in range(max_retries + 1):
        print(f"\n===== attempt {attempt+1} =====")

        # Step 5a: Call the model and FORCE it to call our tool via tool_choice.
        #   This guarantees we get tool_use arguments back, not free-form prose.
        resp = client.messages.create(
            model="claude-haiku-4-5", max_tokens=500,   # cheapest current model
            tools=[EXTRACT_TOOL],
            tool_choice={"type": "tool", "name": "record_ticket"},  # FORCE it
            messages=messages,
        )

        # Step 5b: Pull the tool's arguments out of the response (the first tool_use block).
        args = next(b.input for b in resp.content if b.type == "tool_use")
        print(f"[STEP 5b] model returned args = {args}")

        # Step 5c: THE VALIDATION GATE. Ask Pydantic to build a Ticket from those args.
        #   If the args violate the schema (e.g. priority='urgent'), it raises ValidationError.
        try:
            ticket = Ticket(**args)
            print(f"[STEP 5c] validation PASSED")
            return ticket
        except ValidationError as e:
            # Step 5d: Validation failed -> feed the exact error back and let the model retry.
            print(f"[STEP 5d] validation FAILED -> asking model to fix it")
            messages.append({"role": "assistant", "content": resp.content})
            messages.append({"role": "user",
                "content": f"Validation failed: {e}. Call record_ticket again, fixed."})

    # Step 6: Exhausted retries without a valid object -> fail loudly (never silently).
    raise RuntimeError("Could not produce valid ticket.")


# Step 7: Entry point -- extract a ticket from a sample email and print the typed result.
if __name__ == "__main__":
    email = ("Hi, this is Dana from Acme. Our Model-X batteries drain in an hour. "
             "Production line is DOWN, please treat as urgent!")
    t = extract_ticket(email)
    print(f"\n[STEP 7] VALIDATED TICKET:\n{t.model_dump_json(indent=2)}")
