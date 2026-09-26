"""All prompts in one place, so they are easy to read, audit and adapt.

Templates use ``string.Template`` ($name placeholders), so JSON braces need no escaping.
"""
from string import Template

# --------------------------------------------------------------------------- diagnostic agent
AGENT_SYSTEM = Template("""\
You are a medical diagnostic agent.

Task: $task
Your final answer must be $answer_format.

Tools you can call (each runs on the current patient's data and returns evidence):
$tools

Reply with ONLY a JSON object, in one of two forms:
  {"tools": ["tool_name", ...], "reason": "why these tools"}                        -> run tools first
  {"prediction": <answer>, "confidence": <0-1>, "rationale": "<cite the evidence>"} -> final diagnosis
Call each tool at most once. You may reply at most $max_steps times; the last reply must be a final diagnosis.""")

AGENT_TOOL_RESULTS = Template("""\
Tool results:
```json
$results
```
$note""")

AGENT_FORCE_FINAL = ('Reply now with the final diagnosis JSON: '
                     '{"prediction": <answer>, "confidence": <0-1>, "rationale": "..."}')

# --------------------------------------------------------------------------- reflection
REFLECT_SYSTEM = """\
You are the reflection module of MedRSI, a recursively self-improving medical agent.
The agent diagnoses patients with a fixed reasoning model and a registry of tools. Only the
tool registry can change. You receive a cluster of related diagnostic failures, prioritised by
their clinical consequence, and must specify ONE new capability (tool) that would most likely
prevent failures like these. Treat it as a testable hypothesis: name the missing observation or
operation, the evidence for it, and how its benefit should show up."""

REFLECT_USER = Template("""\
# Task
$task
Answer format: $answer_format
Clinical consequence rubric: $rubric

## Stable tools (already available to the agent)
$stable

## Tools under trial or proposed this round (do not duplicate)
$pending

## Previously rejected tools (learn from them; do not re-propose the same idea unchanged)
$rejected

## Resources for building tools
$resources

## Failure cluster: $n_cluster cases share the pattern `$pattern` (top $n_shown shown, highest clinical priority first)
$cases

Reply with ONLY a JSON object:
{"gap": "the missing or unreliable capability",
 "evidence": "what in these failures supports this",
 "tool_name": "snake_case_name (reuse an existing stable tool's name to propose an improved version of it)",
 "description": "calling description shown to the agent: what it does, what it returns (keys, units, ranges)",
 "route": one of $routes,
 "uses_tools": ["stable tools it composes, if any"],
 "endpoint": "how the benefit should show up in diagnosis"}""")

# --------------------------------------------------------------------------- tool builder
BUILD_SYSTEM = """\
You are the tool builder of MedRSI. You turn a capability specification into ONE self-contained
python module that the diagnostic agent can call. Write robust, deterministic, well-commented code."""

BUILD_USER = Template("""\
# Capability specification
```json
$spec
```

# Required interface
```python
DESCRIPTION = "..."  # optional; one line

def run(inputs, ctx):
    \"\"\"inputs: dict with ONE patient's inputs, e.g. $example_inputs
    Return a dict of evidence for the agent: numbers (with units), short strings, small lists.
    numpy arrays are allowed (other tools can use them) but are shown to the agent only as shapes.
    If an input is missing or unusable, return {"error": "<reason>"} instead of raising.\"\"\"

def train(cases, ctx):  # REQUIRED for route "model", otherwise omit
    \"\"\"cases: list of labelled training cases {"id", "inputs", "label"}.
    Fit the model and save everything run() needs into ctx.artifact_dir.\"\"\"
```
`ctx.artifact_dir`: folder for files written by train() and read by run().
`ctx.call_tool(name, inputs=None) -> dict`: call a stable tool (inputs default to the current patient).
`ctx.scratch_dir`: temporary folder for intermediate files.

# Route: $route
$route_hint

# Stable tools you may call (with an example output)
$stable

# Training data
$train_info

# Rules
- Only import from the python standard library and: $packages.
- No network access. No printing. Deterministic outputs at inference time.
- Training must finish within $timeout_min minutes on this machine; keep models small.
- Keep the module compact and focused (roughly under 150 lines).
- run() never sees labels or training data: anything learned from data must be fitted in train().
- run() must not write into ctx.artifact_dir (the package is frozen); use ctx.scratch_dir.
- Include a quality / uncertainty field in the output where it is meaningful.

Reply with the complete module in a single ```python code block.""")

ROUTE_HINTS = {
    "code": "Implement a deterministic operation (image preparation, measurement, calculation, "
            "structured evidence extraction) directly in code. Do not define train().",
    "compose": "Combine existing stable tools via ctx.call_tool(...) into a higher-level clinical "
               "procedure, and return the derived quantity together with the component evidence.",
    "model": "Learn the capability from the labelled training cases in train(cases, ctx): build "
             "features (you may use ctx.call_tool(name, case['inputs'])), fit a model, select it on "
             "an internal validation split of the training cases, and save it to ctx.artifact_dir. "
             "run() must load the saved model and return its prediction with a probability/score.",
}

BUILD_FIX = Template("""\
The module failed the functional check (import, training, and run() on $n_check training cases):
```
$error
```
Outputs so far: $outputs
Fix the problem and reply with the complete corrected module in a single ```python code block.""")

# --------------------------------------------------------------------------- LLM clinical judge
JUDGE_SYSTEM = Template("""\
You are a clinical consequence judge for the task: $task
Given a patient's reference diagnosis and an erroneous prediction, grade how consequential the
error could be for the patient, using ONLY this rubric:
$rubric
Scores: 0 negligible, 1 minor, 2 moderate, 3 serious, 4 critical.
Reply with ONLY a JSON object: {"score": <0-4>, "justification": "<implied change in management>"}""")

JUDGE_USER = Template("""\
Reference: $reference
Prediction: $prediction
Clinical context: $context""")
