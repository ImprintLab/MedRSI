# MedRSI

**Recursive self-improvement for medical agents.**

MedRSI lets a diagnostic agent learn from its own failures by inventing new tools. It writes code, composes existing tools and trains small models. Two safeguards come from clinical practice:

* **Clinical-cost-aware failure prioritisation.** The agent spends its improvement effort on the failures that matter most for patients, not just the most frequent ones.
* **Fast discovery, slow registration.** A new tool joins the agent only after it shows a sustained benefit, without increasing clinical cost, across several later patient cohorts.

---

## Install

```bash
git clone <this repo> MedRSI && cd MedRSI
pip install -e ".[all]"        # core + OpenAI and Claude clients
```

The core package needs Python ≥ 3.9 and has no dependencies. Install the client you need with `.[openai]` or `.[anthropic]`. Generated tools can use any package installed in your environment (numpy, scipy, scikit-learn, PyTorch, MONAI, ...).

## Use it on your own task

**1. Describe the task** in `task.json`. The consequence rubric is written once, before self-improvement starts:

```json
{
  "name": "glaucoma",
  "description": "Diagnose glaucoma from a colour fundus photograph.",
  "labels": ["normal", "glaucoma"],
  "positive_label": "glaucoma",
  "inputs": {"fundus": "colour fundus photograph"},
  "consequence": {"glaucoma->normal": 3, "normal->glaucoma": 1},
  "rubric": "Missed glaucoma that is discharged is serious (3); unnecessary referral is minor (1)."
}
```

For regression tasks such as ejection fraction, set `"labels": null`, `"tolerance": 5` and `"consequence_bins": [5, 10, 15, 20]`. An error is then `|pred - ref| > tolerance`, and its consequence score is the number of bins the error exceeds.

**2. List the cases** in `cases.jsonl` (`.json` and `.csv` also work). Relative paths are resolved against the file's folder:

```json
{"id": "eye_001", "patient_id": "P001", "inputs": {"fundus": "img/eye_001.png", "age": 63}, "label": "glaucoma"}
```

Optional fields:

* `"split"`: one of `train | discovery | trial | test`. If it is missing, cases are split by patient automatically.
* `"meta"`: extra clinical context for the judge (e.g. disease stage).

The agent never sees `label`, `meta`, the case id or file paths.

**3. Run**, from the command line or from Python:

```bash
medrsi run --task task.json --data cases.jsonl --llm openai:gpt-4o --rounds 20 --workdir runs/glaucoma
medrsi status --workdir runs/glaucoma
medrsi eval   --workdir runs/glaucoma --split test              # current agent
medrsi eval   --workdir runs/glaucoma --split test --round 5    # any earlier snapshot
```

```python
from medrsi import MedRSI, Config, load_cases

rsi = MedRSI("task.json", load_cases("cases.jsonl"), llm="openai:gpt-4o",
             workdir="runs/glaucoma", config=Config(lam=2.0, trial_cohorts=3))
rsi.run(rounds=20)                      # resumable: rerun to continue where it stopped
print(rsi.evaluate("test"))             # bacc, f1, sens, spec, cost_per_100, severe_pct, ...
agent = rsi.agent()                     # the evolved agent
print(agent.diagnose(rsi.splits["test"][0]).prediction)
```

**4. Inspect** the run folder. Everything is plain JSON and Python files:

```
runs/glaucoma/
  library/<tool>-<hash8>/   tool.py, artifacts/ (weights), meta.json (spec, motivating failures, checks)
  rounds/round_007.json     clusters, specs, build attempts, discovery/trial results, decisions
  history.jsonl             one line per round (experience metrics, registry, test if monitored)
  state.json                registry, experimental pool with trial histories, archive, snapshots
  diagnoses.jsonl           cached execution records (makes paired comparisons cheap and resumable)
```

Later, `medrsi.load_agent("runs/glaucoma", llm)` rebuilds the evolved agent without the data.

## Tools

A tool is a single Python file, whether the agent built it or you wrote it:

```python
DESCRIPTION = "Vertical cup-to-disc ratio (0-1) from the disc/cup segmentation."

def run(inputs, ctx):                          # inputs: one patient's input dict
    seg = ctx.call_tool("segment_disc_cup")    # composition: call another stable tool
    ...
    return {"vcdr": 0.62, "quality": "good"}   # evidence for the agent (numpy arrays allowed)

def train(cases, ctx):                         # optional: the model-development route
    ...                                        # fit on labelled cases, save to ctx.artifact_dir
```

**Starting from existing tools** works like initialising MedRSI from MedAgent-Pro. Wrap each tool this way (a folder with `tool.py` plus `artifacts/` is also accepted) and pass the files as seed tools:

```bash
medrsi run ... --seed-tools my_tools/segment_optic_cup.py my_tools/segment_optic_disc.py
```

Every tool version is frozen in its own folder. Its hash covers both code and weights, so an id like `vertical_cdr@6e0e7ad5` always refers to exactly one package. Candidates are trained and checked in a subprocess with a time limit, so a crashing or hanging candidate cannot stop the loop.

## Models

| spec | backend |
|---|---|
| `openai:gpt-4o` | OpenAI |
| `openai:<model>@http://host:port/v1` | any OpenAI-compatible server (vLLM, SGLang, Ollama, LM Studio, OpenRouter, ...) |
| `anthropic:claude-opus-5` | Claude via the Anthropic SDK (server-side refusal fallbacks on by default; `AnthropicChat(fallbacks=None)` turns them off) |
| `py:my_module:MyModel` | any Python callable `fn(messages, tag) -> str`, e.g. a local HuggingFace model |

You can use a stronger model for reflection and tool building than for diagnosis with `--builder-llm` / `MedRSI(..., builder_llm=...)`. Every backend counts its tokens in `llm.usage`, and `AnthropicChat(effort="low" | "medium" | ...)` trades thinking depth for cost.

## Citation

```bibtex
@article{wu2026medrsi,
  title={MedRSI: Recursive Self-Improvement for Medical Agents via Clinically Aligned Self-Evolution},
  author={Wu, Junde and Zhu, Jiayuan and Hu, Minghao and Liu, Fenglin and Pan, Jiazhen},
  journal={arXiv preprint arXiv:2609.24838},
  year={2026}
}
```