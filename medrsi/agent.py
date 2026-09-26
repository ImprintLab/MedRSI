"""The medical agent H = (M, P, T): a fixed reasoning model M, a fixed workflow P and a tool set T.

The workflow is a JSON tool-calling loop that works with any chat model:
the model sees the task, the tool descriptions and the patient inputs (images attached),
may request tools (their outputs are returned as evidence), and finally commits a prediction.
"""
from __future__ import annotations

import json
import os
import tempfile

from . import prompts
from .task import Case, Diagnosis, Task
from .tools import ToolRunner
from .utils import find_images, is_image_path, parse_json, to_jsonable


def describe_tools(tools) -> str:
    if not tools:
        return "(no tools available - diagnose from the inputs directly)"
    return "\n".join(f"- {t.name}: {' '.join(t.description.split())}" for t in tools)


def format_case(task: Task, case: Case, send_images: bool = True):
    """Patient inputs as text + list of attached images. File paths are never shown
    (they often encode labels, e.g. 'Glaucoma/img_001.png')."""
    lines, images = ["Patient case. Inputs:"], []
    for key, value in case.inputs.items():
        desc = f" ({task.inputs[key]})" if key in task.inputs else ""
        if is_image_path(value):
            if send_images:
                images.append(value)
                lines.append(f"- {key}{desc}: [image {len(images)} attached]")
            else:
                lines.append(f"- {key}{desc}: [image, available to tools]")
        elif isinstance(value, str) and os.path.exists(value):
            lines.append(f"- {key}{desc}: [{os.path.splitext(value)[1] or 'file'} file, available to tools]")
        else:
            lines.append(f"- {key}{desc}: {json.dumps(to_jsonable(value, max_str=1500))}")
    return "\n".join(lines), images


class MedAgent:
    def __init__(self, task: Task, llm, tools=(), max_steps: int = 3, temperature: float = 0.0, send_images: bool = True):
        self.task, self.llm = task, llm
        self.tools = {t.name: t for t in tools}
        self.max_steps, self.temperature, self.send_images = max(1, max_steps), temperature, send_images

    def system_prompt(self) -> str:
        return prompts.AGENT_SYSTEM.substitute(
            task=self.task.description,
            answer_format=self.task.answer_format(),
            tools=describe_tools(list(self.tools.values())),
            max_steps=self.max_steps,
        )

    def diagnose(self, case: Case) -> Diagnosis:
        try:
            with tempfile.TemporaryDirectory(prefix="medrsi_") as scratch:
                return self._diagnose(case, ToolRunner(self.tools, case.inputs, scratch))
        except Exception as e:  # LLM/API failure: record it, the caller decides what to do
            return Diagnosis(case.id, error=f"{type(e).__name__}: {e}")

    def _diagnose(self, case: Case, runner: ToolRunner) -> Diagnosis:
        text, images = format_case(self.task, case, self.send_images)
        messages = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": text, "images": images},
        ]
        calls, obj, reply = [], {}, ""
        for step in range(self.max_steps):
            reply = self.llm.chat(messages, temperature=self.temperature, tag="agent")
            obj = parse_json(reply) or {}
            last = step == self.max_steps - 1
            if "prediction" in obj or last:
                break
            wanted = obj.get("tools") or []
            wanted = [wanted] if isinstance(wanted, str) else wanted
            done = {c["tool"].split("@")[0] for c in calls}
            wanted = [n for n in wanted if n in self.tools and n not in done]
            messages.append({"role": "assistant", "content": reply})
            if not wanted:
                messages.append({"role": "user", "content": prompts.AGENT_FORCE_FINAL})
                continue
            results, new_images = {}, []
            for name in wanted:
                try:
                    out = to_jsonable(runner.call(name))
                except Exception as e:
                    out = {"error": f"{type(e).__name__}: {e}"}
                results[name] = out
                calls.append({"tool": self.tools[name].id, "output": out})
                new_images += find_images(out)
            note = (prompts.AGENT_FORCE_FINAL if step + 1 == self.max_steps - 1
                    else "Call more tools, or give the final diagnosis.")
            results_text = json.dumps(results, indent=1)
            if len(results_text) > 12000:
                results_text = results_text[:12000] + "\n...[truncated]"
            messages.append({
                "role": "user",
                "content": prompts.AGENT_TOOL_RESULTS.substitute(results=results_text, note=note),
                "images": new_images if self.send_images else [],
            })
        raw_pred = obj["prediction"] if "prediction" in obj else (None if obj else reply)
        prediction = self.task.parse_prediction(raw_pred)
        try:
            confidence = float(obj.get("confidence")) if obj.get("confidence") is not None else None
        except (TypeError, ValueError):
            confidence = None
        return Diagnosis(
            case_id=case.id,
            prediction=prediction,
            confidence=confidence,
            rationale=str(obj.get("rationale", ""))[:1000],
            tool_calls=calls,
            raw=reply[-1500:],
            error=None if prediction is not None else "no valid prediction",
        )
