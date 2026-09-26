"""Clinical-consequence judges: grade how consequential an erroneous prediction could be.

A judge is any callable ``judge(case, prediction) -> (score, justification)`` with the
score on the ordinal scale 0 negligible, 1 minor, 2 moderate, 3 serious, 4 critical.
Correct predictions always get 0. Keep the judge frozen for a whole trajectory.
"""
from __future__ import annotations

import json
import threading

from . import prompts
from .task import SEVERITY, Case, Task
from .utils import parse_json, to_jsonable


class RubricJudge:
    """Deterministic judge from the task's prespecified rubric (the default).

    Classification: a consequence matrix ``{"reference->prediction": score}``.
    Regression: the score is the number of ``consequence_bins`` the absolute error exceeds.
    """

    def __init__(self, task: Task):
        self.task = task

    def __call__(self, case: Case, prediction):
        t = self.task
        if not t.is_error(case.label, prediction):
            return 0, "correct"
        if prediction is None:
            s = t.default_consequence
            return s, f"no valid prediction -> {SEVERITY[s]} (default)"
        if t.is_classification:
            s = t.consequence.get(f"{case.label}->{prediction}", t.default_consequence)
            return s, f"reference '{case.label}' predicted as '{prediction}' -> {SEVERITY[s]} (rubric)"
        err = abs(float(prediction) - float(case.label))
        s = max(1, min(4, sum(err > b for b in t.consequence_bins)))
        direction = "over" if prediction > case.label else "under"
        return s, f"{direction}-estimated by {err:.1f}{t.units} -> {SEVERITY[s]} (rubric)"


class LLMJudge:
    """LLM clinical judge with a frozen prompt and rubric (as in the paper). Scores are cached."""

    def __init__(self, task: Task, llm, rubric: str = None):
        self.task, self.llm = task, llm
        self.rubric = rubric or task.rubric or json.dumps(task.consequence)
        self._cache, self._lock = {}, threading.Lock()
        self._fallback = RubricJudge(task)

    def __call__(self, case: Case, prediction):
        if not self.task.is_error(case.label, prediction):
            return 0, "correct"
        key = (case.id, str(prediction))
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        context = {k: v for k, v in case.inputs.items() if not isinstance(v, str) or len(v) < 500}
        context.update(case.meta)
        messages = [
            {"role": "system", "content": prompts.JUDGE_SYSTEM.substitute(task=self.task.description, rubric=self.rubric)},
            {"role": "user", "content": prompts.JUDGE_USER.substitute(
                reference=case.label, prediction=prediction, context=json.dumps(to_jsonable(context)))},
        ]
        obj = parse_json(self.llm.chat(messages, temperature=0.0, tag="judge")) or {}
        try:
            result = (max(0, min(4, int(obj["score"]))), str(obj.get("justification", ""))[:500])
        except (KeyError, TypeError, ValueError):
            result = self._fallback(case, prediction)
        with self._lock:
            self._cache[key] = result
        return result


def as_judge(judge, task: Task):
    """Normalise user input: None -> RubricJudge, callable returning int or (int, str) -> judge."""
    if judge is None:
        return RubricJudge(task)

    def wrapped(case, prediction):
        if not task.is_error(case.label, prediction):
            return 0, "correct"
        out = judge(case, prediction)
        return (int(out[0]), str(out[1])) if isinstance(out, (tuple, list)) else (int(out), "")

    return judge if isinstance(judge, (RubricJudge, LLMJudge)) else wrapped
