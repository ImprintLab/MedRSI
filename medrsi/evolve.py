"""From failures to candidate tools: prioritisation, clustering, reflection and building.

Clinical-cost-aware failure prioritisation:  q_i = e_i * (1 + lambda * c_i),  c_i = s_i / 4.
Failures are clustered by their pattern (reference -> prediction, tool-use pattern); clusters are
ranked by their most consequential case (then by total priority), so high-consequence failures
get attention first without one frequent failure mode consuming the whole budget.
With lambda = 0 every error weighs 1 and ranking falls back to error frequency.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from . import prompts
from .agent import describe_tools
from .task import SEVERITY, Case, Diagnosis, Task
from .tools import ToolRunner
from .utils import extract_code, is_image_path, parse_json, slug, to_jsonable

log = logging.getLogger("medrsi")


@dataclass
class Failure:
    case: Case
    diagnosis: Diagnosis
    score: int  # s_i, 0..4
    justification: str
    priority: float  # q_i
    pattern: str


@dataclass
class Cluster:
    pattern: str
    failures: list  # highest-priority representatives shown to reflection
    size: int  # total number of failures with this pattern


def failure_pattern(task: Task, case: Case, d: Diagnosis) -> str:
    """Cluster key: what went wrong and which evidence the agent had."""
    if task.is_classification:
        what = f"{case.label} -> {d.prediction if d.prediction is not None else 'invalid'}"
    elif d.prediction is None:
        what = "invalid prediction"
    else:
        what = "over-estimate" if d.prediction > case.label else "under-estimate"
    tools = sorted(c["tool"].split("@")[0] + ("!" if "error" in (c.get("output") or {}) else "") for c in d.tool_calls)
    return f"{what} | tools: {', '.join(tools) or 'none'}"


def collect_failures(task: Task, cases, diagnoses, judge, lam: float) -> list:
    out = []
    for case, d in zip(cases, diagnoses):
        if not task.is_error(case.label, d.prediction):
            continue
        s, why = judge(case, d.prediction)
        out.append(Failure(case, d, s, why, 1.0 + lam * s / 4.0, failure_pattern(task, case, d)))
    return out


def select_clusters(failures, max_clusters: int, per_cluster: int, cluster: bool = True) -> list:
    """Group failures and pick the clusters that receive this round's development budget."""
    groups = {}
    for i, f in enumerate(failures):
        groups.setdefault(f.pattern if cluster else i, []).append(f)
    ranked = sorted(
        groups.values(),
        key=lambda g: (max(f.priority for f in g), sum(f.priority for f in g)),
        reverse=True,
    )
    return [
        Cluster(g[0].pattern, sorted(g, key=lambda f: f.priority, reverse=True)[:per_cluster], len(g))
        for g in ranked[:max_clusters]
    ]


def _cases_text(task: Task, failures) -> str:
    blocks = []
    for i, f in enumerate(failures, 1):
        d = f.diagnosis
        non_image = {k: v for k, v in f.case.inputs.items() if not isinstance(v, str) or not is_image_path(v)}
        non_image = {k: v for k, v in non_image.items() if not (isinstance(v, str) and len(v) > 300)}
        tools = {c["tool"]: c["output"] for c in d.tool_calls}
        blocks.append(
            f"### Case {i} (priority {f.priority:.2f}; consequence {f.score}/4 {SEVERITY[f.score]})\n"
            f"Reference: {f.case.label} | Agent prediction: {d.prediction} (confidence {d.confidence})\n"
            f"Consequence: {f.justification}\n"
            f"Agent rationale: {d.rationale or '(none)'}\n"
            f"Tool outputs: {json.dumps(to_jsonable(tools, max_str=300))[:2500]}\n"
            f"Non-image inputs: {json.dumps(to_jsonable(non_image, max_str=200))[:800]}"
        )
    return "\n\n".join(blocks)


def _bullets(items) -> str:
    return "\n".join(f"- {x}" for x in items) or "(none)"


class Reflector:
    """Turns one failure cluster into a capability specification (a testable hypothesis)."""

    def __init__(self, task: Task, llm, routes, n_images: int = 2, temperature: float = 0.7):
        self.task, self.llm, self.routes, self.n_images, self.temperature = task, llm, list(routes), n_images, temperature

    def propose(self, cluster: Cluster, context: dict):
        failures = cluster.failures
        images = [v for f in failures for v in f.case.inputs.values() if is_image_path(v)][: self.n_images]
        user = prompts.REFLECT_USER.substitute(
            task=self.task.description,
            answer_format=self.task.answer_format(),
            rubric=self.task.rubric or json.dumps(self.task.consequence),
            stable=describe_tools(context["stable"]) if context["stable"] else "(none)",
            pending=_bullets(context["pending"]),
            rejected=_bullets(context["rejected"]),
            resources=context["resources"],
            n_cluster=cluster.size,
            pattern=cluster.pattern,
            n_shown=len(failures),
            cases=_cases_text(self.task, failures),
            routes=json.dumps(self.routes),
        )
        messages = [{"role": "system", "content": prompts.REFLECT_SYSTEM},
                    {"role": "user", "content": user, "images": images}]
        spec = parse_json(self.llm.chat(messages, temperature=self.temperature, tag="reflect")) or {}
        name = slug(spec.get("tool_name"))
        if not name or not spec.get("description"):
            return None
        spec["tool_name"] = name
        if spec.get("route") not in self.routes:
            spec["route"] = self.routes[0]
        spec["motivating_cases"] = [f.case.id for f in failures]
        spec["pattern"] = cluster.pattern
        return spec


class Builder:
    """Realises a specification as an executable, trained and checked tool package."""

    def __init__(self, task: Task, llm, library, packages, max_fix_attempts: int = 2,
                 timeout: int = 3600, temperature: float = 0.7):
        self.task, self.llm, self.library, self.packages = task, llm, library, packages
        self.max_fix_attempts, self.timeout, self.temperature = max_fix_attempts, timeout, temperature

    def _stable_text(self, stable_tools, example_inputs) -> str:
        if not stable_tools:
            return "(none)"
        lines, runner = [], ToolRunner({t.name: t for t in stable_tools}, example_inputs)
        for t in stable_tools:
            try:
                example = json.dumps(to_jsonable(runner.call(t.name), max_str=200))[:600]
            except Exception as e:
                example = f"(example failed: {type(e).__name__}: {e})"
            lines.append(f"- {t.name}: {' '.join(t.description.split())}\n  example output: {example}")
        return "\n".join(lines)

    def build(self, spec: dict, stable_tools, train_cases, check_cases, round_idx: int):
        """Returns ``(tool or None, info)``; info records every attempt for the development log."""
        example = check_cases[0]["inputs"] if check_cases else {}
        label_counts = {}
        for c in train_cases:
            label_counts[str(c["label"])] = label_counts.get(str(c["label"]), 0) + 1
        train_info = (f"{len(train_cases)} labelled training cases. "
                      f"Label distribution: {json.dumps(label_counts) if self.task.is_classification else 'continuous'}.\n"
                      f"Example case: {json.dumps(to_jsonable(train_cases[0] if train_cases else {}, max_str=200))[:800]}")
        user = prompts.BUILD_USER.substitute(
            spec=json.dumps({k: v for k, v in spec.items() if k != "motivating_cases"}, indent=1),
            example_inputs=json.dumps(to_jsonable(example, max_str=120)),
            route=spec["route"],
            route_hint=prompts.ROUTE_HINTS.get(spec["route"], ""),
            stable=self._stable_text(stable_tools, example),
            train_info=train_info,
            packages=", ".join(self.packages) or "(standard library only)",
            timeout_min=max(1, self.timeout // 60),
        )
        messages = [{"role": "system", "content": prompts.BUILD_SYSTEM}, {"role": "user", "content": user}]
        info = {"attempts": []}
        for attempt in range(self.max_fix_attempts + 1):
            reply = self.llm.chat(messages, temperature=self.temperature, tag="build")
            code = extract_code(reply)
            meta = {"spec": spec, "round": round_idx, "built_on": [t.id for t in stable_tools]}
            tool, report = self.library.build(spec["tool_name"], code, spec["description"], spec["route"],
                                              meta, train_cases, check_cases, stable_tools, self.timeout)
            info["attempts"].append({"ok": tool is not None, "error": report.get("error"), "trained": report.get("trained")})
            if tool is not None:
                info["tool"] = tool.id
                return tool, info
            log.info("    build attempt %d for %s failed: %s", attempt + 1, spec["tool_name"],
                     (report.get("error") or "").strip().splitlines()[-1:] or "")
            messages += [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": prompts.BUILD_FIX.substitute(
                    n_check=len(check_cases), error=report.get("error"),
                    outputs=json.dumps(report.get("outputs", []))[:1500])},
            ]
        return None, info
