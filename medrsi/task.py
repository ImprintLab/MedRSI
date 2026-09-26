"""Core data structures.

* ``Task``      - what to diagnose, the label space and the clinical-consequence rubric.
* ``Case``      - one patient examination (inputs + hidden reference label).
* ``Diagnosis`` - one execution record tau_i: tool calls, evidence and the committed prediction.
"""
from __future__ import annotations

import dataclasses
import json
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

SEVERITY = ["negligible", "minor", "moderate", "serious", "critical"]


@dataclass
class Case:
    """One patient examination. ``label`` is the reference and is never shown to the agent.

    ``inputs`` maps an input name to a value: a file path (image, volume, report...),
    a number or a short text. ``meta`` can hold extra clinical context used only by the
    consequence judge (e.g. disease stage); it is also hidden from the agent.
    """

    id: str
    inputs: dict
    label: Any = None
    patient_id: Optional[str] = None
    split: Optional[str] = None
    meta: dict = field(default_factory=dict)

    @property
    def patient(self) -> str:
        return str(self.patient_id or self.id)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class Task:
    """A clinical task. ``labels=None`` makes it a regression task (e.g. ejection fraction)."""

    name: str
    description: str
    labels: Optional[list] = None
    inputs: dict = field(default_factory=dict)  # input name -> human-readable description
    positive_label: Optional[str] = None  # used for F1 / sensitivity in binary tasks
    # Clinical-consequence rubric, fixed before self-improvement starts.
    # Classification: {"reference->prediction": score in 0..4}; missing pairs use default_consequence.
    consequence: dict = field(default_factory=dict)
    default_consequence: int = 2
    # Regression: an error is |pred - ref| > tolerance; score = number of bins exceeded (max 4).
    tolerance: float = 5.0
    consequence_bins: list = field(default_factory=lambda: [5, 10, 15, 20])
    units: str = ""
    rubric: str = ""  # free-text rubric; shown to reflection and used by the optional LLM judge

    def __post_init__(self):
        flat = {}
        for k, v in (self.consequence or {}).items():
            if isinstance(v, dict):  # nested form {"glaucoma": {"normal": 3}}
                for p, s in v.items():
                    flat[f"{k}->{p}"] = int(s)
            else:
                flat[str(k).replace(" ", "")] = int(v)
        self.consequence = flat
        if self.labels:
            self.labels = [str(x) for x in self.labels]

    # ------------------------------------------------------------------ io
    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            warnings.warn(f"Ignoring unknown task fields: {sorted(unknown)}")
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_json(cls, path: str) -> "Task":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    # ------------------------------------------------------------- helpers
    @property
    def is_classification(self) -> bool:
        return bool(self.labels)

    def answer_format(self) -> str:
        if self.is_classification:
            return "exactly one label from " + json.dumps(self.labels)
        return f"a single number{f' in {self.units}' if self.units else ''}"

    def parse_label(self, value):
        """Normalise a reference label (str for classification, float for regression)."""
        if value is None:
            return None
        return str(value) if self.is_classification else float(value)

    def parse_prediction(self, value):
        """Map a model answer to a valid label / number, or None if it cannot be parsed."""
        if value is None:
            return None
        if self.is_classification:
            s = str(value).strip().strip("'\"").lower()
            for lab in self.labels:
                if s == lab.lower():
                    return lab
            hits = [lab for lab in self.labels if re.search(rf"\b{re.escape(lab.lower())}\b", s)]
            return hits[0] if len(hits) == 1 else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        m = re.search(r"-?\d+(?:\.\d+)?", str(value))
        return float(m.group()) if m else None

    def is_error(self, truth, pred) -> bool:
        """Error indicator e_i (Eq. for classification, tolerance rule for regression)."""
        if pred is None:
            return True
        if self.is_classification:
            return str(pred) != str(truth)
        return abs(float(pred) - float(truth)) > self.tolerance


@dataclass
class Diagnosis:
    """Execution record of one diagnosis: tool calls, their evidence, and the final prediction."""

    case_id: str
    prediction: Any = None
    confidence: Optional[float] = None
    rationale: str = ""
    tool_calls: list = field(default_factory=list)  # [{"tool": "name@hash", "output": {...}}]
    raw: str = ""
    error: Optional[str] = None

    @property
    def tools_used(self) -> list:
        return [c["tool"].split("@")[0] for c in self.tool_calls]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Diagnosis":
        return cls(**d)
