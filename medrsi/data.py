"""Loading cases and patient-level splitting.

Cases file formats:
  * JSONL / JSON list: {"id": "...", "inputs": {"fundus": "img/001.png", "age": 63},
                        "label": "glaucoma", "patient_id": "P001", "split": "train", "meta": {...}}
  * CSV: columns id, label, [patient_id], [split]; columns named meta_* go to meta;
         every other column becomes an input.
Relative file paths in inputs are resolved against the cases file's folder.

Splits (patient is the unit of separation):
  train      - clinical experience batches + labelled data for model development
  discovery  - screening of new candidate tools (fast discovery)
  trial      - divided into K trial cohorts for registration (slow registration)
  test       - locked; only used by ``MedRSI.evaluate``
"""
from __future__ import annotations

import csv
import json
import os
import random

from .task import Case

SPLITS = ("train", "discovery", "trial", "test")


def _resolve(value, base):
    if isinstance(value, str) and value and not os.path.isabs(value):
        candidate = os.path.join(base, value)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return value


def _number(v):
    try:
        f = float(v)
        return int(f) if f.is_integer() and "." not in str(v) else f
    except (TypeError, ValueError):
        return v


def load_cases(path: str) -> list:
    base = os.path.dirname(os.path.abspath(path))
    rows = []
    if path.endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                reserved = {"id", "label", "patient_id", "split"}
                rows.append({
                    "id": r["id"],
                    "label": r.get("label"),
                    "patient_id": r.get("patient_id") or None,
                    "split": r.get("split") or None,
                    "inputs": {k: _number(v) for k, v in r.items() if k not in reserved and not k.startswith("meta_") and v != ""},
                    "meta": {k[5:]: _number(v) for k, v in r.items() if k.startswith("meta_")},
                })
    else:
        with open(path, encoding="utf-8") as f:
            if path.endswith(".jsonl"):
                rows = [json.loads(line) for line in f if line.strip()]
            else:
                rows = json.load(f)
    cases = []
    for r in rows:
        inputs = {k: _resolve(v, base) for k, v in (r.get("inputs") or {}).items()}
        cases.append(Case(str(r["id"]), inputs, r.get("label"), r.get("patient_id"), r.get("split"), r.get("meta") or {}))
    ids = [c.id for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("Case ids must be unique.")
    return cases


def _split_by_patient(cases, fractions: dict, seed: int) -> dict:
    """Assign whole patients to splits so the case counts follow ``fractions``."""
    groups = {}
    for c in cases:
        groups.setdefault(c.patient, []).append(c)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    total, out, bounds, acc = len(cases), {k: [] for k in fractions}, [], 0.0
    norm = sum(fractions.values()) or 1.0
    for name, frac in fractions.items():
        acc += frac / norm
        bounds.append((name, acc * total))
    count = 0
    for k in keys:
        name = next((n for n, b in bounds if count < b - 1e-9), bounds[-1][0])
        out[name] += groups[k]
        count += len(groups[k])
    return out


def split_cases(cases, ratios=(0.5, 0.15, 0.2, 0.15), seed: int = 0) -> dict:
    """Split into train / discovery / trial / test, honouring any explicit ``case.split``."""
    out = {s: [] for s in SPLITS}
    unassigned = []
    for c in cases:
        if c.split in out:
            out[c.split].append(c)
        elif c.split:
            raise ValueError(f"Case {c.id}: unknown split '{c.split}' (use one of {SPLITS}).")
        else:
            unassigned.append(c)
    if unassigned:
        for name, part in _split_by_patient(unassigned, dict(zip(SPLITS, ratios)), seed).items():
            out[name] += part
    return out


def make_cohorts(cases, k: int, seed: int = 0) -> list:
    """Divide the trial split into K disjoint, patient-separated cohorts of similar size."""
    parts = _split_by_patient(cases, {i: 1.0 for i in range(k)}, seed + 1)
    return [parts[i] for i in range(k)]
