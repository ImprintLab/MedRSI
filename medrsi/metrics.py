"""Diagnostic performance P and clinical cost C.

P is balanced accuracy for classification and -nMAE for regression (larger is always better).
C(H, V) = mean_i e_i * c_i with c_i = s_i / 4; we report 100*C as clinical cost per 100 patients.
"""
from __future__ import annotations

from .task import Task
from .utils import mean


def balanced_accuracy(y, yhat) -> float:
    classes = sorted(set(y))
    return mean(mean(p == c for t, p in zip(y, yhat) if t == c) for c in classes)


def binary_stats(y, yhat, positive) -> dict:
    tp = sum(t == positive and p == positive for t, p in zip(y, yhat))
    fn = sum(t == positive and p != positive for t, p in zip(y, yhat))
    fp = sum(t != positive and p == positive for t, p in zip(y, yhat))
    tn = sum(t != positive and p != positive for t, p in zip(y, yhat))
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else 0.0
    return {"sens": sens, "spec": spec, "f1": f1}


def summarize(task: Task, y, yhat, e, s, scale: float = 1.0, fallback: float = 0.0) -> dict:
    """All metrics for one cohort. ``e``: error indicators, ``s``: consequence scores (0-4)."""
    n = len(y)
    out = {"n": n, "errors": int(sum(e))}
    if task.is_classification:
        out["bacc"] = balanced_accuracy(y, yhat) if n else 0.0
        out["acc"] = mean(t == p for t, p in zip(y, yhat))
        if task.positive_label is not None:
            out.update(binary_stats(y, yhat, task.positive_label))
        out["P"] = out["bacc"]
    else:
        abs_err = [abs((fallback if p is None else p) - t) for t, p in zip(y, yhat)]
        out["mae"] = mean(abs_err)
        out["nmae"] = out["mae"] / (scale or 1.0)
        out["within_tol"] = mean(a <= task.tolerance for a in abs_err)
        out["P"] = -out["nmae"]
    out["C"] = mean(ei * si / 4 for ei, si in zip(e, s))
    out["cost_per_100"] = 100 * out["C"]
    out["severe_pct"] = 100 * mean(ei and si >= 3 for ei, si in zip(e, s))
    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in out.items()}
