"""Fast discovery with slow registration: the stable registry, the experimental pool and the archive.

* Stable registry T^S  - the only tools the deployed agent uses when collecting experience.
* Experimental pool T^E - frozen candidates that passed discovery (Delta P_disc > 0) and are
                          accumulating paired trials on K subsequent cohorts.
* Archive              - rejected candidates with their evaluation outcomes (fed back to reflection).

Registration rule after K trials (Eq. registration in the paper):
    mean_k Delta P_k > delta   and   mean_k Delta C_k <= 0
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

from .utils import mean, short_hash


@dataclass
class Candidate:
    tool_id: str
    ref: list  # stable-registry snapshot the candidate is compared against
    round: int  # round of discovery
    discovery: dict  # {"dP": ..., "dC": ...}
    spec: dict = field(default_factory=dict)
    trials: list = field(default_factory=list)  # [{"cohort": k, "dP": ..., "dC": ..., "round": r}]
    reevaluated: bool = False

    @property
    def name(self) -> str:
        return self.tool_id.split("@")[0]


def augment(tool_ids, tool_id) -> list:
    """H (+) T: make a candidate available to the planner (a same-name tool is replaced)."""
    name = tool_id.split("@")[0]
    return sorted([t for t in tool_ids if t.split("@")[0] != name] + [tool_id])


def registration_decision(trials, delta: float):
    """Return (accept, mean dP, mean dC) for a candidate's completed trial history."""
    dP = mean(t["dP"] for t in trials)
    dC = mean(t["dC"] for t in trials)
    return (bool(trials) and dP > delta and dC <= 0), dP, dC


class Registry:
    def __init__(self):
        self.stable: dict = {}  # name -> tool id
        self.pool: list = []  # Candidates under trial
        self.archive: list = []  # rejected candidates (dicts)
        self.events: list = []  # registration history (dicts)

    def stable_ids(self) -> list:
        return sorted(self.stable.values())

    def key(self, ids=None) -> str:
        return short_hash(sorted(self.stable_ids() if ids is None else ids))

    def register(self, cand: Candidate, round_idx: int, how: str, dP=None, dC=None) -> Optional[str]:
        previous = self.stable.get(cand.name)
        self.stable[cand.name] = cand.tool_id
        self.events.append({"round": round_idx, "tool": cand.tool_id, "replaced": previous, "how": how,
                            "discovery": cand.discovery, "trials": cand.trials, "mean_dP": dP, "mean_dC": dC})
        return previous

    def reject(self, cand: Candidate, round_idx: int, reason: str, dP=None, dC=None):
        self.archive.append({"round": round_idx, "tool": cand.tool_id, "reason": reason,
                             "description": cand.spec.get("description", ""), "discovery": cand.discovery,
                             "trials": cand.trials, "mean_dP": dP, "mean_dC": dC})

    # ---------------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {"stable": self.stable, "pool": [asdict(c) for c in self.pool],
                "archive": self.archive, "events": self.events}

    @classmethod
    def from_dict(cls, d: dict) -> "Registry":
        r = cls()
        r.stable = dict(d.get("stable", {}))
        r.pool = [Candidate(**c) for c in d.get("pool", [])]
        r.archive = list(d.get("archive", []))
        r.events = list(d.get("events", []))
        return r
