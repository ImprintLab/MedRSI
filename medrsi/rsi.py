"""MedRSI: the recursive self-improvement loop.

Each round r:
  1. Diagnosis       - the stable agent H_r diagnoses an experience batch (labels withheld).
  2. Prioritisation  - errors get consequence scores s_i and priorities q_i = e_i (1 + lambda s_i/4).
  3. Reflection      - the top failure clusters become capability specifications.
  4. Development     - the builder writes / composes / trains candidate tools (+ functional checks).
  5. Discovery       - candidates with Delta P_disc > 0 on the discovery cohort enter the pool.
  6. Trial & registration - pooled candidates are compared with their reference agent on K
     subsequent cohorts; registered only if mean Delta P > delta and mean Delta C <= 0.
The next round starts from H_{r+1} = (M, P, T^S_{r+1}).
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import logging
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

from .agent import MedAgent
from .data import make_cohorts, split_cases
from .evolve import Builder, Reflector, collect_failures, select_clusters
from .judge import as_judge
from .llm import make_llm
from .metrics import summarize
from .registry import Candidate, Registry, augment, registration_decision
from .task import Diagnosis, Task
from .tools import Library
from .utils import mean, short_hash

log = logging.getLogger("medrsi")
PROMPT_VERSION = "1"
COMMON_PACKAGES = ["numpy", "scipy", "sklearn", "PIL", "skimage", "cv2", "pandas", "torch", "torchvision",
                   "monai", "timm", "nibabel", "pydicom", "SimpleITK"]


@dataclass
class Config:
    # clinical-cost-aware prioritisation
    lam: float = 2.0  # lambda; 0 = uniform error priority (ablation)
    # fast discovery / slow registration
    trial_cohorts: int = 3  # K
    delta: Optional[float] = None  # registration margin; default 0.005 (bAcc) or 0.01 (nMAE)
    trials_per_round: int = 1  # trial cohorts a pooled candidate advances per round
    immediate_registration: bool = False  # ablation: register right after discovery
    # improvement budget per round
    batch_size: int = 60  # experience cases diagnosed per round
    max_candidates: int = 4  # failure clusters reflected on / candidates built per round
    cases_per_cluster: int = 4  # representative failures shown to reflection
    cluster_failures: bool = True  # ablation: False = one failure per cluster
    routes: tuple = ("code", "compose", "model")  # allowed development routes
    max_fix_attempts: int = 2  # builder self-repair attempts after a failed functional check
    build_timeout: int = 3600  # seconds for training + checks of one candidate
    check_cases: int = 3  # training cases used for functional checks
    max_train_cases: Optional[int] = None  # cap on cases passed to train()
    packages: Optional[list] = None  # importable packages for built tools (default: auto-detect)
    # agent
    max_steps: int = 3  # replies per diagnosis (tool requests + final answer)
    diagnosis_temperature: float = 0.0
    build_temperature: float = 0.7
    send_images: bool = True
    reflect_images: int = 2
    # data
    split_ratios: tuple = (0.5, 0.15, 0.2, 0.15)  # train, discovery, trial, test (if no split given)
    # misc
    workers: int = 8  # parallel diagnoses
    seed: int = 0
    eval_every: int = 0  # evaluate on the test split every N rounds (monitoring only)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        known = {f.name for f in dataclasses.fields(cls)}
        bad = set(d) - known
        if bad:
            raise ValueError(f"Unknown config keys: {sorted(bad)}")
        return cls(**d)


class DiagnosisCache:
    """Append-only cache of execution records keyed by (agent+toolset, case)."""

    def __init__(self, path: str):
        self.path, self._lock, self._data = path, threading.Lock(), {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        self._data[(r["key"], r["case"])] = Diagnosis.from_dict(r["d"])

    def get(self, key, case_id):
        return self._data.get((key, case_id))

    def put(self, key, d: Diagnosis):
        with self._lock:
            self._data[(key, d.case_id)] = d
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "case": d.case_id, "d": d.to_dict()}, default=str) + "\n")


class MedRSI:
    """Recursive self-improvement of a medical agent.

    Args:
        task: a ``Task`` or a path to task.json.
        cases: list of ``Case`` (see ``medrsi.load_cases``).
        llm: reasoning model M (``LLM`` object or spec string like "openai:gpt-4o").
        workdir: where tools, logs, caches and the resumable state are written.
        config: ``Config`` (paper defaults) or a dict of overrides.
        builder_llm: optional different model for reflection and tool building.
        judge: consequence judge; default ``RubricJudge(task)``. Use ``LLMJudge`` or any
            callable ``(case, prediction) -> score | (score, justification)``.
        seed_tools: hand-written tools that form the initial stable registry T^S_0
            (``.py`` files or folders with tool.py + artifacts/).
    """

    def __init__(self, task, cases, llm, workdir: str = "medrsi_run", config=None,
                 builder_llm=None, judge=None, seed_tools=()):
        self.task = task if isinstance(task, Task) else Task.from_json(task)
        self.cfg = Config.from_dict(config) if isinstance(config, dict) else (config or Config())
        self.llm = make_llm(llm)
        self.builder_llm = make_llm(builder_llm) if builder_llm else self.llm
        self.judge = as_judge(judge, self.task)
        self.workdir = os.path.abspath(workdir)
        os.makedirs(os.path.join(self.workdir, "rounds"), exist_ok=True)
        self.library = Library(os.path.join(self.workdir, "library"))
        for c in cases:
            c.label = self.task.parse_label(c.label)
        self.cases = {c.id: c for c in cases}
        self._init_state(list(cases), seed_tools)
        self.cache = DiagnosisCache(os.path.join(self.workdir, "diagnoses.jsonl"))
        cfg = self.cfg
        self.delta = cfg.delta if cfg.delta is not None else (0.005 if self.task.is_classification else 0.01)
        labels = [c.label for c in self.splits["train"]]
        if self.task.is_classification:
            self.scale, self.fallback = 1.0, 0.0
        else:
            mu = mean(labels)
            self.scale = (mean((x - mu) ** 2 for x in labels) ** 0.5) or 1.0
            self.fallback = mu
        packages = cfg.packages if cfg.packages is not None else [
            p for p in COMMON_PACKAGES if importlib.util.find_spec(p) is not None]
        self.reflector = Reflector(self.task, self.builder_llm, cfg.routes, cfg.reflect_images, cfg.build_temperature)
        self.builder = Builder(self.task, self.builder_llm, self.library, packages, cfg.max_fix_attempts,
                               cfg.build_timeout, cfg.build_temperature)
        self.packages = packages
        self.agent_key = short_hash([self.llm.name, cfg.max_steps, cfg.send_images, cfg.diagnosis_temperature,
                                     PROMPT_VERSION, self.task.description])

    # ================================================================== state
    @property
    def round(self) -> int:
        return self.state["round"]

    def _path(self, *p) -> str:
        return os.path.join(self.workdir, *p)

    def _init_state(self, cases, seed_tools):
        cfg = self.cfg
        if os.path.exists(self._path("state.json")):
            with open(self._path("state.json"), encoding="utf-8") as f:
                self.state = json.load(f)
            if seed_tools:
                log.warning("Resuming from %s: seed_tools are ignored.", self.workdir)
            missing = [i for ids in self.state["splits"].values() for i in ids if i not in self.cases]
            if missing:
                raise ValueError(f"{len(missing)} case ids in the saved splits are missing, e.g. {missing[:3]}")
        else:
            splits = split_cases(cases, cfg.split_ratios, cfg.seed)
            cohorts = make_cohorts(splits["trial"], cfg.trial_cohorts, cfg.seed)
            for name in ("train", "discovery"):
                if not splits[name]:
                    raise ValueError(f"The '{name}' split is empty; provide more cases or explicit splits.")
            if any(not c for c in cohorts):
                raise ValueError(f"Not enough trial cases for {cfg.trial_cohorts} non-empty trial cohorts.")
            registry = Registry()
            for p in seed_tools or ():
                tool = self.library.add_seed(p)
                registry.stable[tool.name] = tool.id
                log.info("Seed tool %s", tool.id)
            order = [c.id for c in splits["train"]]
            random.Random(cfg.seed).shuffle(order)
            self.state = {
                "round": 0,
                "splits": {k: [c.id for c in v] for k, v in splits.items()},
                "cohorts": [[c.id for c in co] for co in cohorts],
                "experience": {"order": order, "pos": 0, "epoch": 0},
                "snapshots": {},
                "registry": registry.to_dict(),
            }
            with open(self._path("task.json"), "w", encoding="utf-8") as f:
                json.dump(self.task.to_dict(), f, indent=2)
            self._save()
        with open(self._path("config.json"), "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(cfg), f, indent=2)
        self.registry = Registry.from_dict(self.state["registry"])
        self.splits = {k: [self.cases[i] for i in ids] for k, ids in self.state["splits"].items()}
        self.cohorts = [[self.cases[i] for i in ids] for ids in self.state["cohorts"]]

    def _save(self):
        if hasattr(self, "registry"):
            self.state["registry"] = self.registry.to_dict()
        tmp = self._path("state.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=1)
        os.replace(tmp, self._path("state.json"))

    # ============================================================= evaluation
    def agent(self, tool_ids=None, round: int = None) -> MedAgent:
        """The agent with the current stable registry (or a snapshot / explicit tool set)."""
        if tool_ids is None:
            tool_ids = self.state["snapshots"][str(round)] if round is not None else self.registry.stable_ids()
        tools = [self.library.get(i) for i in tool_ids]
        return MedAgent(self.task, self.llm, tools, self.cfg.max_steps, self.cfg.diagnosis_temperature, self.cfg.send_images)

    def diagnose_all(self, tool_ids, cases) -> list:
        """Diagnose cases with the agent built from ``tool_ids`` (cached, parallel)."""
        key = f"{self.agent_key}:{short_hash(sorted(tool_ids))}"
        todo = [c for c in cases if self.cache.get(key, c.id) is None]
        crashed = {}
        if todo:
            agent = self.agent(tool_ids)
            with ThreadPoolExecutor(max(1, self.cfg.workers)) as pool:
                for d in pool.map(agent.diagnose, todo):
                    if d.error and d.error != "no valid prediction":
                        crashed[d.case_id] = d  # not cached, so it is retried next time
                    else:
                        self.cache.put(key, d)
            if crashed:
                first = next(iter(crashed.values())).error
                if len(crashed) > len(todo) / 2:
                    raise RuntimeError(f"{len(crashed)}/{len(todo)} diagnoses failed. First error: {first}")
                log.warning("  %d/%d diagnoses failed (e.g. %s); counted as errors now, retried next time",
                            len(crashed), len(todo), first)
        return [self.cache.get(key, c.id) or crashed[c.id] for c in cases]

    def score(self, cases, diagnoses) -> dict:
        e = [self.task.is_error(c.label, d.prediction) for c, d in zip(cases, diagnoses)]
        s = [self.judge(c, d.prediction)[0] if ei else 0 for c, d, ei in zip(cases, diagnoses, e)]
        return summarize(self.task, [c.label for c in cases], [d.prediction for d in diagnoses], e, s,
                         self.scale, self.fallback)

    def compare(self, ref_ids, tool_id, cohort) -> dict:
        """Paired comparison of H_ref and H_ref (+) T on one cohort: Delta P and Delta C."""
        ref = self.score(cohort, self.diagnose_all(ref_ids, cohort))
        new = self.score(cohort, self.diagnose_all(augment(ref_ids, tool_id), cohort))
        return {"dP": round(new["P"] - ref["P"], 5), "dC": round(new["C"] - ref["C"], 5),
                "P_ref": ref["P"], "P_new": new["P"], "C_ref": ref["C"], "C_new": new["C"]}

    def evaluate(self, split: str = "test", round: int = None, cases=None) -> dict:
        """Metrics of the stable agent (or a round snapshot) on a split. Never used for decisions."""
        cases = cases if cases is not None else self.splits[split]
        if not cases:
            raise ValueError(f"Split '{split}' is empty.")
        ids = self.state["snapshots"][str(round)] if round is not None else self.registry.stable_ids()
        return self.score(cases, self.diagnose_all(ids, cases))

    # ================================================================ the loop
    def run(self, rounds: int) -> list:
        """Run improvement rounds until ``rounds`` rounds have been completed (resumable)."""
        while self.round < rounds:
            self.step()
        return self.history()

    def _next_batch(self) -> list:
        exp, out = self.state["experience"], []
        n = min(self.cfg.batch_size, len(exp["order"]))
        while len(out) < n:
            if exp["pos"] >= len(exp["order"]):
                exp["epoch"] += 1
                random.Random(self.cfg.seed + exp["epoch"]).shuffle(exp["order"])
                exp["pos"] = 0
            out.append(self.cases[exp["order"][exp["pos"]]])
            exp["pos"] += 1
        return out

    def _context(self, proposed) -> dict:
        train = self.splits["train"]
        keys = sorted({k for c in train for k in c.inputs})
        dist = {}
        for c in train:
            dist[str(c.label)] = dist.get(str(c.label), 0) + 1
        resources = (
            f"- Training set: {len(train)} labelled cases (route 'model' may train on them). Input keys: {keys}.\n"
            f"- Labels: {json.dumps(dist) if self.task.is_classification else 'continuous'}.\n"
            f"- Allowed development routes: {list(self.cfg.routes)} (code = deterministic python, compose = "
            f"combine stable tools, model = train a model on the training set).\n"
            f"- Python packages available: {', '.join(self.packages) or 'standard library only'}."
        )
        K = self.cfg.trial_cohorts
        pending = [f"{c.name}: {c.spec.get('description', '')} (under trial, {len(c.trials)}/{K} cohorts)"
                   for c in self.registry.pool]
        pending += [f"{s['tool_name']}: {s['description']} (proposed this round)" for s in proposed]
        rejected = [f"{a['tool'].split('@')[0]}: {a['description']} -- rejected: {a['reason']}"
                    for a in self.registry.archive[-10:]]
        stable = [self.library.get(i) for i in self.registry.stable_ids()]
        return {"stable": stable, "pending": pending, "rejected": rejected, "resources": resources}

    def _training_payload(self):
        train = list(self.splits["train"])
        if self.cfg.max_train_cases and len(train) > self.cfg.max_train_cases:
            train = random.Random(self.cfg.seed).sample(train, self.cfg.max_train_cases)
        payload = [{"id": c.id, "inputs": c.inputs, "label": c.label} for c in train]
        by_label = {}
        for p in payload:  # functional checks cover different labels
            by_label.setdefault(str(p["label"]), []).append(p)
        check, i = [], 0
        while len(check) < min(self.cfg.check_cases, len(payload)):
            for group in by_label.values():
                if i < len(group) and len(check) < self.cfg.check_cases:
                    check.append(group[i])
            i += 1
        return payload, check

    def step(self) -> dict:
        """One improvement round."""
        cfg, r, reg = self.cfg, self.round, self.registry
        rec = {"round": r, "stable_start": reg.stable_ids()}
        self.state["snapshots"][str(r)] = reg.stable_ids()

        # 1-2. diagnose an experience batch, then prioritise and cluster the failures
        batch = self._next_batch()
        diagnoses = self.diagnose_all(reg.stable_ids(), batch)
        rec["experience"] = self.score(batch, diagnoses)
        failures = collect_failures(self.task, batch, diagnoses, self.judge, cfg.lam)
        clusters = select_clusters(failures, cfg.max_candidates, cfg.cases_per_cluster, cfg.cluster_failures)
        ex = rec["experience"]
        log.info("Round %d | experience P=%.3f cost=%.1f/100 errors=%d/%d | stable=%d pool=%d archive=%d",
                 r, ex["P"], ex["cost_per_100"], ex["errors"], ex["n"], len(reg.stable), len(reg.pool), len(reg.archive))
        rec["clusters"] = [{"pattern": c.pattern, "size": c.size, "cases": [f.case.id for f in c.failures],
                            "scores": [f.score for f in c.failures], "max_q": c.failures[0].priority} for c in clusters]

        # 3-5. reflection -> capability development -> discovery screening (fast)
        rec["candidates"], proposed = [], []
        train_payload, check_payload = self._training_payload() if clusters else ([], [])
        for cluster in clusters:
            spec = self.reflector.propose(cluster, self._context(proposed))
            entry = {"pattern": cluster.pattern, "spec": spec}
            rec["candidates"].append(entry)
            if not spec:
                log.info("  reflect [%s]: no proposal", cluster.pattern)
                continue
            if spec["tool_name"] in {c.name for c in reg.pool} | {s["tool_name"] for s in proposed}:
                entry["skipped"] = "duplicate of a pending candidate"
                continue
            proposed.append(spec)
            log.info("  reflect [%s] (%d failures, max q=%.2f) -> %s [%s]", cluster.pattern, cluster.size,
                     cluster.failures[0].priority, spec["tool_name"], spec["route"])
            stable_tools = [self.library.get(i) for i in reg.stable_ids()]
            tool, info = self.builder.build(spec, stable_tools, train_payload, check_payload, r)
            entry["build"] = info
            if tool is None:
                log.info("  build %s: FAILED after %d attempts", spec["tool_name"], len(info["attempts"]))
                continue
            known = set(reg.stable_ids()) | {c.tool_id for c in reg.pool} | {a["tool"] for a in reg.archive}
            if tool.id in known:
                entry["skipped"] = f"identical to existing tool {tool.id}"
                continue
            ref = reg.stable_ids()
            res = self.compare(ref, tool.id, self.splits["discovery"])
            cand = Candidate(tool.id, ref, r, res, spec)
            entry["discovery"] = res
            if res["dP"] <= 0:
                reg.reject(cand, r, f"no gain on the discovery cohort (dP={res['dP']:+.3f})")
                log.info("  discovery %s: dP=%+.3f dC=%+.3f -> archived", tool.id, res["dP"], res["dC"])
            elif cfg.immediate_registration:
                reg.register(cand, r, "immediate")
                log.info("  discovery %s: dP=%+.3f dC=%+.3f -> REGISTERED (immediate)", tool.id, res["dP"], res["dC"])
            else:
                reg.pool.append(cand)
                log.info("  discovery %s: dP=%+.3f dC=%+.3f -> experimental pool", tool.id, res["dP"], res["dC"])

        # 6. trials on subsequent cohorts and serial registration (slow)
        rec["trials"], rec["registered"], rec["rejected"] = self._trials(r), [], []
        K = cfg.trial_cohorts
        ready = [c for c in reg.pool if len(c.trials) >= K]
        for cand in sorted(ready, key=lambda c: mean(t["dP"] for t in c.trials), reverse=True):
            reg.pool.remove(cand)
            current = reg.stable_ids()
            if reg.key(cand.ref) != reg.key(current):  # registry changed during the trial period
                cand.trials = [dict(self.compare(current, cand.tool_id, co), cohort=k, round=r)
                               for k, co in enumerate(self.cohorts)]
                cand.ref, cand.reevaluated = current, True
            ok, dP, dC = registration_decision(cand.trials, self.delta)
            if ok:
                reg.register(cand, r, "trials", dP, dC)
                rec["registered"].append(cand.tool_id)
                log.info("  REGISTERED %s: mean dP=%+.3f mean dC=%+.3f over %d cohorts", cand.tool_id, dP, dC, K)
            else:
                why = f"mean dP={dP:+.3f} (margin {self.delta}) mean dC={dC:+.3f} over {K} trial cohorts"
                reg.reject(cand, r, why, dP, dC)
                rec["rejected"].append(cand.tool_id)
                log.info("  rejected %s: %s", cand.tool_id, why)

        # bookkeeping
        self.state["round"] = r + 1
        self.state["snapshots"][str(r + 1)] = reg.stable_ids()
        rec["stable_end"] = reg.stable_ids()
        if cfg.eval_every and (r + 1) % cfg.eval_every == 0 and self.splits["test"]:
            rec["test"] = self.evaluate("test")
            log.info("  test (monitoring only): P=%.3f cost=%.1f/100", rec["test"]["P"], rec["test"]["cost_per_100"])
        self._save()
        with open(self._path("rounds", f"round_{r:03d}.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=1, default=str)
        summary = {"round": r, "experience": rec["experience"], "stable": rec["stable_end"],
                   "pool": [c.tool_id for c in reg.pool], "n_archive": len(reg.archive),
                   "registered": rec["registered"], "test": rec.get("test")}
        with open(self._path("history.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")
        return rec

    def _trials(self, r: int) -> list:
        """Advance each pooled candidate on its next trial cohort(s), against its reference agent."""
        out = []
        for cand in self.registry.pool:
            if cand.round == r:  # trials use cohorts *subsequent* to discovery
                continue
            for _ in range(self.cfg.trials_per_round):
                k = len(cand.trials)
                if k >= self.cfg.trial_cohorts:
                    break
                res = dict(self.compare(cand.ref, cand.tool_id, self.cohorts[k]), cohort=k, round=r)
                cand.trials.append(res)
                out.append({"tool": cand.tool_id, **res})
                log.info("  trial %d/%d %s: dP=%+.3f dC=%+.3f", k + 1, self.cfg.trial_cohorts, cand.tool_id,
                         res["dP"], res["dC"])
        return out

    # ================================================================ reports
    def history(self) -> list:
        if not os.path.exists(self._path("history.jsonl")):
            return []
        with open(self._path("history.jsonl"), encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def status(self) -> dict:
        reg = self.registry
        return {
            "round": self.round,
            "stable": [{"tool": t.id, "route": t.route, "description": t.description}
                       for t in (self.library.get(i) for i in reg.stable_ids())],
            "pool": [{"tool": c.tool_id, "trials": f"{len(c.trials)}/{self.cfg.trial_cohorts}",
                      "discovery_dP": c.discovery.get("dP")} for c in reg.pool],
            "archived": len(reg.archive),
            "registrations": [{"round": e["round"], "tool": e["tool"], "mean_dP": e.get("mean_dP")} for e in reg.events],
        }


def load_agent(workdir: str, llm, round: int = None, max_steps: int = 3) -> MedAgent:
    """Load the evolved agent from a run folder without the data (for deployment / inspection)."""
    with open(os.path.join(workdir, "state.json"), encoding="utf-8") as f:
        state = json.load(f)
    task = Task.from_json(os.path.join(workdir, "task.json"))
    ids = state["snapshots"][str(round)] if round is not None else sorted(state["registry"]["stable"].values())
    library = Library(os.path.join(workdir, "library"))
    return MedAgent(task, make_llm(llm), [library.get(i) for i in ids], max_steps)
