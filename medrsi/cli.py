"""Command line interface.

    medrsi run    --task task.json --data cases.jsonl --llm openai:gpt-4o --rounds 20 --workdir runs/glaucoma
    medrsi eval   --workdir runs/glaucoma [--split test] [--round 10]
    medrsi status --workdir runs/glaucoma
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from .data import load_cases
from .rsi import Config, MedRSI


def _parse_set(items) -> dict:
    out = {}
    for item in items or []:
        key, _, value = item.partition("=")
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def _run_info(workdir: str) -> dict:
    path = os.path.join(workdir, "run.json")
    if not os.path.exists(path):
        sys.exit(f"{path} not found - start a run with `medrsi run` first.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load(args, info=None) -> MedRSI:
    info = info or {}
    config = Config()
    cfg_path = args.config if getattr(args, "config", None) else (
        os.path.join(args.workdir, "config.json") if info else None)
    overrides = {}
    if cfg_path and os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            overrides.update(json.load(f))
    overrides.update(_parse_set(getattr(args, "set", None)))
    for k in ("routes", "split_ratios"):
        if k in overrides:
            overrides[k] = tuple(overrides[k])
    config = Config.from_dict({**config.__dict__, **overrides})
    task = getattr(args, "task", None) or os.path.join(args.workdir, "task.json")
    data = getattr(args, "data", None) or info.get("data")
    llm = getattr(args, "llm", None) or info.get("llm")
    builder_llm = getattr(args, "builder_llm", None) or info.get("builder_llm")
    return MedRSI(task, load_cases(data), llm, args.workdir, config, builder_llm=builder_llm,
                  seed_tools=getattr(args, "seed_tools", None) or ())


def main(argv=None):
    p = argparse.ArgumentParser(prog="medrsi", description="Recursive self-improvement of medical agents.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run (or resume) self-improvement rounds")
    r.add_argument("--task", required=True, help="task.json")
    r.add_argument("--data", required=True, help="cases .jsonl / .json / .csv")
    r.add_argument("--llm", required=True, help="openai:<model>[@base_url] | anthropic:<model> | py:<module>:<obj>")
    r.add_argument("--builder-llm", default=None, help="optional separate model for reflection + tool building")
    r.add_argument("--workdir", default="medrsi_run")
    r.add_argument("--rounds", type=int, default=20)
    r.add_argument("--config", default=None, help="JSON file with Config overrides")
    r.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="Config overrides, e.g. lam=0 trial_cohorts=1")
    r.add_argument("--seed-tools", nargs="*", help="hand-written tools forming the initial registry")

    e = sub.add_parser("eval", help="evaluate the stable agent (or a round snapshot) on a split")
    e.add_argument("--workdir", required=True)
    e.add_argument("--split", default="test")
    e.add_argument("--round", type=int, default=None)
    e.add_argument("--llm", default=None, help="defaults to the model used for the run")

    s = sub.add_parser("status", help="show registered tools, pool and history")
    s.add_argument("--workdir", required=True)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.cmd == "run":
        info = {"data": os.path.abspath(args.data), "llm": args.llm, "builder_llm": args.builder_llm}
        os.makedirs(args.workdir, exist_ok=True)
        with open(os.path.join(args.workdir, "run.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        rsi = _load(args)
        rsi.run(args.rounds)
        print(json.dumps(rsi.status(), indent=2))
    elif args.cmd == "eval":
        rsi = _load(args, _run_info(args.workdir))
        print(json.dumps(rsi.evaluate(args.split, args.round), indent=2))
    elif args.cmd == "status":
        with open(os.path.join(args.workdir, "state.json"), encoding="utf-8") as f:
            state = json.load(f)
        reg = state["registry"]
        print(f"Completed rounds: {state['round']}")
        stable = [f"  {t}" for t in sorted(reg["stable"].values())] or ["  (none)"]
        pool = [f"  {c['tool_id']}  trials={len(c['trials'])}" for c in reg["pool"]] or ["  (none)"]
        print("Stable tools:", *stable, sep="\n")
        print("Experimental pool:", *pool, sep="\n")
        print(f"Archived candidates: {len(reg['archive'])}")
        hist = os.path.join(args.workdir, "history.jsonl")
        if os.path.exists(hist):
            print("\nround  exp_P   cost/100  #stable  registered")
            with open(hist, encoding="utf-8") as f:
                for line in f:
                    h = json.loads(line)
                    print(f"{h['round']:>5}  {h['experience']['P']:.3f}  {h['experience']['cost_per_100']:8.1f}"
                          f"  {len(h['stable']):7d}  {', '.join(h['registered'])}")


if __name__ == "__main__":
    main()
