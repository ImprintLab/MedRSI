"""Subprocess worker used by ``Library.build``: train a candidate tool and run functional checks.

Usage: ``python -m medrsi._worker job.json`` (writes the report path given in the job).
"""
import json
import os
import shutil
import sys
import tempfile
import time
import traceback

from medrsi.tools import Tool, ToolContext, ToolRunner, call_run, load_module
from medrsi.utils import to_jsonable


def main(job_path: str) -> None:
    with open(job_path, encoding="utf-8") as f:
        job = json.load(f)
    report = {"ok": False, "trained": False, "outputs": [], "error": None}
    scratch = tempfile.mkdtemp(prefix="medrsi_check_")
    try:
        stable = {t.name: t for t in (Tool.from_dir(p) for p in job["stable"])}
        artifact_dir = os.path.join(job["tool_dir"], "artifacts")
        module = load_module(os.path.join(job["tool_dir"], "tool.py"), "medrsi_candidate")
        if not callable(getattr(module, "run", None)):
            raise TypeError("The module must define run(inputs, ctx).")
        if job["route"] == "model" and not callable(getattr(module, "train", None)):
            raise TypeError("Route 'model' requires a train(cases, ctx) function that saves into ctx.artifact_dir.")

        if callable(getattr(module, "train", None)):
            t0 = time.time()
            module.train(job["train_cases"], ToolContext(artifact_dir, ToolRunner(stable, {}), scratch))
            report["trained"], report["train_seconds"] = True, round(time.time() - t0, 1)

        n_ok = 0
        for case in job["check_cases"]:
            ctx = ToolContext(artifact_dir, ToolRunner(stable, case["inputs"]), scratch)
            t0 = time.time()
            out = call_run(module.run, dict(case["inputs"]), ctx)
            if not isinstance(out, dict):
                raise TypeError(f"run() must return a dict, got {type(out).__name__}.")
            summary = to_jsonable(out, max_str=300)
            json.dumps(summary)
            report["outputs"].append({"case": case["id"], "output": summary, "seconds": round(time.time() - t0, 3)})
            n_ok += "error" not in out
        if job["check_cases"] and n_ok == 0:
            raise RuntimeError("run() returned an 'error' for every check case: " + json.dumps(report["outputs"])[:1500])
        report["ok"] = True
    except BaseException:  # noqa: B902 - report everything back to the builder
        report["error"] = traceback.format_exc(limit=8)[-4000:]
    shutil.rmtree(scratch, ignore_errors=True)
    with open(job["report"], "w", encoding="utf-8") as f:
        json.dump(report, f)


if __name__ == "__main__":
    main(sys.argv[1])
