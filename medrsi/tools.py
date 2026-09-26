"""Tools: versioned, inspectable capability packages.

A tool is a plain python file with a ``run`` function (and optionally ``train``)::

    DESCRIPTION = "Vertical cup-to-disc ratio. Returns {'vcdr': float, 'quality': str}."

    def run(inputs, ctx):             # inputs: the patient's input dict
        seg = ctx.call_tool("segment_disc_cup")   # compose other stable tools
        ...
        return {"vcdr": 0.62}         # JSON-like evidence for the agent

    def train(cases, ctx):            # optional (model-development route)
        ...                           # save weights into ctx.artifact_dir

On disk every version lives in its own frozen folder ``library/<name>-<hash8>/`` holding
``tool.py``, ``meta.json`` and ``artifacts/``. The hash covers code and artifacts, so a
tool id ``name@hash8`` always refers to exactly one executable package.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from dataclasses import dataclass, field

from .utils import hash_tool_dir

_MODULES: dict = {}
_LOCK = threading.Lock()


def load_module(path: str, name: str):
    """Import a python file as a module (cached, thread-safe)."""
    with _LOCK:
        if path not in _MODULES:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _MODULES[path] = module
        return _MODULES[path]


def call_run(fn, inputs, ctx):
    """Call ``run(inputs, ctx)``; hand-written tools may also define ``run(inputs)``."""
    try:
        n = len(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        n = 2
    return fn(inputs, ctx) if n >= 2 else fn(inputs)


class ToolContext:
    """What a tool can see besides the patient inputs."""

    def __init__(self, artifact_dir: str, runner: "ToolRunner" = None, scratch_dir: str = None):
        self.artifact_dir = artifact_dir  # files written by train() (weights, thresholds, ...)
        self.scratch_dir = scratch_dir or tempfile.gettempdir()  # per-diagnosis temp space
        self._runner = runner

    def call_tool(self, name: str, inputs: dict = None) -> dict:
        """Call another stable tool (tool composition). ``inputs`` defaults to this patient."""
        if self._runner is None:
            raise RuntimeError("No other tools are available in this context.")
        return self._runner.call(name, inputs)

    @property
    def available_tools(self) -> list:
        return sorted(self._runner.tools) if self._runner else []


@dataclass
class Tool:
    name: str
    path: str  # package directory
    description: str = ""
    route: str = "code"  # code | compose | model | seed
    hash: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.name}@{self.hash[:8]}"

    @property
    def artifact_dir(self) -> str:
        return os.path.join(self.path, "artifacts")

    @property
    def code_path(self) -> str:
        return os.path.join(self.path, "tool.py")

    def module(self):
        return load_module(self.code_path, f"medrsi_tool_{self.name}_{self.hash[:8]}")

    def run(self, inputs: dict, ctx: ToolContext) -> dict:
        return call_run(self.module().run, inputs, ctx)

    @classmethod
    def from_dir(cls, path: str) -> "Tool":
        with open(os.path.join(path, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        return cls(meta["name"], path, meta.get("description", ""), meta.get("route", "code"), meta["hash"], meta)


class ToolRunner:
    """Runs tools for one patient. Outputs are cached, so composed tools reuse each other."""

    def __init__(self, tools: dict, inputs: dict, scratch_dir: str = None, _stack=()):
        self.tools = tools  # name -> Tool
        self.inputs = inputs
        self.scratch_dir = scratch_dir
        self._cache: dict = {}
        self._stack = list(_stack)

    def call(self, name: str, inputs: dict = None) -> dict:
        if inputs is not None and inputs is not self.inputs:  # a different patient / modified inputs
            return ToolRunner(self.tools, inputs, self.scratch_dir, self._stack).call(name)
        if name in self._cache:
            return self._cache[name]
        if name not in self.tools:
            raise KeyError(f"Unknown tool '{name}'. Available: {sorted(self.tools)}")
        if name in self._stack or len(self._stack) > 8:
            raise RecursionError(f"Circular tool composition: {' -> '.join(self._stack + [name])}")
        tool = self.tools[name]
        self._stack.append(name)
        try:
            out = tool.run(dict(self.inputs), ToolContext(tool.artifact_dir, self, self.scratch_dir))
        finally:
            self._stack.pop()
        out = out if isinstance(out, dict) else {"result": out}
        self._cache[name] = out
        return out


class Library:
    """On-disk store of every tool version ever built (stable, experimental or rejected)."""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self._tools: dict = {}

    def get(self, tool_id: str) -> Tool:
        if tool_id not in self._tools:
            name, _, h = tool_id.partition("@")
            self._tools[tool_id] = Tool.from_dir(os.path.join(self.root, f"{name}-{h}"))
        return self._tools[tool_id]

    def _finalize(self, tmp_dir: str, meta: dict) -> Tool:
        """Freeze a package: hash code + artifacts, move to its versioned folder, write meta."""
        h = hash_tool_dir(tmp_dir)
        meta = dict(meta, hash=h)
        final = os.path.join(self.root, f"{meta['name']}-{h[:8]}")
        if os.path.exists(final):
            shutil.rmtree(tmp_dir)
        else:
            os.replace(tmp_dir, final)
            with open(os.path.join(final, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        tool = Tool.from_dir(final)
        self._tools[tool.id] = tool
        return tool

    def add_seed(self, path: str) -> Tool:
        """Import a hand-written tool (a ``.py`` file, or a folder with ``tool.py`` + ``artifacts/``)."""
        src = os.path.join(path, "tool.py") if os.path.isdir(path) else path
        module = load_module(os.path.abspath(src), f"medrsi_seed_{uuid.uuid4().hex[:8]}")
        name = getattr(module, "NAME", None) or os.path.splitext(os.path.basename(path.rstrip("/\\")))[0]
        description = getattr(module, "DESCRIPTION", None) or (module.__doc__ or "").strip()
        if not hasattr(module, "run"):
            raise ValueError(f"Seed tool {path} must define run(inputs, ctx)")
        tmp = os.path.join(self.root, "_tmp", f"{name}-{uuid.uuid4().hex[:8]}")
        os.makedirs(tmp)
        shutil.copy(src, os.path.join(tmp, "tool.py"))
        if os.path.isdir(os.path.join(path, "artifacts")):
            shutil.copytree(os.path.join(path, "artifacts"), os.path.join(tmp, "artifacts"))
        os.makedirs(os.path.join(tmp, "artifacts"), exist_ok=True)
        return self._finalize(tmp, {"name": name, "description": description, "route": "seed", "source": os.path.abspath(path)})

    def build(self, name, code, description, route, meta, train_cases, check_cases, stable_tools, timeout=3600):
        """Package a candidate, train it if it defines ``train`` and run functional checks.

        Training and checks run in a subprocess with a timeout, so a crashing or hanging
        candidate cannot take down the self-improvement loop. Returns ``(tool or None, report)``.
        """
        tmp = os.path.join(self.root, "_tmp", f"{name}-{uuid.uuid4().hex[:8]}")
        os.makedirs(os.path.join(tmp, "artifacts"))
        with open(os.path.join(tmp, "tool.py"), "w", encoding="utf-8") as f:
            f.write(code)
        job = {
            "tool_dir": tmp,
            "route": route,
            "train_cases": train_cases,
            "check_cases": check_cases,
            "stable": [t.path for t in stable_tools],
            "report": os.path.join(tmp, "_report.json"),
        }
        job_path = os.path.join(tmp, "_job.json")
        with open(job_path, "w", encoding="utf-8") as f:
            json.dump(job, f, default=str)
        env = dict(os.environ)
        pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env["PYTHONPATH"] = pkg_parent + os.pathsep + env.get("PYTHONPATH", "")
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "medrsi._worker", job_path],
                capture_output=True, text=True, timeout=timeout, env=env, cwd=tmp,
            )
            if os.path.exists(job["report"]):
                with open(job["report"], encoding="utf-8") as f:
                    report = json.load(f)
            else:
                report = {"ok": False, "error": (proc.stderr or proc.stdout or "worker crashed")[-3000:]}
        except subprocess.TimeoutExpired:
            report = {"ok": False, "error": f"Training/checks exceeded the time limit of {timeout}s."}
        for f in ("_job.json", "_report.json"):
            if os.path.exists(os.path.join(tmp, f)):
                os.remove(os.path.join(tmp, f))
        if not report.get("ok"):
            shutil.rmtree(tmp, ignore_errors=True)
            return None, report
        meta = dict(meta, name=name, description=description, route=route, check_outputs=report.get("outputs", []))
        return self._finalize(tmp, meta), report
