"""LLM backends.

Every MedRSI component talks to a model through one method::

    llm.chat(messages, temperature=None, tag="agent") -> str

``messages`` is a list of ``{"role": "system"|"user"|"assistant", "content": str,
"images": [paths]}``. ``tag`` names the caller ("agent", "reflect", "build", "judge"); real
backends ignore it, but it lets a ``FunctionLLM`` route requests or a user log them.

Backends:
    OpenAIChat     - any OpenAI-compatible endpoint (OpenAI, Azure-compatible gateways, vLLM,
                     Ollama, SGLang, OpenRouter, DeepSeek, Qwen, ...).
    AnthropicChat  - Claude models via the official Anthropic SDK.
    FunctionLLM    - wrap any python callable ``fn(messages, tag) -> str`` (local HF models,
                     custom inference code, ...).

``make_llm("openai:gpt-4o")``, ``make_llm("anthropic:claude-opus-5")``,
``make_llm("openai:Qwen2.5-VL-72B@http://localhost:8000/v1")`` or
``make_llm("py:my_module:MyModel")`` build one from a short spec string.
"""
from __future__ import annotations

import base64
import importlib
import importlib.util
import inspect
import logging
import os
import sys
import threading

from .utils import IMAGE_EXTS

log = logging.getLogger("medrsi")
_USAGE_LOCK = threading.Lock()


def _encode_image(path: str):
    mime = IMAGE_EXTS.get(os.path.splitext(path)[1].lower(), "image/png")
    with open(path, "rb") as f:
        return mime, base64.standard_b64encode(f.read()).decode("utf-8")


class LLM:
    name = "llm"

    def chat(self, messages, temperature=None, tag: str = "") -> str:  # pragma: no cover
        raise NotImplementedError

    @property
    def usage(self) -> dict:
        """Accumulated token usage of this backend: calls, input_tokens, output_tokens."""
        return dict(getattr(self, "_usage", {"calls": 0, "input_tokens": 0, "output_tokens": 0}))

    def _record(self, input_tokens: int, output_tokens: int):
        with _USAGE_LOCK:
            u = self.__dict__.setdefault("_usage", {"calls": 0, "input_tokens": 0, "output_tokens": 0})
            u["calls"] += 1
            u["input_tokens"] += input_tokens or 0
            u["output_tokens"] += output_tokens or 0


class OpenAIChat(LLM):
    """OpenAI-compatible chat completions. Set ``base_url`` (or OPENAI_BASE_URL) for local servers."""

    def __init__(self, model: str = "gpt-4o", base_url=None, api_key=None, max_tokens=None, **client_kwargs):
        from openai import OpenAI

        client_kwargs.setdefault("max_retries", 5)
        if base_url and not api_key and not os.environ.get("OPENAI_API_KEY"):
            api_key = "EMPTY"  # local servers usually ignore the key
        self.client = OpenAI(base_url=base_url, api_key=api_key, **client_kwargs)
        self.model, self.max_tokens = model, max_tokens
        self.name = f"openai:{model}"
        self._temperature_ok = True

    def chat(self, messages, temperature=None, tag: str = "") -> str:
        msgs = []
        for m in messages:
            images = m.get("images") or []
            if images:
                content = [{"type": "text", "text": m["content"]}]
                for p in images:
                    mime, data = _encode_image(p)
                    content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
            else:
                content = m["content"]
            msgs.append({"role": m["role"], "content": content})
        kwargs = {"model": self.model, "messages": msgs}
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        if temperature is not None and self._temperature_ok:
            kwargs["temperature"] = temperature
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as e:  # some reasoning models reject a custom temperature
            if "temperature" not in kwargs or "temperature" not in str(e).lower():
                raise
            self._temperature_ok = False
            kwargs.pop("temperature")
            resp = self.client.chat.completions.create(**kwargs)
        if getattr(resp, "usage", None):
            self._record(resp.usage.prompt_tokens, resp.usage.completion_tokens)
        return resp.choices[0].message.content or ""


class AnthropicChat(LLM):
    """Claude via the official Anthropic SDK.

    Responses are streamed, so long outputs (e.g. tool code after extended thinking) do not hit
    HTTP timeouts. Current Claude models choose their own sampling settings, so ``temperature``
    is ignored. ``effort`` ("low" | "medium" | "high" | "xhigh" | "max") trades thinking depth
    for cost; None uses the model default. ``fallbacks="default"`` enables server-side refusal
    fallbacks (a request declined by a safety classifier is re-run on a fallback model); pass
    ``fallbacks=None`` to disable it, e.g. on Bedrock / Vertex / proxies without the beta.
    """

    def __init__(self, model: str = "claude-opus-5", max_tokens: int = 64000, effort=None,
                 fallbacks="default", **client_kwargs):
        import anthropic

        client_kwargs.setdefault("max_retries", 5)
        self.client = anthropic.Anthropic(**client_kwargs)
        self.model, self.max_tokens, self.effort, self.fallbacks = model, max_tokens, effort, fallbacks
        self.name = f"anthropic:{model}" + (f":{effort}" if effort else "")

    def chat(self, messages, temperature=None, tag: str = "") -> str:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        msgs = []
        for m in messages:
            if m["role"] == "system":
                continue
            blocks = []
            for p in m.get("images") or []:
                mime, data = _encode_image(p)
                blocks.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}})
            blocks.append({"type": "text", "text": m["content"] or "(empty)"})
            if msgs and msgs[-1]["role"] == m["role"]:  # the API expects alternating roles
                msgs[-1]["content"] += blocks
            else:
                msgs.append({"role": m["role"], "content": blocks})
        kwargs = {"model": self.model, "max_tokens": self.max_tokens, "messages": msgs}
        if system:
            kwargs["system"] = system
        extra = {"output_config": {"effort": self.effort}} if self.effort else {}
        if self.fallbacks:
            extra["fallbacks"] = self.fallbacks
            stream = self.client.beta.messages.stream(
                betas=["server-side-fallback-2026-07-01"], extra_body=extra, **kwargs)
        else:
            stream = self.client.messages.stream(extra_body=extra or None, **kwargs)
        with stream as s:
            resp = s.get_final_message()
        self._record(resp.usage.input_tokens, resp.usage.output_tokens)
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"Claude declined the request ({getattr(resp, 'stop_details', None)})")
        if resp.stop_reason == "max_tokens":
            log.warning("Claude reply hit max_tokens=%d and is truncated.", self.max_tokens)
        return "".join(b.text for b in resp.content if b.type == "text")


class FunctionLLM(LLM):
    """Wrap a python callable ``fn(messages)`` or ``fn(messages, tag)`` returning a string."""

    def __init__(self, fn, name: str = None):
        self.fn = fn
        self.name = name or f"py:{getattr(fn, '__name__', type(fn).__name__)}"
        target = fn if inspect.isfunction(fn) or inspect.ismethod(fn) else getattr(fn, "__call__", fn)
        try:
            self._n_args = len(inspect.signature(target).parameters)
        except (TypeError, ValueError):
            self._n_args = 2

    def chat(self, messages, temperature=None, tag: str = "") -> str:
        return self.fn(messages, tag) if self._n_args >= 2 else self.fn(messages)


def _load_object(ref: str):
    """'pkg.module:attr' or 'path/to/file.py:attr' -> python object."""
    mod_ref, _, attr = ref.rpartition(":")
    if mod_ref.endswith(".py"):
        spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(mod_ref))[0], mod_ref)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        module = importlib.import_module(mod_ref)
    return getattr(module, attr)


def make_llm(spec) -> LLM:
    """Build a backend from a spec string (or return ``spec`` unchanged if it already is one)."""
    if isinstance(spec, LLM):
        return spec
    if callable(spec):
        return FunctionLLM(spec)
    provider, _, rest = str(spec).partition(":")
    if not rest:  # bare model name
        provider, rest = ("anthropic" if spec.startswith("claude") else "openai"), spec
    if provider == "openai":
        model, _, base_url = rest.partition("@")
        return OpenAIChat(model, base_url=base_url or None)
    if provider == "anthropic":
        return AnthropicChat(rest)
    if provider == "py":
        obj = _load_object(rest)
        obj = obj() if inspect.isclass(obj) else obj
        return obj if isinstance(obj, LLM) else FunctionLLM(obj, name=f"py:{rest}")
    raise ValueError(f"Unknown LLM spec '{spec}'. Use openai:<model>[@base_url], anthropic:<model> or py:<module>:<callable>.")
