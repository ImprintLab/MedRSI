"""Helpers shared across modules (JSON parsing, code extraction, hashing)."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re

IMAGE_EXTS = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def is_image_path(value) -> bool:
    """True if `value` is a path to an existing image file that a VLM can read."""
    return (
        isinstance(value, str)
        and os.path.splitext(value)[1].lower() in IMAGE_EXTS
        and os.path.isfile(value)
    )


def parse_json(text: str):
    """Extract the first JSON object from an LLM reply (tolerates code fences and prose)."""
    if not isinstance(text, str):
        return None
    text = text.strip()
    candidates = [text]
    candidates += re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for c in candidates:
        try:
            obj = json.loads(c)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def extract_code(text: str) -> str:
    """Return the (longest) python code block in an LLM reply, or the reply itself.

    An unterminated final fence (e.g. a truncated reply) is also accepted."""
    text = text or ""
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)(?:```|\Z)", text, flags=re.S)
    return max(blocks, key=len).strip() + "\n" if blocks else text.strip() + "\n"


def slug(name) -> str:
    """Turn a proposed tool name into a safe python-identifier-like snake_case name."""
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(name or "")).strip("_").lower()
    if s and s[0].isdigit():
        s = "t_" + s
    return s[:60]


def hash_tool_dir(path: str) -> str:
    """Version hash of a tool package: its code plus every artifact (e.g. trained weights)."""
    h = hashlib.sha256()
    files = []
    for root, _, names in os.walk(path):
        for n in names:
            if n == "meta.json" or "__pycache__" in root:
                continue
            files.append(os.path.relpath(os.path.join(root, n), path))
    for rel in sorted(files):
        h.update(rel.replace(os.sep, "/").encode())
        with open(os.path.join(path, rel), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def short_hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, default=str).encode()).hexdigest()[:12]


def to_jsonable(x, max_str: int = 2000, max_list: int = 32):
    """Convert tool outputs to compact JSON-safe evidence (arrays become shape summaries)."""
    if x is None or isinstance(x, (bool, int)):
        return x
    if isinstance(x, float):
        return round(x, 6) if math.isfinite(x) else str(x)
    if isinstance(x, str):
        return x if len(x) <= max_str else x[:max_str] + "...[truncated]"
    if isinstance(x, dict):
        return {str(k): to_jsonable(v, max_str, max_list) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        x = list(x)
        out = [to_jsonable(v, max_str, max_list) for v in x[:max_list]]
        if len(x) > max_list:
            out.append(f"...(+{len(x) - max_list} more)")
        return out
    if hasattr(x, "shape") and hasattr(x, "dtype"):  # numpy / torch arrays and scalars
        shape = tuple(x.shape)
        if shape == ():
            return to_jsonable(x.item(), max_str, max_list)
        if math.prod(shape) <= 16:
            return to_jsonable(x.tolist(), max_str, max_list)
        return f"<array shape={shape} dtype={x.dtype}>"
    if hasattr(x, "item"):
        try:
            return to_jsonable(x.item(), max_str, max_list)
        except Exception:
            pass
    return repr(x)[:200]


def find_images(obj) -> list:
    """All existing image paths mentioned anywhere inside a (JSON-like) object."""
    found = []
    if isinstance(obj, dict):
        for v in obj.values():
            found += find_images(v)
    elif isinstance(obj, list):
        for v in obj:
            found += find_images(v)
    elif is_image_path(obj):
        found.append(obj)
    return found


def mean(xs, default=0.0):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else default
