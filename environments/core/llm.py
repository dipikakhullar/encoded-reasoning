"""One client path for every model: OpenAI-compatible chat completions with
tools, through the gateway named in models.yaml (or a per-model base_url).

Reply objects are normalized so the rest of the harness never sees provider
quirks (missing ids, reasoning fields, non-JSON tool arguments).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = ROOT / "models.yaml"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    return yaml.safe_load(path.read_text())


def enabled_models(registry: dict) -> list[str]:
    return [s for s, m in registry["models"].items() if m.get("enabled", True)]


def _first_env(names) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None


@dataclass
class ToolCall:
    id: str
    name: str
    arguments_raw: str
    args: dict | None  # None when the model emitted invalid JSON


@dataclass
class Reply:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0

    def as_history_message(self) -> dict:
        msg: dict = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments_raw},
                }
                for tc in self.tool_calls
            ]
        return msg


class LLM:
    def __init__(self, slug: str, registry: dict | None = None,
                 reasoning_effort: str | None = None, timeout: float = 180.0):
        from openai import OpenAI

        registry = registry or load_registry()
        if slug not in registry["models"]:
            raise KeyError(f"unknown model slug {slug!r}; see models.yaml")
        spec = registry["models"][slug]
        if not spec.get("enabled", True):
            raise RuntimeError(f"model {slug!r} is disabled in models.yaml: "
                               f"{spec.get('notes', '')}")

        base_url = spec.get("base_url") or registry["gateway"]["base_url"]
        key_envs = spec.get("api_key_env") or registry["gateway"]["api_key_env"]
        api_key = _first_env(key_envs) or "not-needed"

        self.slug = slug
        self.model_id = spec["id"]
        self.reasoning_effort = reasoning_effort
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_latency_s = 0.0

    def complete(self, messages: list[dict], tools: list[dict] | None) -> Reply:
        kwargs: dict = {"model": self.model_id, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort

        delay = 2.0
        for attempt in range(6):
            t0 = time.time()
            try:
                resp = self.client.chat.completions.create(**kwargs)
                break
            except Exception as exc:  # noqa: BLE001 - retry on transient errors
                status = getattr(exc, "status_code", None)
                transient = status in (408, 409, 425, 429, 500, 502, 503, 504) \
                    or status is None
                if attempt == 5 or not transient:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 30)
        latency = time.time() - t0

        choice = resp.choices[0].message
        calls: list[ToolCall] = []
        for tc in choice.tool_calls or []:
            raw = tc.function.arguments or "{}"
            try:
                args = json.loads(raw)
                if not isinstance(args, dict):
                    args = None
            except json.JSONDecodeError:
                args = None
            calls.append(ToolCall(
                id=tc.id or f"call_{uuid.uuid4().hex[:12]}",
                name=tc.function.name, arguments_raw=raw, args=args,
            ))

        usage = {}
        if getattr(resp, "usage", None):
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            }
            self.prompt_tokens += resp.usage.prompt_tokens or 0
            self.completion_tokens += resp.usage.completion_tokens or 0
        self.calls += 1
        self.total_latency_s += latency
        return Reply(content=choice.content, tool_calls=calls,
                     usage=usage, latency_s=latency)

    def stats(self) -> dict:
        return {
            "model": self.slug, "model_id": self.model_id, "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_latency_s": round(self.total_latency_s, 1),
        }
