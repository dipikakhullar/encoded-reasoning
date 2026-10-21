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


def _tool_call(call_id: str | None, name: str, raw: str | None) -> ToolCall:
    raw = raw or "{}"
    try:
        args = json.loads(raw)
        if not isinstance(args, dict):
            args = None
    except json.JSONDecodeError:
        args = None
    return ToolCall(id=call_id or f"call_{uuid.uuid4().hex[:12]}",
                    name=name, arguments_raw=raw, args=args)


@dataclass
class Reply:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0
    reasoning_summary: str | None = None  # provider's summary of hidden reasoning (Responses API only)

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
                 reasoning_effort: str | None = None, timeout: float = 180.0,
                 prompt_cache_key: str | None = None):
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
        self.api = spec.get("api", "chat")  # "responses" for models that only take tools via /v1/responses
        self.reasoning_effort = reasoning_effort
        # OpenAI prompt caching: the key routes an agent's calls to the same cache;
        # retention ("24h") is set per model in models.yaml. Only sent when given.
        if prompt_cache_key and len(prompt_cache_key) > 64:     # OpenAI rejects keys over 64 chars
            import hashlib
            prompt_cache_key = hashlib.sha1(prompt_cache_key.encode()).hexdigest()
        self.prompt_cache_key = prompt_cache_key if spec.get("prompt_cache") else None
        self.prompt_cache_retention = spec.get("prompt_cache_retention")
        self.price = spec.get("price") or {}  # USD per 1M tokens: input, cached_input, output
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.calls = 0
        self.prompt_tokens = 0
        self.cached_tokens = 0  # part of prompt_tokens served from the provider's prompt cache
        self.completion_tokens = 0
        self.total_latency_s = 0.0

    def complete(self, messages: list[dict], tools: list[dict] | None) -> Reply:
        if self.api == "responses":
            return self._complete_responses(messages, tools)
        kwargs: dict = {"model": self.model_id, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        self._cache_kwargs(kwargs)

        resp, latency = self._with_retries(self.client.chat.completions.create, kwargs)

        choice = resp.choices[0].message
        calls = [_tool_call(tc.id, tc.function.name, tc.function.arguments)
                 for tc in choice.tool_calls or []]

        usage = {}
        if getattr(resp, "usage", None):
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "cached_tokens": getattr(getattr(resp.usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0,
                "completion_tokens": resp.usage.completion_tokens,
            }
        return self._record(Reply(content=choice.content, tool_calls=calls,
                                  usage=usage, latency_s=latency))

    def _complete_responses(self, messages: list[dict], tools: list[dict] | None) -> Reply:
        """Same contract as complete(), through /v1/responses. Chat-format history
        is translated to input items each call (stateless, store=False)."""
        items: list[dict] = []
        for m in messages:
            if m["role"] == "tool":
                items.append({"type": "function_call_output", "call_id": m["tool_call_id"],
                              "output": m.get("content") or ""})
                continue
            if m.get("content"):
                items.append({"role": m["role"], "content": m["content"]})
            for tc in m.get("tool_calls") or []:
                items.append({"type": "function_call", "call_id": tc["id"],
                              "name": tc["function"]["name"],
                              "arguments": tc["function"]["arguments"]})
        kwargs: dict = {"model": self.model_id, "input": items, "store": False}
        if tools:
            kwargs["tools"] = [{"type": "function", **t["function"]} for t in tools]
        # always request the detailed reasoning summary so it can be logged
        kwargs["reasoning"] = {"summary": "detailed"}
        if self.reasoning_effort:
            kwargs["reasoning"]["effort"] = self.reasoning_effort
        self._cache_kwargs(kwargs)

        resp, latency = self._with_retries(self.client.responses.create, kwargs)

        calls = [_tool_call(o.call_id, o.name, o.arguments)
                 for o in resp.output if o.type == "function_call"]
        summary = "\n\n".join(s.text for o in resp.output if o.type == "reasoning"
                               for s in (o.summary or []) if getattr(s, "text", None))
        usage = {}
        if getattr(resp, "usage", None):
            usage = {"prompt_tokens": resp.usage.input_tokens,
                     "cached_tokens": getattr(getattr(resp.usage, "input_tokens_details", None), "cached_tokens", 0) or 0,
                     "completion_tokens": resp.usage.output_tokens}
        if getattr(resp, "usage", None):
            usage["reasoning_tokens"] = getattr(getattr(resp.usage, "output_tokens_details", None),
                                                "reasoning_tokens", 0) or 0
        return self._record(Reply(content=resp.output_text or None, tool_calls=calls,
                                  usage=usage, latency_s=latency, reasoning_summary=summary or None))

    def _cache_kwargs(self, kwargs: dict) -> None:
        if self.prompt_cache_key:
            kwargs["prompt_cache_key"] = self.prompt_cache_key
            if self.prompt_cache_retention:
                # raw body field: older SDKs (the oasis_social venv) lack the kwarg
                kwargs["extra_body"] = {"prompt_cache_retention": self.prompt_cache_retention}

    def cost_usd(self) -> float | None:
        """List-price cost of this client's calls; None when models.yaml has no price."""
        p = self.price
        if not p:
            return None
        fresh = self.prompt_tokens - self.cached_tokens
        return (fresh * p["input"] + self.cached_tokens * p.get("cached_input", p["input"])
                + self.completion_tokens * p["output"]) / 1e6

    def _record(self, reply: Reply) -> Reply:
        self.prompt_tokens += reply.usage.get("prompt_tokens") or 0
        self.cached_tokens += reply.usage.get("cached_tokens") or 0
        self.completion_tokens += reply.usage.get("completion_tokens") or 0
        self.calls += 1
        self.total_latency_s += reply.latency_s
        return reply

    @staticmethod
    def _with_retries(create, kwargs: dict):
        delay = 2.0
        for attempt in range(6):
            t0 = time.time()
            try:
                resp = create(**kwargs)
                break
            except Exception as exc:  # noqa: BLE001 - retry on transient errors
                status = getattr(exc, "status_code", None)
                transient = status in (408, 409, 425, 429, 500, 502, 503, 504) \
                    or status is None
                if attempt == 5 or not transient:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 30)
        return resp, time.time() - t0

    def stats(self) -> dict:
        return {
            "model": self.slug, "model_id": self.model_id, "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": None if self.cost_usd() is None else round(self.cost_usd(), 4),
            "total_latency_s": round(self.total_latency_s, 1),
        }
