"""CAMEL model backend that routes OASIS agents through core.llm.

OASIS agents are CAMEL ChatAgents, and CAMEL only speaks chat completions,
which rejects tools for gpt-6-astra. This backend hands CAMEL's chat-format
messages and tool schemas to core.llm.LLM (Responses API for Astra), so OASIS
runs get the same prompt caching, list-price cost meter, retries and
detailed reasoning summaries as the social / negotiation envs.

Memory window: CAMEL would trim the oldest message on every call once an
agent's memory is full, which changes the prompt prefix each call and defeats
prompt caching (2026-10-21: cache hits fell from 94% to ~0% after day 10).
So CAMEL is given no limit, and this backend trims in blocks instead: when the
history passes `token_limit`, it is cut back to about half, at the start of a
day (a user message), and that cut stays fixed until the next trim.

One backend per agent: its own prompt-cache key, token counters, and daily
log at <out_dir>/agent_logs/agent_<i>.txt (reasoning summary, tool calls,
the tool results that come back on the next call).
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from camel.models import BaseModelBackend
from camel.types import ModelType
from camel.utils import OpenAITokenCounter
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from core.llm import LLM

RESULT_CHARS = 600  # tool results are truncated in the daily log


class CoreLLMBackend(BaseModelBackend):
    def __init__(self, slug: str, agent_id: int, log_dir: Path | None,
                 cache_key: str, token_limit: int = 60_000,
                 reasoning_effort: str | None = None):
        super().__init__(model_type=slug, model_config_dict={})
        self.llm = LLM(slug, reasoning_effort=reasoning_effort, prompt_cache_key=cache_key)
        self.agent_id = agent_id
        self.log_path = (log_dir / f"agent_{agent_id}.txt") if log_dir else None
        self._window_tokens = token_limit
        self._cut = 0          # leading non-system messages dropped by the block trim
        self._counter = OpenAITokenCounter(ModelType.GPT_4O_MINI)
        self.day: int | None = None
        self._logged_day: int | None = None

    @property
    def token_counter(self):
        return self._counter

    @property
    def token_limit(self) -> int:
        return 10_000_000      # CAMEL never trims; _window does, in blocks

    def _window(self, messages: List[Dict]) -> List[Dict]:
        head = messages[:1] if messages and messages[0].get("role") == "system" else []
        body = messages[len(head):]
        if self._cut > len(body):
            self._cut = 0
        sizes = [len(json.dumps(m, default=str)) / 3.5 for m in body]   # ~tokens
        fixed = sum(len(json.dumps(m, default=str)) / 3.5 for m in head)
        if fixed + sum(sizes[self._cut:]) > self._window_tokens:
            i, total = self._cut, fixed + sum(sizes[self._cut:])
            while i < len(body) and total > self._window_tokens / 2:
                total -= sizes[i]
                i += 1
            while i < len(body) and body[i].get("role") != "user":   # cut only at a day start
                i += 1
            if i < len(body):
                self._cut = i
                self._write(f"[memory trimmed: dropped the oldest {i} messages]\n")
        return head + body[self._cut:]

    def _run(self, messages: List[Dict], response_format: Optional[Type[BaseModel]] = None,
             tools: Optional[List[Dict[str, Any]]] = None) -> ChatCompletion:
        self._log_tool_results(messages)
        reply = self.llm.complete(self._window(messages), tools or None)
        self._log_reply(reply)
        return self._to_completion(reply)

    async def _arun(self, messages: List[Dict], response_format: Optional[Type[BaseModel]] = None,
                    tools: Optional[List[Dict[str, Any]]] = None) -> ChatCompletion:
        return await asyncio.to_thread(self._run, messages, response_format, tools)

    def _to_completion(self, reply) -> ChatCompletion:
        calls = [{"id": tc.id, "type": "function",
                  "function": {"name": tc.name, "arguments": tc.arguments_raw}}
                 for tc in reply.tool_calls]
        u = reply.usage
        return ChatCompletion.model_validate({
            "id": f"core-{uuid.uuid4().hex[:12]}", "object": "chat.completion",
            "created": int(time.time()), "model": self.llm.model_id,
            "choices": [{"index": 0, "finish_reason": "tool_calls" if calls else "stop",
                         "message": {"role": "assistant", "content": reply.content,
                                     "tool_calls": calls or None}}],
            "usage": {"prompt_tokens": u.get("prompt_tokens", 0),
                      "completion_tokens": u.get("completion_tokens", 0),
                      "total_tokens": u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)},
        })

    # -- daily log ---------------------------------------------------------

    def _write(self, text: str) -> None:
        if self.log_path is None:
            return
        if self.day is not None and self.day != self._logged_day:
            text = f"===== Day {self.day} =====\n" + text
            self._logged_day = self.day
        with self.log_path.open("a") as fh:
            fh.write(text)

    def _log_tool_results(self, messages: List[Dict]) -> None:
        """Tool results for the previous call arrive as trailing role=tool messages."""
        tail = []
        for m in reversed(messages):
            if m.get("role") != "tool":
                break
            tail.append(m)
        for m in reversed(tail):
            self._write(f"  -> {str(m.get('content') or '')[:RESULT_CHARS]}\n")

    def _log_reply(self, reply) -> None:
        lines = [f"--- call (reasoning tokens: {reply.usage.get('reasoning_tokens')})",
                 "[reasoning summary]", reply.reasoning_summary or "(none returned)"]
        if reply.content:
            lines += ["[text reply]", reply.content]
        lines += [f"[action] {tc.name} {tc.arguments_raw}" for tc in reply.tool_calls]
        self._write("\n".join(lines) + "\n")
