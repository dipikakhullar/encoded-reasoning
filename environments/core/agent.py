"""One agent: a system prompt, a persistent chat history, an LLM, a ToolSet.

run_turn() lets the model call non-terminal tools freely, and ends the turn
on the first successful terminal tool (send_message, a task decision tool) or
on a text-only reply.
"""

from __future__ import annotations

MAX_TOOL_CALLS_PER_TURN = 10
TRACE_RESULT_CHARS = 600  # tool results are truncated in the step trace (full text stays in history)


class Agent:
    def __init__(self, agent_id: str, llm, system_prompt: str, tools,
                 max_tool_calls: int = MAX_TOOL_CALLS_PER_TURN):
        self.agent_id = agent_id
        self.llm = llm
        self.tools = tools
        self.max_tool_calls = max_tool_calls
        self.history: list[dict] = [{"role": "system", "content": system_prompt}]

    def run_turn(self, user_message: str, label: str, extra_tools=None,
                 allow_tools: bool = True, required_tool: str | None = None) -> dict:
        """With required_tool set (a decision phase), the turn ends only when
        that tool succeeds; other terminal tools (send_message) just continue,
        and a text-only reply gets one reminder before the turn is abandoned."""
        self.tools.begin_turn(label)
        self.tools.set_extra(extra_tools)
        self.history.append({"role": "user", "content": user_message})

        text = None
        terminal_tool = None
        calls_made: list[str] = []
        steps: list[dict] = []  # one per model call
        reminded = False

        for _ in range(self.max_tool_calls):
            reply = self.llm.complete(
                self.history, self.tools.schemas() if allow_tools else None)
            self.history.append(reply.as_history_message())
            step = {"reasoning": reply.reasoning_summary, "text": reply.content,
                    "reasoning_tokens": reply.usage.get("reasoning_tokens"), "calls": []}
            steps.append(step)

            if not reply.tool_calls:
                text = reply.content
                if required_tool and not reminded:
                    reminded = True
                    self.history.append({"role": "user", "content":
                        f"You have not called {required_tool} yet. Call it now to "
                        f"close this phase."})
                    continue
                break

            ended = False
            for tc in reply.tool_calls:
                result, terminal = self.tools.dispatch(tc.name, tc.args)
                calls_made.append(tc.name)
                step["calls"].append({"name": tc.name, "arguments": tc.arguments_raw,
                                      "result": (result or "")[:TRACE_RESULT_CHARS]})
                self.history.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": result})
                if terminal and (required_tool is None or tc.name == required_tool):
                    terminal_tool = tc.name
                    ended = True
            if ended:
                break

        self.tools.set_extra(None)
        return {"agent": self.agent_id, "label": label, "text": text,
                "terminal_tool": terminal_tool, "tool_calls": calls_made, "steps": steps}


def format_turn_log(header: str, out: dict) -> str:
    """Plain-text record of one turn for a per-agent daily log: for each model
    call, the reasoning summary, any text reply, and each tool call with its
    (truncated) result."""
    lines = [f"===== {header} ====="]
    for i, st in enumerate(out.get("steps") or [], 1):
        lines.append(f"--- call {i} (reasoning tokens: {st.get('reasoning_tokens')})")
        lines.append("[reasoning summary]")
        lines.append(st.get("reasoning") or "(none returned)")
        if st.get("text"):
            lines.append("[text reply]")
            lines.append(st["text"])
        for c in st.get("calls") or []:
            lines.append(f"[action] {c['name']} {c['arguments']}")
            lines.append(f"  -> {c['result']}")
    return "\n".join(lines) + "\n\n"

