"""One agent: a system prompt, a persistent chat history, an LLM, a ToolSet.

run_turn() lets the model call non-terminal tools freely, and ends the turn
on the first successful terminal tool (send_message, a task decision tool) or
on a text-only reply.
"""

from __future__ import annotations

MAX_TOOL_CALLS_PER_TURN = 10


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
        reminded = False

        for _ in range(self.max_tool_calls):
            reply = self.llm.complete(
                self.history, self.tools.schemas() if allow_tools else None)
            self.history.append(reply.as_history_message())

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
                self.history.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": result})
                if terminal and (required_tool is None or tc.name == required_tool):
                    terminal_tool = tc.name
                    ended = True
            if ended:
                break

        self.tools.set_extra(None)
        return {"agent": self.agent_id, "label": label, "text": text,
                "terminal_tool": terminal_tool, "tool_calls": calls_made}
