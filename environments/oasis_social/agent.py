"""CreatorAgent: OASIS's SocialAgent with our daily turn (subclassed here, OASIS
itself is unchanged).

OASIS asks each agent to "pick one" action per step. CreatorAgent instead
gives it the day's budget, notifications, message board and DMs (from
Extras) plus its OASIS feed, and lets it keep acting until it replies without
a tool call or runs out of model calls. Every OASIS action is charged
against the budget before it runs.
"""

from __future__ import annotations

import functools
import json

from camel.messages import BaseMessage
from oasis import SocialAgent


class CreatorAgent(SocialAgent):
    def __init__(self, *args, extras=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.extras = extras
        self.log_backend = kwargs["model"]           # CoreLLMBackend: writes the agent's daily log
        own = {"post_to_board", "send_dm"}          # these charge themselves
        for name, tool in self._internal_tools.items():
            if name in extras.costs and name not in own:
                tool.func = self._charged(name, tool.func)

    def _charged(self, name, func):
        extras, aid = self.extras, self.social_agent_id

        @functools.wraps(func)
        async def wrapped(*a, **k):
            err = extras.charge(aid, name)
            if err:
                return {"success": False, "error": err}
            return await func(*a, **k)
        return wrapped

    async def perform_action_by_llm(self):
        followers = await self.env.get_followers_env()
        follows = await self.env.get_follows_env()
        if self.extras.reach == "followers":
            feed = {"success": True, "posts": self.extras.feed_for(self.social_agent_id)}
            if not feed["posts"]:
                feed = {"success": False, "message": "no recent posts from accounts you follow"}
        else:
            feed = await self.env.action.refresh()  # a feed refresh: pays the authors for views
        if feed.get("success"):
            posts = self.env.posts_env_template.substitute(posts=json.dumps(feed["posts"], indent=4))
        else:
            posts = "After refreshing, there are no posts in your feed."
            self.log_backend._write(  # surface OASIS's swallowed error in the agent log
                f"[feed refresh failed] {feed.get('error') or feed.get('message')}\n")
        content = (f"Day {self.extras.day}. You are agent_{self.social_agent_id}.\n\n"
                   f"{self.extras.observation_block(self.social_agent_id)}\n\n"
                   f"YOUR FEED:\n{followers}\n{follows}\n{posts}")
        try:
            return await self.astep(BaseMessage.make_user_message(role_name="User", content=content))
        except Exception as e:  # noqa: BLE001 - one agent's failure must not end the day
            return e
