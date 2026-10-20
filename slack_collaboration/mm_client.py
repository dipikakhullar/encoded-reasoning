"""Minimal async Mattermost REST client for agent messaging.

One ``MMClient`` per agent, authenticated with that agent's bot token (see
``bootstrap.py``). Agents post to a shared channel (the board) or message one other agent
privately through the pair's ``w2w-<a>-<b>`` channel, and poll for new posts
with ``since``. Pair channels are made by ``bootstrap.py``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import aiohttp


@dataclass
class Post:
    id: str
    t: float              # seconds since epoch
    sender: str           # agent name (username without the ``<team>-`` prefix)
    channel_id: str
    text: str
    props: dict[str, Any]


class MMClient:
    def __init__(self, url: str, token: str, team: str, session: aiohttp.ClientSession) -> None:
        self.api = url.rstrip("/") + "/api/v4"
        self.team_name = team
        self.s = session
        self.h = {"Authorization": f"Bearer {token}"}
        self.me: dict[str, Any] = {}
        self.team_id = ""
        self._names: dict[str, str] = {}   # user id -> username

    async def _req(self, method: str, path: str, **kw: Any) -> Any:
        async with self.s.request(method, self.api + path, headers=self.h, **kw) as r:
            body = await r.json(content_type=None)
            if r.status >= 400:
                raise RuntimeError(f"{method} {path} -> {r.status}: {body.get('message', body)}")
            return body

    async def login(self) -> "MMClient":
        self.me = await self._req("GET", "/users/me")
        self.team_id = (await self._req("GET", f"/teams/name/{self.team_name}"))["id"]
        return self

    async def username(self, user_id: str) -> str:
        if user_id not in self._names:
            name = (await self._req("GET", f"/users/{user_id}"))["username"]
            self._names[user_id] = name.removeprefix(f"{self.team_name}-")
        return self._names[user_id]

    async def channel_id(self, name: str) -> str:
        return (await self._req("GET", f"/teams/{self.team_id}/channels/name/{name}"))["id"]

    @property
    def agent(self) -> str:
        return self.me["username"].removeprefix(f"{self.team_name}-")

    async def dm_channel_id(self, other: str) -> str:
        return await self.channel_id(pair_channel(self.agent, other))

    async def post(self, channel_id: str, text: str, props: dict[str, Any] | None = None,
                   root_id: str = "") -> Post:
        p = await self._req("POST", "/posts", json={"channel_id": channel_id, "message": text,
                                                    "props": props or {}, "root_id": root_id})
        return await self._to_post(p)

    async def dm(self, other: str, text: str, props: dict[str, Any] | None = None) -> Post:
        return await self.post(await self.dm_channel_id(other), text, props)

    async def read(self, channel_id: str, since: float = 0.0) -> list[Post]:
        """Posts in a channel created after ``since`` (epoch seconds), oldest first."""
        r = await self._req("GET", f"/channels/{channel_id}/posts",
                            params={"since": int(since * 1000), "per_page": 200})
        posts = [r["posts"][i] for i in r["order"]]
        posts = [p for p in posts if p["create_at"] > since * 1000 and not p.get("type")]  # skip system posts
        return [await self._to_post(p) for p in sorted(posts, key=lambda p: p["create_at"])]

    async def my_channels(self) -> list[dict[str, Any]]:
        return await self._req("GET", f"/users/me/teams/{self.team_id}/channels")

    async def read_all(self, since: float = 0.0) -> list[Post]:
        """Everything this agent can see (board + its DMs) since ``since``, oldest first."""
        out: list[Post] = []
        for c in await self.my_channels():
            out += await self.read(c["id"], since)
        return sorted(out, key=lambda p: p.t)

    async def _to_post(self, p: dict[str, Any]) -> Post:
        return Post(id=p["id"], t=p["create_at"] / 1000, sender=await self.username(p["user_id"]),
                    channel_id=p["channel_id"], text=p["message"], props=p.get("props") or {})


def pair_channel(a: str, b: str) -> str:
    return "w2w-" + "-".join(sorted((a, b)))


def now() -> float:
    return time.time()
