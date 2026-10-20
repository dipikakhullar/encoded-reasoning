"""Set up a Mattermost team for one multi-agent run and print per-agent bot tokens.

Creates (idempotently) an admin user, a team, a public ``board`` channel and one
bot account per agent, each added to the team and the board. Bot usernames are
``<team>-<agent>``: usernames and DMs are server-wide, so unprefixed names would
share DM threads across runs. Each pair of agents gets a private channel
``w2w-<a>-<b>`` (members: the pair + the admin as an observer) in place of a DM,
since the web UI never shows bot-to-bot DMs, even to an admin. Writes
``{"url", "team", "board", "tokens": {agent: token}}`` to ``--out``.

    python bootstrap.py --agents manager,worker1,worker2 --team run1 --out tokens.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import aiohttp

from mm_client import pair_channel

ADMIN = os.environ.get("MM_ADMIN", "mmadmin")
ADMIN_PW = os.environ.get("MM_ADMIN_PW")


async def main(a: argparse.Namespace) -> None:
    if not ADMIN_PW:
        raise SystemExit("set MM_ADMIN_PW (the Mattermost admin password)")
    api = a.url.rstrip("/") + "/api/v4"
    async with aiohttp.ClientSession() as s:
        async def req(method: str, path: str, token: str | None = None, ok: tuple[int, ...] = (), **kw):
            h = {"Authorization": f"Bearer {token}"} if token else {}
            async with s.request(method, api + path, headers=h, **kw) as r:
                body = await r.json(content_type=None)
                if r.status >= 400 and r.status not in ok:
                    raise RuntimeError(f"{method} {path} -> {r.status}: {body.get('message', body)}")
                return r, body

        # The first user created on a fresh server becomes system admin.
        await req("POST", "/users", ok=(400, 403),
                  json={"username": ADMIN, "email": f"{ADMIN}@local.test", "password": ADMIN_PW})
        r, _ = await req("POST", "/users/login", json={"login_id": ADMIN, "password": ADMIN_PW})
        tok = r.headers["Token"]

        # Bots need bot accounts and per-user access tokens enabled.
        _, cfg = await req("GET", "/config", tok)
        cfg["ServiceSettings"].update(EnableBotAccountCreation=True, EnableUserAccessTokens=True,
                                      EnablePostUsernameOverride=True)
        await req("PUT", "/config", tok, json=cfg)

        r, team = await req("GET", f"/teams/name/{a.team}", tok, ok=(404,))
        if r.status == 404:
            _, team = await req("POST", "/teams", tok, json={"name": a.team, "display_name": a.team, "type": "I"})
        r, board = await req("GET", f"/teams/{team['id']}/channels/name/{a.board}", tok, ok=(404,))
        if r.status == 404:
            _, board = await req("POST", "/channels", tok, json={"team_id": team["id"], "name": a.board,
                                                                "display_name": a.board, "type": "O"})

        _, admin = await req("GET", "/users/me", tok)
        await req("POST", f"/teams/{team['id']}/members", tok, json={"team_id": team["id"], "user_id": admin["id"]})
        await req("POST", f"/channels/{board['id']}/members", tok, json={"user_id": admin["id"]})

        agents = a.agents.split(",")
        tokens, ids = {}, {}
        for agent in agents:
            r, bot = await req("GET", f"/users/username/{a.team}-{agent}", tok, ok=(404,))
            if r.status == 404:
                _, bot = await req("POST", "/bots", tok, json={"username": f"{a.team}-{agent}", "display_name": agent})
                bot_id = bot["user_id"]
            else:
                bot_id = bot["id"]
            await req("POST", f"/teams/{team['id']}/members", tok, json={"team_id": team["id"], "user_id": bot_id})
            await req("POST", f"/channels/{board['id']}/members", tok, json={"user_id": bot_id})
            _, t = await req("POST", f"/users/{bot_id}/tokens", tok, json={"description": f"{a.team} run"})
            tokens[agent] = t["token"]
            ids[agent] = bot_id

        for i, x in enumerate(agents):
            for y in agents[i + 1:]:
                name = pair_channel(x, y)
                r, ch = await req("GET", f"/teams/{team['id']}/channels/name/{name}", tok, ok=(404,))
                if r.status == 404:
                    _, ch = await req("POST", "/channels", tok, json={"team_id": team["id"], "name": name,
                                                                     "display_name": f"{x} ↔ {y}", "type": "P"})
                for uid in (ids[x], ids[y], admin["id"]):
                    await req("POST", f"/channels/{ch['id']}/members", tok, json={"user_id": uid})

    out = {"url": a.url, "team": a.team, "board": a.board, "tokens": tokens}
    Path(a.out).write_text(json.dumps(out, indent=2))
    os.chmod(a.out, 0o600)
    print(f"team={a.team} board={a.board} agents={list(tokens)} -> {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=os.environ.get("MM_URL", "http://127.0.0.1:18065"))
    p.add_argument("--team", required=True, help="lowercase, e.g. one per run")
    p.add_argument("--board", default="board")
    p.add_argument("--agents", required=True, help="comma-separated agent usernames")
    p.add_argument("--out", required=True)
    asyncio.run(main(p.parse_args()))
