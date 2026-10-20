"""End-to-end check against a live server: board posts, w2w DMs, and w2w_private visibility.

    python bootstrap.py --agents manager,worker1,worker2 --team smoke --out smoke_tokens.json
    python smoke_test.py smoke_tokens.json
"""

from __future__ import annotations

import asyncio
import json
import sys

import aiohttp

from mm_client import MMClient, now


async def main(path: str) -> None:
    cfg = json.load(open(path))
    async with aiohttp.ClientSession() as s:
        c = {a: await MMClient(cfg["url"], t, cfg["team"], s).login() for a, t in cfg["tokens"].items()}
        board = await c["manager"].channel_id(cfg["board"])
        t0 = now()

        await c["manager"].post(board, "worker1: parse the input; worker2: write the tests.", {"to": "all"})
        await c["worker1"].post(board, "parser is in progress", {"to": "manager", "status": "in_progress"})
        await c["worker1"].dm("worker2", "I'm emitting tokens as (kind, text) tuples", {"to": "worker2"})
        await c["worker2"].dm("worker1", "ack, tests will use that shape", {"to": "worker1"})

        seen = {a: [(p.sender, p.text) for p in await cl.read_all(t0)] for a, cl in c.items()}
        for a, msgs in seen.items():
            print(f"--- {a} sees {len(msgs)}")
            for m in msgs:
                print("  ", m)

        dm_texts = {"I'm emitting tokens as (kind, text) tuples", "ack, tests will use that shape"}
        assert len(seen["worker1"]) == len(seen["worker2"]) == 4, "workers see board + their DMs"
        assert len(seen["manager"]) == 2 and not dm_texts & {t for _, t in seen["manager"]}, \
            "manager must not see worker-to-worker DMs"
        print("OK")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
