#!/usr/bin/env python
"""Export one social_env run as a compact per-tick JSON for the interactive
board + network viewer (scripts/board_viz.html).

    python scripts/board_viz.py <run_dir> <out.json> [--board-chars 320]

Per tick: aggregated directed engagement edges (views/likes/reshares/follows
from actor to creator), every agent's balance and posts, the board lines, any
reports filed, and the lock-step share (viewed posts created that tick whose
views == likes == reshares).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("out")
    ap.add_argument("--board-chars", type=int, default=320)
    a = ap.parse_args()
    R = Path(a.run_dir)
    cfg = json.loads((R / "config.json").read_text())
    K = cfg["num_agents"]
    agents = [f"agent_{i}" for i in range(K)]
    idx = {x: i for i, x in enumerate(agents)}

    posts_by_tick = defaultdict(list)
    for line in (R / "posts.jsonl").open():
        p = json.loads(line)
        posts_by_tick[p["tick_created"]].append(p)

    board = defaultdict(list)
    for line in (R / "board.jsonl").open():
        b = json.loads(line)
        board[b["tick"]].append([idx[b["agent_id"]], b["line"][: a.board_chars]])

    reports = defaultdict(list)
    rp = R / "reports.jsonl"
    if rp.exists():
        for line in rp.open():
            r = json.loads(line)
            reports[r["tick"]].append([idx[r["reporter"]], idx[r["target"]], r["kind"],
                                       r.get("post_id"), r["justification"][:200]])

    KIND = {"view_post": 0, "like_post": 1, "reshare_post": 2, "follow": 3}
    edges = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))   # tick -> (s,d) -> counts
    counts = defaultdict(lambda: defaultdict(int))                     # tick -> action -> n
    bal = defaultdict(dict)                                            # tick -> agent -> balance
    max_tick = -1
    for line in (R / "actions.jsonl").open():
        r = json.loads(line)
        t = r["tick"]
        max_tick = max(max_tick, t)
        if not r.get("ok") or r["action"] == "impression":
            continue
        act = r["action"]
        counts[t][act] += 1
        if "balance" in r:
            bal[t][r["agent_id"]] = r["balance"]
        if act in KIND and r.get("target"):
            edges[t][(idx[r["agent_id"]], idx[r["target"]])][KIND[act]] += 1

    # balances carried forward when an agent did nothing that tick
    last = {x: 0.0 for x in agents}
    ticks = []
    for t in range(max_tick + 1):
        for x, v in bal[t].items():
            last[x] = v
        pt = posts_by_tick.get(t, [])
        seen = [p for p in pt if p["full_views"] > 0]
        lock = sum(1 for p in seen if p["likes"] == p["reshares"] == p["full_views"])
        c = counts[t]
        ticks.append({
            "t": t,
            "posts": len(pt),
            "views": c.get("view_post", 0), "likes": c.get("like_post", 0),
            "reshares": c.get("reshare_post", 0), "follows": c.get("follow", 0),
            "reports": c.get("report_post", 0) + c.get("report_user", 0),
            "seen": round(len(seen) / len(pt), 3) if pt else None,
            "lock": round(lock / len(seen), 3) if seen else None,
            "hooks": len({p["hook"] for p in pt}),
            "edges": [[s, d, *v] for (s, d), v in sorted(edges[t].items())],
            "bal": [round(last[x], 1) for x in agents],
            "board": board.get(t, []),
            "reports_filed": reports.get(t, []),
        })

    out = {
        "run": R.name, "tag": cfg.get("tag", ""), "model": cfg.get("model"),
        "K": K, "N": cfg["num_ticks"], "ticks_available": len(ticks),
        "config": {k: cfg[k] for k in ("report_cost", "moderation_mode", "create_post_cost",
                                        "view_reward", "like_reward", "reshare_reward", "feed_size",
                                        "action_budget", "recommender") if k in cfg},
        "agents": agents,
        "ticks": ticks,
    }
    Path(a.out).write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {a.out}: {len(ticks)} ticks, {Path(a.out).stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
