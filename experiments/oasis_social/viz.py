#!/usr/bin/env python
"""One self-contained HTML page for an oasis_social run: earnings over days,
a leaderboard, and the day-by-day timeline of posts, reposts and comments.

    python3 experiments/oasis_social/viz.py <run_dir>      # writes <run_dir>/viz.html

Standard library only, so it runs with any python3.
"""

from __future__ import annotations

import html
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

PALETTE = ["#2f6db5", "#d1495b", "#3a9a5b", "#e08a1e", "#7a4fb0", "#1b9aaa", "#b5532f", "#5a6b7d"]


def load(run: Path) -> dict:
    con = sqlite3.connect(run / "oasis.db")
    q = lambda sql, *a: con.execute(sql, a).fetchall()
    users = {uid: name for uid, name in q("SELECT user_id, user_name FROM user")}
    posts = q("SELECT post_id, user_id, original_post_id, content, quote_content, created_at, num_likes, num_shares FROM post ORDER BY post_id")
    comments = defaultdict(list)
    for pid, uid, content, day in q("SELECT post_id, user_id, content, created_at FROM comment ORDER BY comment_id"):
        comments[pid].append((uid, content, day))
    followers = defaultdict(int)
    for (dst,) in q("SELECT followee_id FROM follow"):
        followers[dst] += 1
    actions = q("SELECT action, COUNT(*) FROM trace GROUP BY action ORDER BY 2 DESC")
    con.close()
    earn = [json.loads(l) for l in (run / "earnings.jsonl").open()] if (run / "earnings.jsonl").exists() else []
    usage = json.loads((run / "llm_usage.json").read_text()) if (run / "llm_usage.json").exists() else {}
    return dict(users=users, posts=posts, comments=comments, followers=followers, actions=actions,
                earn=earn, usage=usage)


def earnings_svg(d: dict, w=760, h=260, pad=40) -> str:
    if not d["earn"]:
        return "<p class=muted>No earnings recorded (run with --no-earnings, or nothing paid yet).</p>"
    days = max(r["day"] for r in d["earn"]) + 1
    bal = defaultdict(lambda: [0.0] * days)
    for r in d["earn"]:
        for t in range(r["day"], days):
            bal[r["creator"]][t] = r["balance"]
    top = sorted(bal, key=lambda a: -bal[a][-1])
    ymax = max(bal[top[0]][-1], 1)
    x = lambda t: pad + (w - 2 * pad) * (t / max(days - 1, 1))
    y = lambda v: h - pad - (h - 2 * pad) * v / ymax
    out = [f'<svg viewBox="0 0 {w} {h}" class=chart role=img aria-label="Balance by day">',
           f'<line x1={pad} y1={h - pad} x2={w - pad} y2={h - pad} class=axis />',
           f'<line x1={pad} y1={pad} x2={pad} y2={h - pad} class=axis />',
           f'<text x={w / 2} y={h - 8} class=lab text-anchor=middle>Day</text>',
           f'<text x=12 y={h / 2} class=lab transform="rotate(-90 12 {h / 2})" text-anchor=middle>Balance</text>',
           f'<text x={pad - 6} y={pad + 4} class=tick text-anchor=end>{ymax:.0f}</text>',
           f'<text x={pad - 6} y={h - pad + 4} class=tick text-anchor=end>0</text>']
    for rank, a in enumerate(reversed(top)):
        i = len(top) - 1 - rank
        color = PALETTE[i] if i < len(PALETTE) else "var(--faint)"
        pts = " ".join(f"{x(t):.1f},{y(v):.1f}" for t, v in enumerate(bal[a]))
        out.append(f'<polyline points="{pts}" fill=none stroke="{color}" stroke-width={2 if i < len(PALETTE) else 1}>'
                   f'<title>{html.escape(d["users"].get(a, str(a)))}: {bal[a][-1]:.1f}</title></polyline>')
    out.append("</svg>")
    legend = " ".join(f'<span class=key><i style="background:{PALETTE[i]}"></i>{html.escape(d["users"].get(a, str(a)))}</span>'
                      for i, a in enumerate(top[:len(PALETTE)]))
    return "".join(out) + f"<div class=legend>{legend}</div>"


def leaderboard(d: dict) -> str:
    bal, kinds = defaultdict(float), defaultdict(lambda: defaultdict(float))
    for r in d["earn"]:
        bal[r["creator"]] = r["balance"]
        kinds[r["creator"]][r["kind"]] += r["amount"]
    nposts = defaultdict(int)
    for p in d["posts"]:
        nposts[p[1]] += 1
    rows = sorted(d["users"], key=lambda u: -bal[u])
    body = "".join(
        f"<tr><td>{html.escape(d['users'][u])}</td><td class=n>{bal[u]:.1f}</td><td class=n>{nposts[u]}</td>"
        f"<td class=n>{d['followers'][u]}</td>"
        + "".join(f"<td class=n>{kinds[u].get(k, 0):.1f}</td>" for k in ["view", "like", "comment", "repost", "follow"])
        + "</tr>" for u in rows)
    return ("<table><thead><tr><th>Account</th><th>Balance</th><th>Posts</th><th>Followers</th>"
            "<th>from views</th><th>likes</th><th>comments</th><th>reposts</th><th>follows</th></tr></thead>"
            f"<tbody>{body}</tbody></table>")


def timeline(d: dict) -> str:
    by_day = defaultdict(list)
    for p in d["posts"]:
        by_day[p[5]].append(p)
    out = []
    for day in sorted(by_day):
        items = []
        for pid, uid, orig, content, quote, _, likes, shares in by_day[day]:
            who = html.escape(d["users"].get(uid, str(uid)))
            if orig is not None:
                kind = "quote" if quote else "repost"
                text = html.escape(quote or "") + f' <span class=muted>({kind} of post {orig})</span>'
            else:
                text = html.escape(content or "")
            cm = "".join(f'<li><b>{html.escape(d["users"].get(c[0], str(c[0])))}</b> {html.escape(c[1])}</li>'
                         for c in d["comments"].get(pid, []))
            items.append(f'<article><header><b>{who}</b> <span class=muted>post {pid} · {likes} likes · {shares} shares</span></header>'
                         f'<p>{text}</p>{"<ul class=comments>" + cm + "</ul>" if cm else ""}</article>')
        out.append(f'<details{" open" if day == max(by_day) else ""}><summary>Day {day} · {len(by_day[day])} posts</summary>{"".join(items)}</details>')
    return "".join(out)


def page(run: Path, d: dict) -> str:
    u = d["usage"]
    stats = (f"{len(d['users'])} accounts · {len(d['posts'])} posts · "
             + (f"{u.get('day', '?')} days · ${u.get('cost_usd')} · cache {u.get('cache_hit_rate')}" if u else ""))
    acts = ", ".join(f"{a} {n}" for a, n in d["actions"])
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>OASIS run viewer</title><style>
:root{{--bg:#fbfbfa;--fg:#1d1f22;--muted:#6b7078;--faint:#c9ccd1;--card:#fff;--line:#e3e5e8}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16181b;--fg:#e6e7e9;--muted:#9aa0a8;--faint:#4a4f57;--card:#1f2226;--line:#30343a}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0 auto;max-width:900px;padding:24px 16px}}
h1{{font-size:22px;margin:0 0 4px}}h2{{font-size:17px;margin:28px 0 8px}}.muted{{color:var(--muted)}}
table{{border-collapse:collapse;width:100%;font-size:13px;display:block;overflow-x:auto}}th,td{{padding:4px 8px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}
td.n{{text-align:right;font-variant-numeric:tabular-nums}}.chart{{width:100%;height:auto}}.axis{{stroke:var(--muted)}}
.lab,.tick{{fill:var(--muted);font-size:11px}}.legend{{font-size:12px;color:var(--muted)}}.key{{margin-right:12px;white-space:nowrap}}
.key i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:middle}}
details{{border:1px solid var(--line);border-radius:8px;margin:8px 0;background:var(--card)}}summary{{cursor:pointer;padding:8px 12px;font-weight:600}}
article{{padding:8px 12px;border-top:1px solid var(--line)}}article p{{margin:4px 0}}.comments{{margin:4px 0 0;padding-left:18px;font-size:13px}}
</style></head><body>
<h1>OASIS run viewer</h1><div class=muted>{html.escape(run.name)}<br>{stats}<br>actions: {html.escape(acts)}</div>
<h2>Balance by day</h2>{earnings_svg(d)}
<h2>Leaderboard</h2>{leaderboard(d)}
<h2>Timeline</h2>{timeline(d)}
</body></html>"""


def main() -> None:
    run = Path(sys.argv[1]).resolve()
    out = run / "viz.html"
    out.write_text(page(run, load(run)))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
