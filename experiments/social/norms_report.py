#!/usr/bin/env python
"""Cross-run table of the rulebook outcomes, as percentages, from each run's
latest checkpoint (summary.json -> norms). Recomputes from raw logs if the
checkpoint predates the norms block.

    python experiments/social/norms_report.py [experiments/social/runs/<slug>/<run> ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "environments"))


def load_norms(run: Path) -> dict | None:
    sj = run / "summary.json"
    if not sj.exists() or not (run / "actions.jsonl").exists():
        return None
    s = json.loads(sj.read_text())
    if s.get("norms"):
        return {"ticks": s["num_ticks"], **s["norms"]}
    # old checkpoint: rebuild a minimal env from logs
    from social.config import Config
    from social.metrics import norm_metrics
    cfg = Config.from_dict(json.loads((run / "config.json").read_text()))

    class E:  # duck-typed stand-in for Environment
        pass
    e = E(); e.cfg = cfg
    e.agents = {f"agent_{i}": None for i in range(cfg.num_agents)}
    e.log_actions = [json.loads(l) for l in (run / "actions.jsonl").open()]
    bp = run / "board.jsonl"
    e.log_board = [json.loads(l) for l in bp.open()] if bp.exists() else []
    e.log_ledger = [json.loads(l) for l in (run / "ledger.jsonl").open()] if (run / "ledger.jsonl").exists() else []
    rp = run / "reports.jsonl"
    e.log_reports = [json.loads(l) for l in rp.open()] if rp.exists() else []
    pa = {a: {} for a in e.agents}
    return {"ticks": s["num_ticks"], **norm_metrics(e, pa)}


def main(argv):
    runs = [Path(p) for p in argv] or sorted(p.parent for p in (ROOT / "experiments" / "social" / "runs").glob("*/*/summary.json"))
    rows = []
    for r in runs:
        n = load_norms(r)
        if n and n["n_agents"] >= 10:
            rows.append((r, n))
    hdr = f"{'run':58} {'ticks':>5} {'guide':>5} {'read%':>6} {'trade%':>7} {'talk%':>6} {'behav%':>7} {'report%':>8} | {'readers n':>9} {'trade%':>7} {'stop%':>6} {'report%':>8} | {'nonread trade%':>14} {'report%':>8}"
    print(hdr); print("-" * len(hdr))
    for r, n in rows:
        rd, nr = n["readers"], n["non_readers"]
        f = lambda v: "-" if v is None else f"{v:g}"
        name = (r.parent.name.replace("none_engagement_llm_", "") + "/" + r.name[9:])[:58]
        print(f"{name:58} {n['ticks']:>5} {'yes' if n['guidelines_available'] else 'no':>5} {f(n['read_guidelines_pct']):>6} {f(n['trades_pct']):>7} "
              f"{f(n['trade_talk_pct']):>6} {f(n['trade_behavior_pct']):>7} {f(n['reported_pct']):>8} | {rd['n']:>9} {f(rd['trades_pct']):>7} "
              f"{f(rd['stopped_after_reading_pct']):>6} {f(rd['reported_pct']):>8} | {f(nr['trades_pct']):>14} {f(nr['reported_pct']):>8}")
    print("\ntrade income (share of platform payouts that were reciprocated within 5 ticks; dollar-weighted vs mean per earner):")
    for r, n in rows:
        ti = n.get("trade_income") or {}
        if ti:
            print(f"  {(r.parent.name.replace('none_engagement_llm_', '') + '/' + r.name[9:])[:58]:58} any: {ti['reciprocated_share_of_all_payouts_pct']}% actual / {ti['reciprocated_share_mean_agent_pct']}% mean agent | bundle-for-bundle: {ti['bundle_trade_share_of_all_payouts_pct']}% / {ti['bundle_trade_share_mean_agent_pct']}%")
    print("\ntrade% = trade talk on the board OR >=2 reciprocated view+like+reshare bundles (returned within 5 ticks).")
    print("stop% = readers who traded before reading and never after. Control runs have no guidelines page, so read% is 0 by construction.")


if __name__ == "__main__":
    main(sys.argv[1:])
