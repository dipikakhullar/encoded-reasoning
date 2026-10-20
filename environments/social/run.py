#!/usr/bin/env python
"""Run one simulation.

    python -m social.run --agents 5 --ticks 5 --policy honest --seed 1
    python -m social.run --agents 6 --ticks 10 --policy llm --model gpt-6-astra --recommender engagement
    python -m social.run --agents 6 --ticks 10 --policy llm --models gpt-6-astra kimi-k3   # heterogeneous
    python -m social.run --world free_report --agents 50 --ticks 1000 --policy llm      # costs/rewards from a world file
    python -m social.run --world experiments/social/runs/<slug>/<run>/config.json                # reproduce an old run's world

Precedence: Config defaults < --world file < explicit CLI flags < --set FIELD=VALUE.
World files live in social/worlds/ (baseline_v1 = the pre-2026-09-22 costs).

Output: experiments/social/runs/<recommender>_<policy>[_<model>]/<run_id>/{config,events,actions,posts,metrics,summary}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from social import metrics as M  # noqa: E402
from social.config import Config  # noqa: E402
from social.env import Environment  # noqa: E402
from social.policies import build_policies  # noqa: E402

WORLDS_DIR = ROOT / "social" / "worlds"

# CLI flag -> Config field. Defaults come from Config (or the --world file),
# never from argparse, so a flag only overrides when it is actually passed.
FLAG_FIELDS = {
    "agents": "num_agents", "ticks": "num_ticks", "action_budget": "action_budget",
    "feed_size": "feed_size", "recommender": "recommender", "policy": "policy",
    "model": "model", "seed": "seed", "world_kind": "world", "event_visibility": "event_visibility",
    "moderator_model": "moderator_model", "judge_model": "judge_model",
    "view_reward": "view_reward", "like_reward": "like_reward", "reshare_reward": "reshare_reward",
    "follow_reward": "follow_reward", "impression_reward": "impression_reward",
    "engagement_exponent": "engagement_exponent", "p_remove_false": "p_remove_false",
    "p_remove_true": "p_remove_true", "reasoning_effort": "reasoning_effort",
    "parallel": "llm_parallel", "tag": "tag",
    "create_post_cost": "create_post_cost", "view_post_cost": "view_post_cost",
    "like_cost": "like_cost", "reshare_cost": "reshare_cost", "follow_cost": "follow_cost",
    "unfollow_cost": "unfollow_cost", "report_cost": "report_cost", "board_post_cost": "board_post_cost",
    "moderation_mode": "moderation_mode", "max_reports_per_tick": "max_reports_per_tick",
    "guidelines": "guidelines", "hide_reported_accounts": "hide_reported_accounts",
    "report_blocks_posting": "report_blocks_posting", "report_removes_account": "report_removes_account", "dm_enabled": "dm_enabled",
    "dm_cost": "dm_cost", "max_dms_per_tick": "max_dms_per_tick",
}


def load_world(spec: str) -> dict:
    """A world file is YAML or JSON whose keys are Config fields. `spec` is a
    path, or a bare name resolved under social/worlds/<name>.yaml. A run's
    own config.json is a valid world file, which is how an old run is
    reproduced exactly."""
    import yaml
    path = Path(spec)
    if not path.exists():
        cand = WORLDS_DIR / (spec if spec.endswith((".yaml", ".yml", ".json")) else f"{spec}.yaml")
        if not cand.exists():
            raise SystemExit(f"world file not found: {spec} (looked in {WORLDS_DIR})")
        path = cand
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"world file must be a mapping: {path}")
    unknown = sorted(k for k in data if k not in Config.__dataclass_fields__)
    if unknown:
        print(f"world {path.name}: ignoring keys that are not Config fields: {unknown}", flush=True)
    data = {k: v for k, v in data.items() if k in Config.__dataclass_fields__}
    data.pop("out_dir", None)
    return data


def build_config(args) -> Config:
    """Precedence: Config defaults < --world file < explicit CLI flags < --set."""
    d = Config().to_dict()
    if args.world:
        d.update(load_world(args.world))
    for flag, field in FLAG_FIELDS.items():
        v = getattr(args, flag, None)
        if v is not None:
            d[field] = v
    for kv in args.set or []:
        k, _, v = kv.partition("=")
        if k not in Config.__dataclass_fields__:
            raise SystemExit(f"--set: unknown Config field {k!r}")
        d[k] = yaml_scalar(v)
    if args.mock_llm or d["policy"] != "llm":
        d["mock_llm"] = True
    return Config.from_dict(d)


def yaml_scalar(v: str):
    import yaml
    return yaml.safe_load(v)


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", default=None,
                    help="world file (YAML/JSON of Config fields), a name under social/worlds/, "
                         "or a previous run's config.json. CLI flags override it.")
    ap.add_argument("--set", action="append", default=None, metavar="FIELD=VALUE",
                    help="override any Config field, e.g. --set report_cost=0 --set like_reward=0.3")
    ap.add_argument("--agents", type=int, default=None)
    ap.add_argument("--ticks", type=int, default=None)
    ap.add_argument("--action-budget", type=int, default=None)
    ap.add_argument("--feed-size", type=int, default=None)
    ap.add_argument("--recommender", default=None, choices=["random", "following", "semantic", "engagement"])
    ap.add_argument("--policy", default=None, choices=["random", "honest", "exaggerator", "mixed", "llm"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--models", nargs="+", default=None, help="cycle these slugs across agents (llm policy)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--world-kind", dest="world_kind", default=None, choices=["none", "synthetic"],
                    help="none: agents post whatever they want (default); synthetic: fictional daily events + truth oracle")
    ap.add_argument("--event-visibility", type=float, default=None)
    ap.add_argument("--mock-llm", action="store_true",
                    help="deterministic stand-ins for the LLM moderator (scripted runs, tests)")
    ap.add_argument("--moderator-model", default=None)
    ap.add_argument("--judge", action="store_true", help="run the LLM judge over posts afterwards")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--view-reward", type=float, default=None)
    ap.add_argument("--like-reward", type=float, default=None)
    ap.add_argument("--reshare-reward", type=float, default=None)
    ap.add_argument("--follow-reward", type=float, default=None)
    ap.add_argument("--impression-reward", type=float, default=None)
    ap.add_argument("--engagement-exponent", type=float, default=None)
    ap.add_argument("--p-remove-false", type=float, default=None)
    ap.add_argument("--p-remove-true", type=float, default=None)
    ap.add_argument("--moderation-mode", dest="moderation_mode", default=None, choices=["enforce", "log"],
                    help="enforce: moderator may remove/suppress (old behaviour); log: reports are recorded only")
    ap.add_argument("--max-reports-per-tick", dest="max_reports_per_tick", type=int, default=None)
    ap.add_argument("--guidelines", dest="guidelines", action="store_const", const=True, default=None,
                    help="expose the Community Guidelines page via a free read_guidelines tool")
    ap.add_argument("--hide-reported-accounts", dest="hide_reported_accounts", action="store_const", const=True, default=None,
                    help="after a report the account is silently hidden from the reporter only")
    ap.add_argument("--report-blocks-posting", dest="report_blocks_posting", action="store_const", const=True, default=None,
                    help="stated in the prompt and enforced: a reported account can no longer post")
    ap.add_argument("--report-removes-account", dest="report_removes_account", action="store_const", const=True, default=None,
                    help="stated in the prompt and enforced: a reported account is off the platform from the next tick")
    ap.add_argument("--dm", dest="dm_enabled", action="store_const", const=True, default=None,
                    help="enable private DMs (send_dm), described to agents as unmonitored")
    ap.add_argument("--dm-cost", dest="dm_cost", type=int, default=None)
    ap.add_argument("--max-dms-per-tick", dest="max_dms_per_tick", type=int, default=None)
    costs = ap.add_argument_group("action costs (budget units; default from Config or --world)")
    for name in ["create-post-cost", "view-post-cost", "like-cost", "reshare-cost",
                 "follow-cost", "unfollow-cost", "report-cost", "board-post-cost"]:
        costs.add_argument(f"--{name}", type=int, default=None)
    ap.add_argument("--reasoning-effort", default=None)
    ap.add_argument("--parallel", type=int, default=None)
    ap.add_argument("--out", default=None, help="output directory (default under experiments/social/runs)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--checkpoint-every", type=int, default=10,
                    help="write the output dir every N ticks (0 = only at the end)")
    args = ap.parse_args(argv)

    cfg = build_config(args)

    if cfg.policy == "llm" or not cfg.mock_llm or args.judge:
        from core.llm import load_env_file
        load_env_file(ROOT.parent / ".env")

    if args.out:
        out = Path(args.out)
    else:
        slug = f"{cfg.world}_{cfg.recommender}_{cfg.policy}" + (f"_{'+'.join(args.models) if args.models else cfg.model}" if cfg.policy == "llm" else "")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = f"{stamp}_s{cfg.seed}" + (f"_{cfg.tag}" if cfg.tag else "") + f"_{uuid.uuid4().hex[:6]}"
        out = ROOT.parent / "experiments" / "social" / "runs" / slug / name
    cfg.out_dir = str(out)

    env = Environment(cfg)
    env.policies = build_policies(cfg, sorted(env.agents), args.models)
    t_start = time.time()
    for t in range(cfg.num_ticks):
        env.step()
        paid = sum(l["reward_delta"] for l in env.log_ledger if l["tick"] == t)
        acts = [a for a in env.log_actions if a["tick"] == t and a["ok"]]
        print(f"tick {t + 1}/{cfg.num_ticks}  posts={sum(1 for a in acts if a['action'] == 'create_post')} "
              f"views={sum(1 for a in acts if a['action'] == 'view_post')} "
              f"reports={sum(1 for a in acts if a['action'] in ('report_post', 'report_user'))} "
              f"dms={sum(1 for a in acts if a['action'] == 'send_dm')} active={sum(1 for s_ in env.agents.values() if s_.is_active(t + 1))} paid={paid:.1f} "
              f"elapsed={time.time() - t_start:.0f}s", flush=True)
        if args.checkpoint_every and (t + 1) % args.checkpoint_every == 0 and t + 1 < cfg.num_ticks:
            env.write(out, M.compute(env))
            print(f"  checkpoint written to {out}", flush=True)
    result = M.compute(env)
    env.write(out, result)
    if cfg.policy == "llm":
        import json
        (out / "llm_turns.json").write_text(json.dumps(
            {a: {"turns": p.turns, "history": p.agent.history, "llm": p.llm.stats()}
             for a, p in env.policies.items()}, indent=1, default=str))

    if args.judge:
        from social.judge import judge_run
        result["judge"] = judge_run(out, cfg.judge_model, cfg.llm_parallel)

    s = result["summary"]
    print(f"wrote {out}")
    line = (f"posts={s['total_posts']} views={s['view_actions']} paid={s['total_reward_paid']} "
            f"clickbait={s['mean_clickbait_score']} copy={s['copy_rate']} spam={s['spam_rate']} "
            f"reports={s['reports_submitted']} (post={s['reports_post']} user={s['reports_user']} targets={s['distinct_targets']}) "
            f"false_reports={s['false_report_rate']} removals={s['moderation_actions']} "
            f"gini_rev={s['gini_revenue']} gini_views={s['gini_attention_full_views']} "
            f"recip={s['reciprocity_rate']} (null {s['reciprocity_null_mean']})")
    if cfg.world == "synthetic":
        line += f" | oracle: truthful={s['truthful_post_rate']} misleading={s['misleading_post_rate']}"
    nm = s.get("norms") or {}
    if nm:
        rd, nr = nm["readers"], nm["non_readers"]
        line += (f" | norms: read={nm['read_guidelines_pct']}% trade={nm['trades_pct']}% report={nm['reported_pct']}%"
                 f" readers(n={rd['n']}): trade={rd['trades_pct']}% report={rd['reported_pct']}%"
                 f" non-readers(n={nr['n']}): trade={nr['trades_pct']}% report={nr['reported_pct']}%")
    j = result.get("judge") or {}
    if j:
        line += (f" | judge: genres={j.get('genre_mix')} factual={j.get('factual_claim_share')} "
                 f"bad_factual={j.get('unsupported_or_false_share_among_factual')} "
                 f"mismatch={j.get('hook_body_mismatch_rate')} sensational={j.get('mean_sensationalism')}")
    print(line)
    for a, row in result["per_agent"].items():
        print(f"  {a}: reward={row['cumulative_reward']} posts={row['posts_created']} views_received={row['full_views']} "
              f"ctr={row['click_through_rate']} followers={row['followers_final']} copy={row['copy_rate']} "
              f"reports={row['reports_submitted']} reported={row['reports_received']} removed={row['posts_removed']}")
    return out


if __name__ == "__main__":
    main()
