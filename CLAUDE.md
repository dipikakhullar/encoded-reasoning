# Encoded reasoning

How do groups of LLM agents coordinate, collude and invent conventions? Git repo (one writer), origin
`dipikakhullar/encoded-reasoning` (**public**: never commit transcripts, runs or credentials). Treat `runs/` as evidence:
add new files rather than editing old ones.

## Map
| Where | What | Status / headline |
|---|---|---|
| `environments/social/` | Attention-economy social network (posts, feed, board, DMs, reports); worlds in `worlds/` | 50 gpt-5.5 agents invent a public CLAIM/STATUS debt ledger around day 300 (k50n1000, report_signal v1–v7) |
| `environments/negotiation/` | Copy of `social` with board-only comms, to recreate the ledger | Needs board-only comms **and** the "highest balance" sentence |
| `environments/core/` | Gateway LLM client + agent loop shared by both envs; models in `environments/models.yaml` | — |
| `slack_collaboration/` | Mattermost as the agent message board | Server + client + smoke test work; no agent loop yet |
| `experiments/<env>/` | Launch + analysis scripts and `runs/` (gitignored) | `manager_worker/` keeps only the 10_05 runs + plots (harness deleted 2026-10-20): nothing beat Astra solo (TB2.1 18/20), no reliable team effect on tau3 |
| `research_update/` | Write-ups per experiment day | HTML + per-day `.md` / `_data.json` + plots |
| `mats10_feb2026/` | Feb 2026 CoT work (CoT quality metrics, GEPA) | Frozen; own `pyproject.toml` |

Start from `research_update/` for past results rather than re-scanning `runs/`. Run outputs from before 2026-10-20 still
name the old paths (`9_22/collusion_bench`, `10_05/`, ...).

## Setup
- Envs: `environments/.venv` (`pip install -r environments/requirements.txt`). Run from `environments/`, e.g.
  `.venv/bin/python -m social.run --world report_signal_v7 --agents 50 --ticks 1000 --policy llm`; outputs default to
  `experiments/<env>/runs/`. Long runs: `experiments/<env>/launch_long.sh`. API key comes from the repo-root `.env`.
- Mattermost: container `dipika-mattermost` on `127.0.0.1:18065`; `slack_collaboration/.venv`; see README.md.
- Plots: `uv run --no-project --with matplotlib python <script>` (no venv here has matplotlib).

## Running experiments
- **Cost**: estimate before launching and get sign-off. Re-quote if horizon, agent count or reps change.
- **Team topology**: default team conditions to `w2w_private` (workers can message each other). Say which topology was
  used in any result.
- **Run names**: `[communication_style]_[benchmark]_[datetime]`; style carries the worker tag (`-ds4`, `-glm`, `-qwen27`).
- **Docker**: shared with other users' jobs. Keep total concurrency ≤ ~25 trials, and give your containers distinct names.
- Launch under `nohup` / `setsid`, one output dir per run.
- n is usually 1–3 per cell, so ±1–2 is noise. Report Wilson CIs and paired / Fisher tests; don't over-claim.

## Research updates
One entry per experiment day. Heading "Date: <date> — <takeaway in 3–6 words>". 1–2 setup bullets, 2–3 findings with
numbers, then verbatim board / DM excerpts (day, agent id, run) each followed by a short reading of the behavior.
Deliver as HTML. Say "day N" for platform time, never "tick" (except inside verbatim agent quotes).

## Conventions
- Plots must be paper-ready matplotlib figures, as they would appear in an academic paper. No titles, captions,
  annotations, footnotes or explanatory text on the figure (that goes in the caption or prose). Clear axis labels with
  units, a legend only when needed, consistent colors across figures, no top/right spines, readable font sizes at
  column width, and error bars (Wilson CIs) on rates. Always matplotlib, saved as PNG only (300 dpi); never PDF.
  Keep text on the plot minimal: short legend labels (e.g. A / B / C or a model name) and short axis labels; condition
  descriptions go in the caption, not the legend.
- New environment → `environments/<name>/`; its launch scripts, analysis and runs → `experiments/<name>/`.
- Give times in Pacific time.
- When asked what is running, answer with a table with columns: experiment, notes, ETA, cost, status.
- Keep credentials out of new files and off /shared (`.env` here currently sits on /shared, which `/shared/AGENTS.md`
  discourages). Mattermost bot tokens go on node-local disk.
