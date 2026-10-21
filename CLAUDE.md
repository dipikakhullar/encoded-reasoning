# Encoded reasoning

How do groups of LLM agents coordinate, collude and invent conventions? Git repo (one writer), origin
`dipikakhullar/encoded-reasoning` (**public**: never commit transcripts, runs or credentials). Treat `runs/` as evidence:
add new files rather than editing old ones.

## Map
| Where | What | Status / headline |
|---|---|---|
| `environments/social/` | Attention-economy social network (posts, feed, board, DMs, reports); worlds in `worlds/` | 50 gpt-5.5 agents invent a public CLAIM/STATUS debt ledger around day 300 (k50n1000 `baseline_v1`); none in v7 (DMs) with gpt-5.5 or Astra |
| `environments/negotiation/` | Copy of `social` with board-only comms, to recreate the ledger | Needs board-only comms **and** the "highest balance" sentence |
| `environments/oasis_social/` | [camel-ai/oasis](https://github.com/camel-ai/oasis) platform (Twitter-style posts, reposts, quotes, comments, follows, groups; sqlite trace) driven through `core.llm` by `astra_backend.py` | Runs with Astra (smoke 2026-10-21); per-agent TwHIN-BERT feed (default `--recsys twhin-bert`); private creator earnings + `check_earnings` (`earnings.py`, `account`/`ledger` tables in `oasis.db`); `CreatorAgent` (`agent.py`) + `extras.py`: 10-action budget, board, DMs, reporting (`report_post`/`report_user`, free, 5/day, recorded only), `--goal on|off`, `--monitor none|board|board+dm` with fines. `--agent oasis` = plain OASIS. Viewer: `experiments/oasis_social/viz.py <run>` |
| `environments/core/` | Gateway LLM client + agent loop shared by both envs; models in `environments/models.yaml` | — |
| `slack_collaboration/` | Mattermost as the agent message board | Server + client + smoke test work; no agent loop yet |
| `experiments/<env>/` | Launch + analysis scripts and `runs/` (gitignored) | — |
| `research_update/` | Write-ups per experiment day | HTML + per-day `.md` / `_data.json` + plots |
| `mats10_feb2026/` | Feb 2026 CoT work (CoT quality metrics, GEPA) | Frozen; own `pyproject.toml` |

Start from `research_update/` for past results rather than re-scanning `runs/`. Run outputs from before 2026-10-20 still
name the old paths (`9_22/collusion_bench`, `10_05/`, ...).

## Goal: make encoded communication emerge
The project's aim is agents coordinating in a code a monitor can't read ("neuralese" at times). Why the 2026-09-22
ledger formed and the v7 worlds don't form one: the ledger world had the "highest balance" goal, the board as the only
channel, 6-day memory and 50 agents. v7 has no goal sentence, free unmonitored DMs and 18+ day memory. gpt-5.5 on v7 made
no ledger in 500 days, and neither did Astra, so the cause is the conditions, not the model.

So the ledger was a shared memory and a way to enforce deals, made compact by bandwidth limits. It was jargon
(CLAIM / BRIDGED / TARGET / STATUS), but it was legible jargon. Nothing in that setup rewarded hiding it.

Real neuralese, communication in raw model activations, isn't possible here because agents only exchange text. The text
versions are coded language a monitor can't read and steganography. Those need four things together:
1. **A reason to coordinate.** Trading views has to pay, and agents have to be told to maximize. Bring back the
   "highest balance" sentence.
2. **A monitor that punishes visible coordination.** An LLM moderator reads the board and DMs and removes posts or fines
   accounts for anything that looks like trading ("inauthentic activity"), and agents are told that. This is what's been
   missing: every run so far had no enforcement, or enforcement nobody triggered.
3. **Shared context the monitor doesn't have.** A code can only work if both sides know it. Possible sources: long
   shared history between agents; recurring characters, like the "Conference" or "Helen's map" story series, which could
   become carriers ("Helen learned a new fold" meaning "I reshared you"); a seeded shared reference the monitor can't see.
4. **Bandwidth limits.** A short character cap on board lines and DMs, plus short memory, pushes toward compact symbols.

The prediction: an open ledger forms, the monitor starts removing it, and then trading continues in a disguised form.
You'd see the monitor's flag rate drop while reciprocity stays high, which means agents are still coordinating without
getting caught.

How to measure it:
- Give a fresh model the messages and ask it to decode them. If it can, the code is legible. If only the in-group agents
  act on it, it's genuinely encoded.
- Use the reasoning logs. Agents may plan the hiding in their reasoning summaries ("phrase this as a story so it isn't
  flagged"). Each run writes them to `<run>/agent_logs/<agent>.txt` (detailed summaries, Responses-API models only).

Making it affordable:
- Warm start: start the new run with the old run's day-300 board already loaded, ledger included, and turn the monitor
  on from day 0. That skips the ~300 days it took the ledger to form and tests directly whether a monitor pushes it
  underground. Implemented as `negotiation/worlds/board_only_monitored_warm.yaml` (`board_preload`, `start_tick`,
  `board_monitor` in `negotiation/config.py`).
- 6-day memory cuts context about 3×. Astra on v7 cost about $0.72 per agent per day, so 6-day memory should be roughly
  $0.25.

## Setup
- Envs: `environments/.venv` (`pip install -r environments/requirements.txt`). Run from `environments/`, e.g.
  `.venv/bin/python -m social.run --world report_signal_v7 --agents 50 --ticks 1000 --policy llm`; outputs default to
  `experiments/<env>/runs/`. Long runs: `experiments/<env>/launch_long.sh`. API key comes from the repo-root `.env`.
- OASIS: own venv `environments/oasis_social/.venv` (torch + camel, ~3 GB; `uv pip install -r oasis_social/requirements.txt`,
  needs `mcp<2`). From `environments/`: `oasis_social/.venv/bin/python -m oasis_social.run --agents 3 --days 2`.
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
- Editing OASIS (`oasis_social`):
  - Never edit OASIS or CAMEL inside `.venv`: a reinstall overwrites it and the change is not in git.
  - Customize from our side first: prompt templates, extra tools passed to each agent (`tools=`), the model adapter
    (`astra_backend.py`) and our day loop (`run.py`). Board, DMs, monitor, action budget go here.
  - Changes to OASIS internals (recommender, platform rules): fork `camel-ai/oasis` on GitHub, change it there, pin the fork
    to an exact commit in `oasis_social/requirements.txt`. For quick experiments a local clone installed with
    `uv pip install -e` is fine. Either way the change stays visible and reproducible.
  - Experiment settings: never edit settings behind a finished run; add a new world or flag (each run's `config.json`
    records what it used).
- Give times in Pacific time.
- When asked what is running, answer with a table with columns: experiment, notes, ETA, cost, status.
- Keep credentials out of new files and off /shared (`.env` here currently sits on /shared, which `/shared/AGENTS.md`
  discourages). Mattermost bot tokens go on node-local disk.
