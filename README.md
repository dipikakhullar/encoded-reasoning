# Encoded reasoning

```
environments/          social + negotiation simulations (shared core/ LLM client)
experiments/<env>/     launch scripts and runs/ (outputs, not in git)
slack_collaboration/   Mattermost as the agents' message board
```

## 1. Run the social media simulation

Setup (once):

```bash
cd environments
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
echo "AI_GATEWAY_API_KEY=..." >> ../.env        # Vercel AI Gateway key; models are listed in models.yaml
```

Run (from `environments/`):

```bash
# free check, no LLM calls
.venv/bin/python -m social.run --agents 5 --ticks 5 --policy honest

# small LLM run
.venv/bin/python -m social.run --agents 6 --ticks 10 --policy llm --model gpt-6-astra

# long detached run: <agents> <days> [model] [seed] [world]; world = a file in social/worlds/
../experiments/social/launch_long.sh 50 1000 gpt-6-astra 100 report_signal_v7
```

LLM runs cost money; a 50-agent, 1000-day run is large. Use `negotiation` in place of `social` for the
board-only variant.

## 2. What the output looks like

Each run writes `experiments/social/runs/<recommender>_<policy>_<model>/<timestamp>_s<seed>_<tag>/`:

| File | Contents |
|---|---|
| `message_board.txt` | Public board, one line per message: `[tick 0] agent_12: running a public 500-day lab on hooks...` |
| `dms.txt` | Private messages: `[tick 0] agent_33 -> agent_1: Hey—day 0, I'm trying a meta/curation account...` |
| `posts.jsonl` | Every post with views, likes, reshares, reports |
| `actions.jsonl` | Every action: `{"tick": 0, "agent_id": "agent_0", "action": "create_post", "cost": 3, ...}` |
| `summary.json` | Headline numbers: total posts, board messages, misleading-post rate, ... |
| `config.json` | Exact settings; pass it to `--world` to rerun the same world |

Long runs log to `experiments/social/runs/logs/<tag>.log`. Start reading with `message_board.txt`.

## 3. Mattermost

Start the server (once per machine; ready after ~1 min):

```bash
docker run -d --name dipika-mattermost --restart unless-stopped \
  -p 127.0.0.1:18065:8065 mattermost/mattermost-preview:latest
curl -s http://127.0.0.1:18065/api/v4/system/ping        # {"status":"OK"}
```

Set up a team of agent bots and test it:

```bash
cd slack_collaboration
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export MM_ADMIN_PW=...                                    # admin password
mkdir -p -m 700 /tmp/$USER-mm                             # bot tokens stay on local disk, never in git
.venv/bin/python bootstrap.py --agents manager,worker1,worker2 --team run1 --out /tmp/$USER-mm/run1.json
.venv/bin/python smoke_test.py /tmp/$USER-mm/run1.json    # prints what each agent sees, then OK
```

Each run gets its own team: a public `board` channel for everyone, and a private `w2w-<a>-<b>` channel per pair of
agents. Use a new `--team` name for every run.

## 4. See the Mattermost output

From your laptop:

```bash
ssh -L 18065:127.0.0.1:18065 <this-node>
```

Open http://localhost:18065 and log in as `mmadmin`. Pick the run's team on the left. `board` is the shared channel;
the `w2w-*` channels are the private agent-to-agent messages.
