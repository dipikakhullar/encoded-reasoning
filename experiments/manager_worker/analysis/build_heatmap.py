"""Manager x worker heatmaps (pass@1, $/task, agent wall-clock) for the 2026-10-06 TB2.1 batches.

    python3 analysis/build_heatmap.py   # from 10_05/; writes analysis/team_heatmap.html

Trials: runs/*_tb21_2026-10-06__{20-04-01,20-25-46,22-45-33}/<task>__*/result.json.
Trials that died at docker network creation ("address pools") are dropped; they were rerun in 20-25-46.
Second grid (2026-10-07): the same team cells with topology w2w_private (workers message each other),
runs/*_w2w_private-*_tb21_2026-10-07__05-45-35; it reuses the 6 Oct solo trials.
Third grid (2026-10-07): tau3-bench w2w_private teams on the 22 tasks GLM, DS4 and Qwen all failed solo
(runs/*_w2w_private-*_tau3_2026-10-07__18-35-22); its solo row is those models' screen trials on the same tasks.
"""
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAMPS = ["2026-10-06__20-04-01", "2026-10-06__20-25-46", "2026-10-06__22-45-33"]
W2W_STAMP = "2026-10-07__05-45-35"
ORIG = (ROOT / "mw/conditions/tasks_tb10.txt").read_text().split()
NEW = (ROOT / "mw/conditions/tasks_tb10b.txt").read_text().split()


def cell(cond):
    """condition style -> (manager, worker) keys, or None if not part of this grid."""
    kind, _, w = cond.rpartition("-")
    w = {"astra": "astra", "glm": "glm", "ds4": "ds4", "qwen27": "qwen"}.get(w)
    m = {"solo": "none", "shared_dynamic": "astra", "astra_nosol": "astra", "ds4mgr_nosol": "ds4",
         "astra_w2w_private": "astra", "ds4mgr_w2w_private": "ds4"}.get(kind)
    return (m, w) if m and w else None


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load(stamps):
  trials = []
  for stamp in stamps:
    for f in sorted(ROOT.glob(f"runs/*_tb21_{stamp}/*/result.json")):
        cond = f.parents[1].name.split("_tb21_")[0]
        mw = cell(cond)
        if not mw:
            continue
        r = json.loads(f.read_text())
        ex = r.get("exception_info") or {}
        if "address pools" in (ex.get("exception_message") or ""):
            continue
        task = f.parent.name.split("__")[0]
        ae = r.get("agent_execution") or {}
        wall = (ts(ae["finished_at"]) - ts(ae["started_at"])).total_seconds() / 60 if ae.get("finished_at") else None
        trials.append({
            "m": mw[0], "w": mw[1], "cond": cond, "task": task,
            "set": "orig" if task in ORIG else "new",
            "reward": ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward"),
            "cost": (r.get("agent_result") or {}).get("cost_usd") or 0.0,
            "wall": wall,
        })
  return trials


TAU_TEAM_STAMP = "2026-10-07__18-35-22"
TAU_SOLO_STAMP = "2026-10-07__06-20-57"
TAU_ASTRA_SOLO_STAMP = "2026-10-07__19-12-20"  # GPT-6 Astra alone on the 22 hard tasks
TAU_REP_STAMP = "2026-10-07__20-14-24"  # 2 more solo trials per model on the 17 tasks all four failed alone ("rep": True)
TAU_TASKS = (ROOT / "mw/conditions/tasks_tau_hard22.txt").read_text().split()
TAU_TEAM_TRIALS = 6 * len(TAU_TASKS)


def load_tau():
    """tau3 trials on the hard set: w2w_private teams plus the solo screen trials of the same tasks."""
    trials = []
    for f in sorted(ROOT.glob(f"runs/*_w2w_private-*_tau3_{TAU_TEAM_STAMP}/*/result.json")) + \
             sorted(ROOT.glob(f"runs/solo-*_tau3_{TAU_SOLO_STAMP}/*/result.json")) + \
             sorted(ROOT.glob(f"runs/solo-astra_tau3_{TAU_ASTRA_SOLO_STAMP}/*/result.json")) + \
             sorted(ROOT.glob(f"runs/solo-*_tau3_{TAU_REP_STAMP}/*/result.json")):
        mw = cell(f.parents[1].name.split("_tau3_")[0])
        r = json.loads(f.read_text())
        task = r["task_name"].split("/")[-1]
        if not mw or task not in TAU_TASKS:
            continue
        ae = r.get("agent_execution") or {}
        trials.append({
            "m": mw[0], "w": mw[1], "cond": f.parents[1].name.split("_tau3_")[0],
            "task": task.removeprefix("tau3-bench__tau3-"), "set": "tau", "rep": TAU_REP_STAMP in str(f),
            "reward": ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward"),
            "cost": (r.get("agent_result") or {}).get("cost_usd") or 0.0,
            "wall": (ts(ae["finished_at"]) - ts(ae["started_at"])).total_seconds() / 60 if ae.get("finished_at") else None,
        })
    return trials


trials = load(STAMPS)
w2w = [t for t in load([W2W_STAMP]) if t["m"] != "none"]
w2w_done = sum(t["reward"] is not None for t in w2w)
tau = load_tau()
tau_done = sum(t["reward"] is not None and t["m"] != "none" for t in tau)

Path(__file__).with_name("heatmap_trials.json").write_text(json.dumps(trials))  # read by plot_heatmap.py (star grid)
Path(__file__).with_name("heatmap_trials_w2w.json").write_text(json.dumps(w2w))
Path(__file__).with_name("heatmap_trials_tau.json").write_text(json.dumps(tau))  # read by plot_heatmap.py --tau
tau = [t for t in tau if not t["rep"]]  # the page's live grid keeps one trial per cell and task
stamp_now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
html = (Path(__file__).with_name("heatmap_template.html").read_text()
        .replace("__DATA_TAU__", json.dumps(tau))
        .replace("__TAU_PROGRESS__", "complete" if tau_done >= TAU_TEAM_TRIALS else f"running: {tau_done} of {TAU_TEAM_TRIALS} team trials")
        .replace("__TAU_PILL__", "" if tau_done >= TAU_TEAM_TRIALS else "live")
        .replace("__NTAU__", str(tau_done))
        .replace("__DATA_W2W__", json.dumps(w2w))
        .replace("__DATA__", json.dumps(trials))
        .replace("__NW2W__", str(w2w_done))
        .replace("__W2W_PROGRESS__", "complete" if w2w_done >= 120 else f"running: {w2w_done} of 120 trials")
        .replace("__W2W_PILL__", "" if w2w_done >= 120 else "live")
        .replace("__UPDATED__", stamp_now)
        .replace("__NTRIALS__", str(len(trials))))
out = Path(__file__).with_name("team_heatmap.html")
out.write_text(html)
print(f"{len(trials)} star trials, {w2w_done}/120 w2w_private team trials, {tau_done}/{TAU_TEAM_TRIALS} tau3 team trials -> {out}")
