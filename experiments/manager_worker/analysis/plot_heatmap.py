"""Static matplotlib version of team_heatmap.html: three manager x worker heatmaps -> analysis/heatmap.png.

    python3 analysis/build_heatmap.py                                  # writes heatmap_trials.json
    uv run --with matplotlib python analysis/plot_heatmap.py [--all] [--w2w]   # default: tasks every cell finished

--w2w plots the 7 Oct workers-can-message grid (6 Oct solo row + heatmap_trials_w2w.json) -> heatmap_w2w.png.
--tau plots the tau3-bench hard-task grid (heatmap_trials_tau.json, workers can message each other) -> heatmap_tau.png.
--tau --allfail keeps only the tau3 tasks every solo model failed, GPT-6 Astra included -> heatmap_tau_allfail.png.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

HERE = Path(__file__).parent
MANAGERS = [("none", "No manager\n(solo)"), ("astra", "GPT-6 Astra"), ("ds4", "DeepSeek V4 Pro")]
WORKERS = [("astra", "GPT-6\nAstra"), ("ds4", "DeepSeek\nV4 Pro"), ("glm", "GLM-5.3"), ("qwen", "Qwen 3.8\n27B")]  # same order as the manager rows
CELLS = [(m, w) for m, _ in MANAGERS for w, _ in WORKERS if m == "none" or w != "astra"]

W2W = "--w2w" in sys.argv
TAU = "--tau" in sys.argv
ALLFAIL = "--allfail" in sys.argv
trials = json.loads((HERE / "heatmap_trials.json").read_text())
if TAU:
    trials = json.loads((HERE / "heatmap_trials_tau.json").read_text())
elif W2W:
    trials = [t for t in trials if t["m"] == "none"] + json.loads((HERE / "heatmap_trials_w2w.json").read_text())
trials = [t for t in trials if t["reward"] is not None]
if ALLFAIL:
    # The 17 tasks all four models failed on their first solo trial; solo cells add the 2 repeat trials per task.
    allfail = {k.removeprefix("tau3-bench__tau3-")
               for k in (HERE.parent / "mw/conditions/tasks_tau_allfail17.txt").read_text().split()}
    trials = [t for t in trials if t["task"] in allfail]
elif TAU:
    trials = [t for t in trials if not t.get("rep")]
if "--all" not in sys.argv:
    done = {}
    for t in trials:
        done.setdefault(t["task"], set()).add((t["m"], t["w"]))
    present = [c for c in CELLS if any((t["m"], t["w"]) == c for t in trials)]  # cells never run don't drop tasks
    keep = {k for k, v in done.items() if all(c in v for c in present)}
    trials = [t for t in trials if t["task"] in keep]
n_tasks = len({t["task"] for t in trials})

grids = {k: np.full((len(MANAGERS), len(WORKERS)), np.nan) for k in ("pass", "cost", "wall")}
counts = np.zeros((len(MANAGERS), len(WORKERS)), dtype=object)
for i, (m, _) in enumerate(MANAGERS):
    for j, (w, _) in enumerate(WORKERS):
        r = [t for t in trials if t["m"] == m and t["w"] == w]
        if not r:
            continue
        passed = sum(t["reward"] == 1 for t in r)
        grids["pass"][i, j] = passed / len(r)
        grids["cost"][i, j] = sum(t["cost"] for t in r) / len(r)
        grids["wall"][i, j] = sum(t["wall"] for t in r) / len(r)
        counts[i, j] = f"{passed}/{len(r)}"

panels = [("pass", "Pass@1 (higher is better)", "Blues", lambda v: f"{v:.0%}", (0, 1)),
          ("cost", "$ per task (lower is better)", "Oranges", lambda v: f"${v:.2f}", None),
          ("wall", "Wall-clock, min (lower is better)", "Greens", lambda v: f"{v:.1f}", None)]
fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
for ax, (key, title, cmap, fmt, lim) in zip(axes, panels):
    g = grids[key]
    vmin, vmax = lim or (0, np.nanmax(g))
    if key == "pass":  # 0% is drawn pure white, not the colormap's faint first shade
        cmap = ListedColormap(np.vstack([[1, 1, 1, 1], plt.get_cmap(cmap)(np.linspace(0, 1, 256))[1:]]))
    im = ax.imshow(np.ma.masked_invalid(g), cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_facecolor("#eeeeee")
    for i in range(len(MANAGERS)):
        for j in range(len(WORKERS)):
            v = g[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", color="#888888", fontsize=12)
                continue
            dark = (v - vmin) / (vmax - vmin or 1) > 0.55
            label = fmt(v) + (f"\n{counts[i, j]}" if key == "pass" else "")
            ax.text(j, i, label, ha="center", va="center", fontsize=11, color="white" if dark else "black")
    ax.set_xticks(range(len(WORKERS)), [w for _, w in WORKERS])
    ax.set_yticks(range(len(MANAGERS)), [m for _, m in MANAGERS])
    ax.set_xlabel("Worker (or solo model)")
    ax.set_ylabel("Manager")
    ax.set_title(title)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.62, pad=0.03)
scope = "all finished trials" if "--all" in sys.argv else f"{n_tasks} tasks every cell has finished"
topo = "workers can message each other" if (W2W or TAU) else "workers message only the manager, not each other"
bench = ("tau3-bench, 17 tasks every model failed alone on its first try" if ALLFAIL
         else "tau3-bench, 22 tasks GLM/DS4/Qwen all failed alone" if TAU else "Terminal-Bench 2.1")
trials_note = "teams 1 trial per task, solo 3 trials per task" if ALLFAIL else "1 trial per task"
fig.suptitle(f"Manager × worker on {bench}, {topo}, {trials_note} ({scope})", fontsize=13)
out = HERE / ("heatmap_tau_allfail.png" if ALLFAIL else "heatmap_tau.png" if TAU else "heatmap_w2w.png" if W2W else "heatmap.png")
fig.savefig(out, dpi=150, facecolor="white")
print(f"{n_tasks} tasks -> {out}")
