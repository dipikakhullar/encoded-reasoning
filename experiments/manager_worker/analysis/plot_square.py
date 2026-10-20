"""Square manager x worker heatmaps (same models, same order on both axes) -> analysis/square_<layout>_<grid>.png.

    uv run --no-project --with matplotlib python analysis/plot_square.py A star   # A: 4x4 + "alone" margin column
    uv run --no-project --with matplotlib python analysis/plot_square.py B star   # B: 5x5 with "nobody" on both axes

grid: star (heatmap_trials.json), w2w (heatmap_trials_w2w.json teams + the 6 Oct solo trials), or tau
(heatmap_trials_tau.json: tau3-bench 22 hard tasks, w2w_private teams + solo trials; -> heatmap_tau.png).
Models are ordered by solo pass rate, so the diagonal is a model managing copies of itself and the upper
triangle is a stronger manager with weaker workers.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

HERE = Path(__file__).parent
MODELS = [("astra", "GPT-6\nAstra"), ("ds4", "DeepSeek\nV4 Pro"), ("glm", "GLM-5.3"), ("qwen", "Qwen 3.8\n27B")]
layout, grid = (sys.argv[1:3] + ["A", "star"][len(sys.argv[1:3]):])
trials = json.loads((HERE / "heatmap_trials.json").read_text())
if grid == "tau":
    trials = json.loads((HERE / "heatmap_trials_tau.json").read_text())
elif grid == "w2w":
    trials = [t for t in trials if t["m"] == "none"] + json.loads((HERE / "heatmap_trials_w2w.json").read_text())
trials = [t for t in trials if t["reward"] is not None]


def agg(m, w):
    r = [t for t in trials if t["m"] == m and t["w"] == w]
    if not r:
        return None
    return {"pass": sum(t["reward"] == 1 for t in r) / len(r), "n": len(r), "k": sum(t["reward"] == 1 for t in r),
            "cost": sum(t["cost"] for t in r) / len(r), "wall": sum(t["wall"] or 0 for t in r) / len(r)}


solo = {m: agg("none", m) for m, _ in MODELS}
team = {(m, w): agg(m, w) for m, _ in MODELS for w, _ in MODELS}
PANELS = [("pass", "Pass@1 (higher is better)", "Blues", lambda s: f"{s['pass']:.0%}\n{s['k']}/{s['n']}"),
          ("cost", "$ per task (lower is better)", "Oranges", lambda s: f"${s['cost']:.2f}"),
          ("wall", "Wall-clock, min (lower is better)", "Greens", lambda s: f"{s['wall']:.1f}")]
n = len(MODELS)
labels = [l for _, l in MODELS]


def draw_cell(ax, x, y, s, key, fmt, cmap, vmax):
    if s is None:
        ax.add_patch(Rectangle((x - .5, y - .5), 1, 1, facecolor="#f3f2ee", edgecolor="#d9d8d2", hatch="///", lw=0))
        ax.text(x, y, "not run", ha="center", va="center", fontsize=8, color="#8a8984")
        return
    v = s[key] / vmax
    ax.add_patch(Rectangle((x - .5, y - .5), 1, 1, facecolor=plt.get_cmap(cmap)(0.08 + 0.92 * v), lw=0))
    ax.text(x, y, fmt(s), ha="center", va="center", fontsize=10, color="white" if v > 0.55 else "black")


fig, axes = plt.subplots(1, 3, figsize=(19, 6.4), constrained_layout=True)
for ax, (key, title, cmap, fmt) in zip(axes, PANELS):
    vals = [s[key] for s in list(solo.values()) + list(team.values()) if s]
    vmax = 1.0 if key == "pass" else max(vals)
    if layout == "A":
        # 4x4 square, then a gap column, then the "alone" column (each row's model working alone)
        for i, (m, _) in enumerate(MODELS):
            for j, (w, _) in enumerate(MODELS):
                draw_cell(ax, j, i, team[(m, w)], key, fmt, cmap, vmax)
            draw_cell(ax, n + .6, i, solo[m], key, fmt, cmap, vmax)
        ax.plot([-.5, n - .5], [-.5, n - .5], color="#0b0b0b", lw=1, ls=(0, (3, 3)), alpha=.45)  # diagonal
        ax.set_xlim(-.5, n + 1.1)
        ax.set_xticks(list(range(n)) + [n + .6], labels + ["works\nalone"])
        ax.axvline(n + .05, color="#ffffff", lw=6)
        ax.set_yticks(range(n), labels)
        ax.set_ylim(n - .5, -.5)
        ax.set_xlabel("Workers (or the row's model alone)")
        ax.set_ylabel("Manager")
    else:
        # 5x5 with "nobody" first on both axes: (nobody, Y) = Y alone, (X, nobody) = X alone
        axis = [("none", "nobody")] + MODELS
        for i, (m, _) in enumerate(axis):
            for j, (w, _) in enumerate(axis):
                if m == "none" and w == "none":
                    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor="#ffffff", edgecolor="#d9d8d2", lw=1))
                    continue
                s = solo[w] if m == "none" else solo[m] if w == "none" else team[(m, w)]
                draw_cell(ax, j, i, s, key, fmt, cmap, vmax)
        ax.plot([.5, n + .5], [.5, n + .5], color="#0b0b0b", lw=1, ls=(0, (3, 3)), alpha=.45)
        ax.axhline(.5, color="#ffffff", lw=4)
        ax.axvline(.5, color="#ffffff", lw=4)
        ax.set_xlim(-.5, n + .5)
        ax.set_ylim(n + .5, -.5)
        ax.set_xticks(range(n + 1), ["nobody\n(manager\nalone)"] + labels)
        ax.set_yticks(range(n + 1), ["nobody\n(worker alone)"] + labels)
        ax.set_xlabel("Workers")
        ax.set_ylabel("Manager")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
name = {"A": "Option A: 4x4 square + 'works alone' column", "B": "Option B: 5x5 with 'nobody' on both axes"}[layout]
topo = {"star": "star (workers talk only to the manager), 6 Oct", "w2w": "workers can message each other, 7 Oct",
        "tau": "workers can message each other, 7 Oct"}[grid]
if grid == "tau":
    fig.suptitle("tau3-bench, 22 tasks GLM/DS4/Qwen all failed alone · manager × workers, " + topo
                 + " · 1 trial per task, all finished trials · dashed line = model manages copies of itself", fontsize=12)
else:
    fig.suptitle(f"{name} · Terminal-Bench 2.1, {topo} · dashed line = model manages copies of itself", fontsize=12)
out = HERE / ("heatmap_tau.png" if grid == "tau" else f"square_{layout}_{grid}.png")
fig.savefig(out, dpi=130, facecolor="white")
print(out)
