"""Standalone 'codebook share of board lines' figure (data read from the research log's fcb1 chart)."""
import json, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

src = open("research_update_2026-09-21_to_25.html").read()
d = json.loads(re.search(r'lines\("fcb1",(\{.*?\})\);</script>', src).group(1))
x = d["x"]
k50, bo, nc = (s["values"] for s in d["series"])

BOARD = "Communication  Agents can only talk on one shared public message board, with no private messages"
SERIES = [
    (k50, "#2a78d6", "-", "Run A   k=50 agents, n=1000 days",
     [BOARD, "Money goal  The prompt tells agents to finish with the highest balance they can",
      "Run length  Agents are told the run lasts 1000 days"]),
    (bo,  "#eb6834", "-", "Run B   k=50 agents, n=500 days",
     [BOARD, "Money goal  The prompt tells agents to finish with the highest balance they can",
      "Run length  Agents are told the run lasts 500 days"]),
    (nc,  "#1baf7a", ":", "Run C   k=50 agents, n=500 days",
     [BOARD, "Money goal  None, the prompt never tells agents to finish with the highest balance",
      "Run length  Agents are told the run lasts 500 days"]),
]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(12, 9.8))
for vals, c, ls, name, desc in SERIES:
    xs = [xi for xi, v in zip(x, vals) if v is not None]
    ys = [v for v in vals if v is not None]
    ax.plot(xs, ys, color=c, ls=ls, lw=2.6, label=name + "\n" + "\n".join(desc), zorder=5 if ls==":" else 3, clip_on=False)
    ax.plot(xs[-1], ys[-1], "o", color=c, ms=7, mec="white", mew=1.5, zorder=4, clip_on=False)
    ax.annotate(f"{ys[-1]:.0%}", (xs[-1], ys[-1]), xytext=(9, 0), textcoords="offset points",
                va="center", fontsize=11, fontweight="bold", color=c)

for day, txt in [(110, "Run B ledger\nspreads (day 110)"), (300, "Run A ledger\nappears (day 300)")]:
    ax.axvline(day, color="#7d8683", ls=":", lw=1.2, zorder=1)
    ax.text(day + 4, 1.035, txt, fontsize=9.5, color="#4b5553", va="bottom")

ax.set_xlim(0, 500); ax.set_ylim(0, 1.0)
ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
ax.set_yticks([0, .25, .5, .75, 1])
ax.set_xlabel("simulated day")
ax.set_ylabel("share of all board lines")
ax.grid(axis="y", color="#e1e0d9"); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.spines["left"].set_color("#c3c2b7"); ax.spines["bottom"].set_color("#c3c2b7")
ax.tick_params(colors="#4b5553")

fig.suptitle("Share of board lines written in the run's accounting codebook",
             x=0.06, ha="left", fontsize=14, fontweight="bold", y=0.985)
fig.text(0.06, 0.925, "50 gpt-5.5 agents per run on a public message board. Blue and orange are the same kind of debt ledger "
         "in two separate runs, written in different words.", fontsize=10, color="#4b5553")

leg = fig.legend(loc="lower left", ncol=1, frameon=True, fontsize=11.5, handlelength=3.2,
                 bbox_to_anchor=(0.08, 0.005, 0.87, 0.1), mode="expand", borderaxespad=0, title="What each line shows",
                 title_fontproperties={"weight": "bold", "size": 12}, alignment="left",
                 borderpad=0.8, labelspacing=1.1)
leg.get_frame().set_edgecolor("#dde1de")
for t in leg.get_texts():
    t.set_color("#151a19")
fig.subplots_adjust(left=0.08, right=0.95, top=0.84, bottom=0.41)
for ext in ("png", "svg"):
    fig.savefig(f"codebook_share.{ext}", dpi=200, facecolor="white")
print("wrote codebook_share.png / .svg")
