import matplotlib as mpl
import matplotlib.pyplot as plt

import analysis
import movement

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.labelsize": 10, "axes.labelcolor": INK2, "axes.edgecolor": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
})

mk, cd = analysis.load()
d24 = analysis.snapshot(24, mk, cd)
live = d24[d24["live"]]
labels = [l for _, l in analysis.HORIZONS]


def save(fig, name):
    fig.savefig(f"figures/{name}")
    plt.close(fig)
    print(f"wrote figures/{name}")


def errbars(ax, x, tab, col, color, label=None):
    ax.errorbar(x, tab[col], yerr=[tab[col] - tab[f"{col}_lo"], tab[f"{col}_hi"] - tab[col]],
                fmt="o-", ms=6, lw=1.6, color=color, mec="white", mew=1.2, capsize=0, label=label)


ht = analysis.horizon_table(mk, cd)
cohort = ht[ht["sample"] == "cohort"].set_index("horizon").loc[labels]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
for ax, col, title in zip(axes, ["brier", "res", "rel"],
                          ["Brier score", "Resolution (higher = sharper)", "Reliability (lower = better calibrated)"]):
    errbars(ax, range(6), cohort, col, BLUE)
    ax.set_xticks(range(6)), ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylim(0, None)
axes[0].set_xlabel("time before close")
fig.suptitle("Brier score and its components by horizon, fixed cohort of 4,555 markets quoted six days out",
             fontsize=11, fontweight="bold", y=1.03)
save(fig, "horizon_decomposition.png")


ct = analysis.category_table(mk, cd)
cats = list(analysis.KALSHI_TABLE1)
fig, axes = plt.subplots(1, 5, figsize=(13, 3.2), sharey=True)
for ax, cat in zip(axes, cats):
    t = ct[ct["category"] == cat]
    x = range(4)
    ax.errorbar(x, t["brier"], yerr=[t["brier"] - t["lo"], t["hi"] - t["brier"]], fmt="o-", ms=6,
                lw=1.6, color=BLUE, mec="white", mew=1.2, capsize=0, label="this study (95% CI)")
    ax.plot(x, t["kalshi"], "s--", ms=5, lw=1.4, color=ORANGE, label="Kalshi Research, Table 1")
    ax.set_xticks(x), ax.set_xticklabels(["6d / 1wk", "1d", "1h", "close"], fontsize=8.5)
    ax.set_title(cat.replace("Science and Technology", "Sci & Tech"))
    ax.set_ylim(0, 0.15)
axes[0].set_ylabel("Brier score")
axes[0].legend(loc="upper right", fontsize=8.5)
save(fig, "category_replication.png")


vt = analysis.volume_table(d24)
buckets = ["<10K", "10-50K", "50-200K", ">=200K"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.4))
for sample, color in [("all", BLUE), ("live", ORANGE)]:
    t = vt[vt["sample"] == sample].set_index("bucket").loc[buckets]
    errbars(ax1, range(4), t, "brier", color, label={"all": "all markets", "live": "live markets only"}[sample])
ax1.set_xticks(range(4)), ax1.set_xticklabels(buckets)
ax1.set_xlabel("event dollar volume"), ax1.set_ylabel("Brier score at 24h"), ax1.set_ylim(0, 0.21)
ax1.legend(loc="center right")
ax1.set_title("Brier score by event volume")
share = vt[vt["sample"] == "all"].set_index("bucket").loc[buckets, "live_share"]
ax2.bar(range(4), share, color=BLUE, width=0.6)
for i, s in enumerate(share):
    ax2.text(i, s + 0.012, f"{s:.0%}", ha="center", color=INK2, fontsize=9)
ax2.set_xticks(range(4)), ax2.set_xticklabels(buckets)
ax2.set_xlabel("event dollar volume"), ax2.set_ylabel("share live at 24h"), ax2.set_ylim(0, 0.65)
fig.suptitle("Brier score and share of live markets by event volume at the 24-hour horizon",
             fontsize=11, fontweight="bold", y=1.03)
ax2.set_title("Share of markets still uncertain at 24h")
save(fig, "volume_composition.png")


fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
for ax, d, title in [(axes[0], live, f"Live at 24h (n={len(live)})"),
                     (axes[1], live[live["liquid"]], f"Live, spread <= 10c (n={int(live['liquid'].sum())})")]:
    t = analysis.reliability_table(d)
    ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color=MUTED)
    ax.errorbar(t["mean_price"], t["freq_yes"], yerr=[t["freq_yes"] - t["ci_lo"], t["ci_hi"] - t["freq_yes"]],
                fmt="o", ms=6, lw=1.4, color=BLUE, mec="white", mew=1.2, capsize=0)
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.set_aspect("equal")
    ax.set_xlabel("mid price"), ax.set_title(title)
axes[0].set_ylabel("share resolving yes")
save(fig, "reliability.png")


st = analysis.spread_table(live)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.4), sharex=True)
for ax, col, ref, title in [(ax1, "b", 1.0, "slope b (1 = calibrated shape)"),
                            (ax2, "a", 0.0, "intercept a (0 = no tilt)")]:
    ax.axhline(ref, ls="--", lw=1.2, color=MUTED)
    errbars(ax, range(len(st)), st, col, BLUE)
    ax.set_xticks(range(len(st))), ax.set_xticklabels([f"<= {c:.2f}" for c in st["max_spread"]])
    ax.set_xlabel("maximum bid-ask spread ($)"), ax.set_title(title)
fig.suptitle("Logistic fit on live markets at 24h by maximum bid-ask spread", fontsize=11,
             fontweight="bold", y=1.03)
save(fig, "logistic_spread.png")


mt = movement.movement_table(mk, cd, starts=(144,))
fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.axhline(1.0, ls="--", lw=1.2, color=MUTED)
for quotes, color, label in [("all", ORANGE, "all quotes"), ("liquid", BLUE, "liquid quotes (spread <= 10c)")]:
    t = mt[mt["quotes"] == quotes].set_index("step_h").loc[[1, 6, 24]]
    errbars(ax, range(3), t, "total", color, label=label)
ax.set_xticks(range(3)), ax.set_xticklabels(["hourly", "every 6h", "daily"])
ax.set_xlabel("sampling of the price path"), ax.set_ylabel("$R$")
ax.set_ylim(0.7, 1.7)
ax.legend(loc="upper right")
ax.set_title("Price paths from six days out to settlement")
save(fig, "movement.png")
