#!/usr/bin/env python3
"""Merged Figure 3: range width (a) + sample density (b), one two-panel block.

Reads the same result JSONs as experiments/make_figures.py and reuses its
colours, scatter treatment and legend style so the merged panel is visually
identical to the two figures it replaces.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = os.path.dirname(__file__)
FIGS = os.path.join(HERE, "..", "figures")
OUT = os.path.join(FIGS, "fig_merged_range_density.png")

RED, TEAL, GREY = "#c0392b", "#16a085", "#7f8c8d"
LEG = dict(frameon=True, framealpha=0.9, fontsize=7, borderpad=0.3,
           handlelength=1.6, labelspacing=0.25)

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 8,
    "figure.dpi": 300, "savefig.dpi": 300,
})


def load(tag):
    with open(os.path.join(FIGS, f"{tag}_result.json")) as f:
        return json.load(f)


def scatter(ax, x, per, color, jitter):
    rng = np.random.default_rng(20260613)
    for xi, vals in zip(x, per):
        v = np.asarray(vals, dtype=float)
        ax.scatter(xi + rng.uniform(-jitter, jitter, size=v.size), v,
                   s=3, color=color, alpha=0.16, linewidths=0, zorder=1)


def main():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.4, 2.15))

    # ---------------- (a) range width ----------------
    r = load("c2")
    pts = sorted(r["points"], key=lambda p: p["x"])
    x = np.array([p["x"] for p in pts])
    med = np.array([p["median"] for p in pts])
    q1 = np.array([p["q1"] for p in pts]); q3 = np.array([p["q3"] for p in pts])
    ceil = np.array([p["diagnostics"]["ceiling_rho"] for p in pts])
    per = [p["per_realization"] for p in pts]

    scatter(axA, x, per, RED, 0.18)
    axA.fill_between(x, q1, q3, color=RED, alpha=0.18, zorder=2, label="bootstrap IQR")
    axA.plot(x, med, "o-", color=RED, lw=1.4, ms=4, zorder=3,
             label=r"median $\Delta\rho_\uparrow$")
    axA.set_xlabel(r"training-range width $W$ ($\rho$ units)")
    axA.set_ylabel(r"same-class $\Delta\rho_\uparrow$")
    axA.set_xticks(x)
    top = max(max(v) for v in per)
    axA.set_ylim(0, max(top, med.max()) * 1.28)
    axA.grid(alpha=0.25)
    axA2 = axA.twinx()
    axA2.plot(x, ceil, "s--", color=TEAL, lw=1.1, ms=3.5, alpha=0.9, label="absolute ceiling")
    axA2.set_ylabel(r"absolute ceiling $\rho$", color=TEAL)
    axA2.tick_params(axis="y", labelcolor=TEAL)
    h1, l1 = axA.get_legend_handles_labels(); h2, l2 = axA2.get_legend_handles_labels()
    axA.legend(h1 + h2, l1 + l2, loc="upper left", **LEG)

    # ---------------- (b) sample density ----------------
    r = load("c3")
    pts = sorted(r["points"], key=lambda p: p["x"])
    x = np.array([p["x"] for p in pts])
    med = np.array([p["median"] for p in pts])
    q1 = np.array([p["q1"] for p in pts]); q3 = np.array([p["q3"] for p in pts])
    psl = np.array([p["diagnostics"]["per_sample_len"] for p in pts])
    per = [p["per_realization"] for p in pts]

    scatter(axB, x, per, RED, 0.35)
    axB.fill_between(x, q1, q3, color=RED, alpha=0.18, zorder=2, label="bootstrap IQR")
    axB.plot(x, med, "o-", color=RED, lw=1.4, ms=4, zorder=3,
             label=r"median $\Delta\rho_\uparrow$")
    axB.axhline(0.30, ls=":", color="#34495e", lw=0.9, zorder=1)
    axB.set_xlabel(r"number of training samples $M$")
    axB.set_ylabel(r"same-class $\Delta\rho_\uparrow$")
    # log x: the six sample counts are near-uniform in log space, which un-crowds
    # the M = 2, 3 end where linear spacing put the labels on top of each other
    axB.set_xscale("log")
    axB.set_xticks(x)
    axB.set_xticks([], minor=True)
    axB.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(round(v))}"))
    axB.set_xlim(x.min() * 0.82, x.max() * 1.18)
    axB.set_ylim(0, 1.22)
    axB.grid(alpha=0.25)
    for xi, pl in zip(x, psl):
        axB.text(xi, 0.045, f"{pl//1000}k", ha="center", va="bottom",
                 fontsize=6.5, color=GREY)
    axB.legend(loc="upper right", **LEG)

    # panel labels
    for ax, lab in ((axA, "(a)"), (axB, "(b)")):
        ax.text(-0.02, 1.06, lab, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="bottom", ha="left")

    fig.tight_layout(pad=0.5, w_pad=1.6, rect=(0, 0, 1, 0.97))
    fig.savefig(OUT, dpi=300)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
