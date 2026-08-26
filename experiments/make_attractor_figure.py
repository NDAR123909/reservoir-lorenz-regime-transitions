"""
Manuscript Figure 2 (file name fig5_* predates the manuscript numbering):
attractor climate at unseen rho, true Lorenz vs ESN cold extrapolation.

A qualitative companion to C1. The quantitative pass lives in the z-maxima
metrics (run_c1_v2.py), and this figure shows the object those statistics
summarize.
One reservoir realization is trained exactly as in C1 (locked config, master
seed, realization 0, the four training rho), then free-runs at three rho values
the network never saw, one just above the Hopf, one in the interior of the
training range, and one beyond its upper edge. The comparison is CLIMATE, not
tracking. Past the valid prediction time a free-running ESN diverges from the
true trajectory pointwise by construction, so the claim on display is that the
reconstructed attractor has the right geometry, not that the paths overlay.

Panels are labelled (a)-(f) in reading order, as JURPA requires for multi-part
figures; which rho and which of truth/ESN each panel shows is carried by the
caption. Rendered at 5.0 in wide -- INSERT AT 5.0 in, not 3.5 in, or the panel
type drops below legibility.

Deterministic end to end on the pinned stack (see REPRODUCIBILITY.md). Run:
    python make_attractor_figure.py          # ~40 s, writes figures/fig5_attractor_climate.png
"""
import sys, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Sized for a 5.0 in printed width at 100% insertion, so authored point sizes
# land 1:1 on the page against ~10 pt body type.
plt.rcParams.update({
    "font.size":        8,
    "axes.labelsize":   8,
    "axes.titlesize":   8,
    "xtick.labelsize":  6.5,
    "ytick.labelsize":  6.5,
    "legend.fontsize":  6.5,
    "figure.dpi":       300,
    "savefig.dpi":      300,
})

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import lorenz, training
from reservoir import ESNConfig

MASTER = 20260613
RHO_TRAIN = [24.56, 26.06, 27.56, 29.06]     # the locked C1 training set
L_TOTAL = 120_000                            # fixed data budget (methodology 4)
RHO_SHOW = [25.0, 28.0, 31.0]                # unseen, near Hopf / interior / beyond range
N_FREE, DISCARD, SEED = 6000, 1500, 0        # identical to predicted_bifurcation
OUT = os.path.join(os.path.dirname(__file__), "..", "figures",
                   "fig5_attractor_climate.png")

TRUTH_C = "#5a5a5a"
ESN_C = "#c0392b"


def main():
    t0 = time.time()

    # train realization 0 exactly as run_c1 does
    segments = training.build_segments(RHO_TRAIN, L_TOTAL, transient_time=80.0,
                                       ic_seed=MASTER + 1)
    esn = training.train_realization(ESNConfig(seed=MASTER), segments)
    primers = [esn.standardize(seg[0][-1500:]) for seg in segments]
    train_rhos = np.array([r for _, r in segments])
    print(f"[fig5] realization 0 trained ({time.time()-t0:.0f}s)")

    # deterministic true trajectories, one seeded IC per rho
    ic_rng = np.random.default_rng(MASTER + 777)
    fig = plt.figure(figsize=(5.0, 3.4))
    for col, rho in enumerate(RHO_SHOW):
        x0 = ic_rng.uniform(-15, 15, size=3)
        true_tr = lorenz.integrate_esn_grid(rho, N_FREE, transient_time=80.0, x0=x0)

        j = int(np.argmin(np.abs(train_rhos - rho)))
        esn_tr = esn.cold_extrapolate(rho, n_free=N_FREE, discard=DISCARD,
                                      seed=SEED, primer_hat=primers[j])

        # row 0 = true Lorenz, row 1 = ESN cold extrapolation; the caption keys
        # (a)-(f) to the row/rho pairing.
        for row, (tr, c) in enumerate([(true_tr, TRUTH_C), (esn_tr, ESN_C)]):
            ax = fig.add_subplot(2, 3, row * 3 + col + 1, projection="3d")
            # rasterized so the vector PDF stays small; the text stays vector
            ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=c, lw=0.35, alpha=0.85,
                    rasterized=True)
                        ax.set_xlabel("x", fontsize=8, labelpad=-9)
            ax.set_ylabel("y", fontsize=8, labelpad=-9)
            ax.set_zlabel("z", fontsize=8, labelpad=-5)
            # One frame for all six panels, so (a)-(c) and (d)-(f) are directly
            # comparable and the tick choice does not vary panel to panel.
            ax.set_xlim(-22, 22)
            ax.set_ylim(-30, 30)
            ax.set_zlim(0, 55)
            # x and y carry no quantitative claim here, and their numerals
            # collide at the front corner at this panel size; only z is labelled.
            ax.set_xticks([-20, 0, 20]); ax.set_xticklabels([])
            ax.set_yticks([-20, 0, 20]); ax.set_yticklabels([])
            ax.set_zticks([0, 20, 40])
            ax.tick_params(axis="z", labelsize=6.5, pad=-1)
            ax.view_init(elev=18, azim=-60)
            ax.set_box_aspect((1, 1, 0.9))
            ax.text2D(0.02, 0.95, f"({'abcdef'[row * 3 + col]})",
                      transform=ax.transAxes, fontsize=9, fontweight="bold")
        print(f"[fig5] rho={rho:g} done ({time.time()-t0:.0f}s)")

    # matplotlib's layout engines mis-measure 3-D bounding boxes, which clips the
    # right column's z labels; place the grid explicitly instead.
    fig.subplots_adjust(left=0.00, right=0.95, top=0.98, bottom=0.05,
                        wspace=0.10, hspace=0.14)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.splitext(OUT)[0] + "." + ext, dpi=300)
    print(f"[fig5] figure -> {os.path.relpath(OUT)} (+ .pdf)   ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
