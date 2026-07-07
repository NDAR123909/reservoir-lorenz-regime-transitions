"""
Figure 5, attractor climate at unseen rho, true Lorenz vs ESN cold extrapolation.

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

Deterministic end to end on the pinned stack (see REPRODUCIBILITY.md). Run:
    python make_attractor_figure.py          # ~40 s, writes figures/fig5_attractor_climate.png
"""
import sys, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    fig = plt.figure(figsize=(11.5, 7.2))
    for col, rho in enumerate(RHO_SHOW):
        x0 = ic_rng.uniform(-15, 15, size=3)
        true_tr = lorenz.integrate_esn_grid(rho, N_FREE, transient_time=80.0, x0=x0)

        j = int(np.argmin(np.abs(train_rhos - rho)))
        esn_tr = esn.cold_extrapolate(rho, n_free=N_FREE, discard=DISCARD,
                                      seed=SEED, primer_hat=primers[j])

        rel = ("just above the Hopf" if col == 0 else
               "inside the training range" if col == 1 else
               "beyond the training range")
        for row, (tr, c, tag) in enumerate(
                [(true_tr, TRUTH_C, "true Lorenz"),
                 (esn_tr, ESN_C, "ESN, cold extrapolation")]):
            ax = fig.add_subplot(2, 3, row * 3 + col + 1, projection="3d")
            ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=c, lw=0.35, alpha=0.85)
            ax.set_title(f"{tag}\n$\\rho={rho:g}$ ({rel})" if row == 0
                         else tag, fontsize=8.5, pad=-2)
            ax.set_xlabel("x", fontsize=7, labelpad=-6)
            ax.set_ylabel("y", fontsize=7, labelpad=-6)
            ax.set_zlabel("z", fontsize=7, labelpad=-6)
            ax.tick_params(labelsize=6, pad=-2)
            ax.view_init(elev=18, azim=-60)
            ax.set_box_aspect((1, 1, 0.9))
        print(f"[fig5] rho={rho:g} done ({time.time()-t0:.0f}s)")

    fig.suptitle("Attractor climate at unseen $\\rho$, true system vs ESN cold "
                 "extrapolation\n(free-run geometry comparison, one realization, "
                 "training $\\rho\\in$ {24.56, 26.06, 27.56, 29.06})",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT, dpi=140)
    print(f"[fig5] figure -> {os.path.relpath(OUT)}   ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
