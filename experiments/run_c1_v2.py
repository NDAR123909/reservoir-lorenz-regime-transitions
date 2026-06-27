"""
run_c1_v2.py
============
C1 re-validation under methodology v2 (Option A): the model is unchanged from the
Session-5 best-found configuration (gamma_p=0.1, spectral_radius=0.6, everything
else at section 1.5); only the amplitude criterion's evaluation domain changes
(z-maxima RMSE scoped to the chaotic band rho >= 24.74, training.HOPF_RHO).

Because background jobs do not persist in this environment, realizations are
checkpointed one at a time to data/c1v2_preds/ and the run resumes where it left
off. `--mode finalize` aggregates whatever is on disk, applies the v2 acceptance
test, writes the result JSON, and draws Figure 1.

    python run_c1_v2.py --mode run --upto 10 --batch 5
    python run_c1_v2.py --mode finalize --upto 10
"""
import sys, os, json, time, argparse, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lorenz, training
from reservoir import ESNConfig

RHO_TRAIN = [24.56, 26.06, 27.56, 29.06]
RHO_GRID = np.round(np.arange(24.0, 32.0 + 1e-9, 0.1), 2)
L_TOTAL = 120_000
MASTER = 20260613
HERE = os.path.dirname(__file__)
PRED_DIR = os.path.join(HERE, "..", "data", "c1v2_preds")
CACHE = os.path.join(HERE, "..", "data", f"ground_truth_{MASTER+777}.pkl")
LOCKED = dict(gamma_p=0.1, spectral_radius=0.6)   # Session-5 best-found


def _load_truth():
    with open(CACHE, "rb") as f:
        return pickle.load(f)


def run(upto, batch, n_free):
    os.makedirs(PRED_DIR, exist_ok=True)
    truth = _load_truth()
    segments = training.build_segments(RHO_TRAIN, L_TOTAL, transient_time=80.0,
                                       ic_seed=MASTER + 1)
    done = 0
    t0 = time.time()
    for k in range(upto):
        path = os.path.join(PRED_DIR, f"real_{k:02d}.pkl")
        if os.path.exists(path):
            continue
        cfg = ESNConfig(seed=MASTER + k, **LOCKED)
        esn = training.train_realization(cfg, segments)
        pred = training.predicted_bifurcation(esn, RHO_GRID, segments,
                                              n_free=n_free, discard=1200,
                                              seed=MASTER + k)
        with open(path, "wb") as f:
            pickle.dump(pred, f)
        done += 1
        print(f"[c1v2] realization {k} saved ({time.time()-t0:.0f}s)")
        if done >= batch:
            break
    n_on_disk = len([f for f in os.listdir(PRED_DIR) if f.endswith(".pkl")])
    print(f"[c1v2] {n_on_disk}/{upto} realizations on disk")


def finalize(upto, n_free):
    truth = _load_truth()
    preds = []
    for k in range(upto):
        path = os.path.join(PRED_DIR, f"real_{k:02d}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                preds.append(pickle.load(f))
    R = len(preds)
    agg = training.aggregate_realizations(preds)
    m = training.c1_metrics(agg, truth)                 # rmse_rho_min defaults to HOPF
    passed, parts = training.c1_pass(m)
    m_v1 = training.c1_metrics(agg, truth, rmse_rho_min=None)  # full-grid, for the record

    print(f"\n[c1v2] ----- acceptance (methodology v2, section 7) -----   R={R}")
    print(f"  regime-class accuracy : {m['class_acc']*100:5.1f}%  "
          f"(>=95%? {parts['class']})  [{m['n_class_wrong']}/{m['n_rho']} wrong]")
    print(f"  z-maxima RMSE (rho>={m['rmse_rho_min']}) : "
          f"{m['zmax_rmse_frac']*100:.2f}% of z-range {m['z_range']:.2f}  "
          f"over {m['n_rmse_points']} pts  (<=5%? {parts['rmse']})")
    print(f"  lyap/spread agreement : {m['lyap_proxy_err']*100:.1f}%  "
          f"(<=10%? {parts['lyap']})")
    print(f"  [for the record] full-grid (v1) RMSE: {m_v1['zmax_rmse_frac']*100:.2f}%")
    print(f"  C1 {'PASS' if passed else 'FAIL'}\n")

    outdir = os.path.join(HERE, "..", "figures")
    os.makedirs(outdir, exist_ok=True)
    _plot(agg, truth, m, passed, os.path.join(outdir, "fig1_c1_v2_bifurcation.png"))

    result = {"tag": "c1_v2", "master_seed": MASTER, "realizations": R,
              "locked_cfg": LOCKED, "revision": "methodology v2 / Option A",
              "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating))
                              else v) for k, v in m.items()},
              "full_grid_rmse_frac": float(m_v1["zmax_rmse_frac"]),
              "passed": bool(passed), "parts": {k: bool(v) for k, v in parts.items()},
              "rho_train": RHO_TRAIN, "n_free": n_free}
    with open(os.path.join(outdir, "c1_v2_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[c1v2] result -> figures/c1_v2_result.json")
    return result


def _plot(agg, truth, m, passed, path):
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    for rho, zm in zip(truth["rho"], truth["zmax"]):
        if zm.size:
            ax.plot(np.full(zm.size, rho), zm, ".", color="0.55", ms=1.4,
                    alpha=0.5, rasterized=True)
    for rho, zm in zip(agg["rho"], agg["zmax"]):
        if zm.size:
            ax.plot(np.full(zm.size, rho), zm, ".", color="#c0392b", ms=1.0,
                    alpha=0.35, rasterized=True)
    ax.axvline(training.HOPF_RHO, color="#2c3e50", lw=1.0, ls="--", alpha=0.8)
    ax.axvspan(RHO_GRID.min(), training.HOPF_RHO, color="0.85", alpha=0.35, zorder=0)
    _y0, _y1 = ax.get_ylim()
    ax.text(24.84, 0.80, "Hopf  $\\rho\\approx24.74$",
            transform=ax.get_xaxis_transform(),
            fontsize=8, rotation=0, va="top", ha="left", color="#2c3e50",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.9))
    ax.annotate("coexistence region\n(excluded from amplitude RMSE)",
                xy=(24.37, _y0 + 0.13 * (_y1 - _y0)),
                xytext=(25.55, _y0 + 0.05 * (_y1 - _y0)),
                fontsize=7, va="center", ha="left", color="0.4",
                arrowprops=dict(arrowstyle="->", color="0.5", lw=0.8))
    for rt in RHO_TRAIN:
        ax.axvline(rt, color="#16a085", lw=0.8, ls=":", alpha=0.6)
    ax.plot([], [], ".", color="0.55", ms=6, label="ground truth")
    ax.plot([], [], ".", color="#c0392b", ms=6, label="ESN (cold extrapolation)")
    ax.plot([], [], ":", color="#16a085", label="training $\\rho$")
    ax.set_xlabel("$\\rho$ (Rayleigh parameter)")
    ax.set_ylabel("successive $z$-maxima")
    ax.set_title(f"C1 reproduction (methodology v2), predicted vs true "
                 f"({'PASS' if passed else 'FAIL'})\n"
                 f"class acc {m['class_acc']*100:.1f}% (full grid), "
                 f"z-max RMSE {m['zmax_rmse_frac']*100:.2f}% of range "
                 f"($\\rho\\geq24.74$)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[c1v2] figure -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["run", "finalize"], default="run")
    ap.add_argument("--upto", type=int, default=10)
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--n-free", type=int, default=4000)
    args = ap.parse_args()
    if args.mode == "run":
        run(args.upto, args.batch, args.n_free)
    else:
        finalize(args.upto, args.n_free)
