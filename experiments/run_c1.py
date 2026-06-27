"""
run_c1.py
=========
C1 gate (methodology C1, acceptance in section 7). Train the parameter-aware ESN
on rho in {24.56, 26.06, 27.56, 29.06}, reconstruct the z-maxima bifurcation
diagram over rho in [24, 32] by cold extrapolation, aggregate over reservoir
realizations, and test the section-7 acceptance criterion.

Usage:
    python run_c1.py --realizations 32 --master-seed 20260613
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lorenz
import training
from reservoir import ESNConfig

RHO_TRAIN = [24.56, 26.06, 27.56, 29.06]
RHO_GRID = np.round(np.arange(24.0, 32.0 + 1e-9, 0.1), 2)
L_TOTAL = 120_000          # total training data length (methodology 4)


def run(realizations=32, master_seed=20260613, cfg_overrides=None,
        n_free=5000, discard=1200, outdir="../figures", tag="c1",
        rmse_full_grid=False):
    t0 = time.time()
    cfg_kw = dict()
    if cfg_overrides:
        cfg_kw.update(cfg_overrides)

    print(f"[C1] master_seed={master_seed}  R={realizations}  cfg={cfg_kw}")
    # ground truth (cached to disk: depends only on the grid + IC seed)
    import pickle
    cache = os.path.join(os.path.dirname(__file__), "..", "data",
                         f"ground_truth_{master_seed+777}.pkl")
    if os.path.exists(cache):
        print("[C1] loading cached ground-truth bifurcation diagram ...")
        with open(cache, "rb") as f:
            truth = pickle.load(f)
    else:
        print("[C1] computing ground-truth bifurcation diagram ...")
        truth = training.ground_truth_bifurcation(RHO_GRID, n_esn_steps=8000,
                                                  transient_time=80.0, with_lyap=True,
                                                  ic_seed=master_seed + 777)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump(truth, f)
    print(f"[C1]   ground truth ready ({time.time()-t0:.1f}s)")

    # training segments are shared across realizations (data is fixed; only the
    # reservoir draw changes -- methodology 5)
    segments = training.build_segments(RHO_TRAIN, L_TOTAL, transient_time=80.0,
                                       ic_seed=master_seed + 1)

    preds = []
    for k in range(realizations):
        cfg = ESNConfig(seed=master_seed + k, **cfg_kw)
        esn = training.train_realization(cfg, segments)
        pred = training.predicted_bifurcation(esn, RHO_GRID, segments,
                                              n_free=n_free, discard=discard,
                                              seed=master_seed + k)
        preds.append(pred)
        print(f"[C1]   realization {k+1}/{realizations} done ({time.time()-t0:.1f}s)")

    agg = training.aggregate_realizations(preds)
    m = training.c1_metrics(agg, truth,
                            rmse_rho_min=None if rmse_full_grid else training.HOPF_RHO)
    passed, parts = training.c1_pass(m)

    print("\n[C1] ----- acceptance (methodology section 7) -----")
    print(f"  regime-class accuracy : {m['class_acc']*100:5.1f}%  "
          f"(>=95%? {parts['class']})  [{m['n_class_wrong']}/{m['n_rho']} wrong]")
    print(f"  z-maxima RMSE         : {m['zmax_rmse']:.3f}  "
          f"= {m['zmax_rmse_frac']*100:.2f}% of z-range {m['z_range']:.2f}  "
          f"(<=5%? {parts['rmse']})")
    print(f"  lyap/spread agreement : {m['lyap_proxy_err']*100:.1f}%  "
          f"(<=10%? {parts['lyap']})")
    print(f"  C1 {'PASS' if passed else 'FAIL'}   ({time.time()-t0:.1f}s total)\n")

    os.makedirs(outdir, exist_ok=True)
    _plot(agg, truth, m, passed, os.path.join(outdir, f"fig1_{tag}_bifurcation.png"))

    result = {"tag": tag, "master_seed": master_seed, "realizations": realizations,
              "cfg_overrides": cfg_kw, "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                                   for k, v in m.items()},
              "passed": bool(passed), "parts": {k: bool(v) for k, v in parts.items()},
              "rho_train": RHO_TRAIN, "elapsed_s": time.time() - t0}
    with open(os.path.join(outdir, f"{tag}_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result, agg, truth


def _plot(agg, truth, m, passed, path):
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    # true z-maxima scatter
    for rho, zm in zip(truth["rho"], truth["zmax"]):
        if zm.size:
            ax.plot(np.full(zm.size, rho), zm, ".", color="0.55", ms=1.4,
                    alpha=0.5, rasterized=True)
    # predicted z-maxima scatter (pooled across realizations)
    for rho, zm in zip(agg["rho"], agg["zmax"]):
        if zm.size:
            ax.plot(np.full(zm.size, rho), zm, ".", color="#c0392b", ms=1.0,
                    alpha=0.35, rasterized=True)
    ax.axvline(24.74, color="#2c3e50", lw=1.0, ls="--", alpha=0.8)
    # label sits in the clear band below the legend and above the data, read L->R
    ax.text(24.84, 0.80, "Hopf  $\\rho\\approx24.74$",
            transform=ax.get_xaxis_transform(),
            fontsize=8, rotation=0, va="top", ha="left", color="#2c3e50",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.9))
    for rt in truth["rho"][np.isin(truth["rho"], RHO_TRAIN)]:
        ax.axvline(rt, color="#16a085", lw=0.8, ls=":", alpha=0.6)
    ax.plot([], [], ".", color="0.55", ms=6, label="ground truth")
    ax.plot([], [], ".", color="#c0392b", ms=6, label="ESN (cold extrapolation)")
    ax.plot([], [], ":", color="#16a085", label="training $\\rho$")
    ax.set_xlabel("$\\rho$ (Rayleigh parameter)")
    ax.set_ylabel("successive $z$-maxima")
    ax.set_title(f"C1 reproduction of the bifurcation diagram, predicted vs true "
                 f"({'PASS' if passed else 'FAIL'})\n"
                 f"class acc {m['class_acc']*100:.1f}%, "
                 f"z-max RMSE {m['zmax_rmse_frac']*100:.2f}% of range")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[C1] figure -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--realizations", type=int, default=32)
    ap.add_argument("--master-seed", type=int, default=20260613)
    ap.add_argument("--gamma-p", type=float, default=None)
    ap.add_argument("--spectral-radius", type=float, default=None)
    ap.add_argument("--ridge", type=float, default=None)
    ap.add_argument("--N", type=int, default=None)
    ap.add_argument("--n-free", type=int, default=5000)
    ap.add_argument("--tag", type=str, default="c1")
    ap.add_argument("--rmse-full-grid", action="store_true",
                    help="score amplitude RMSE on the full grid (the legacy v1 metric)")
    args = ap.parse_args()

    overrides = {}
    if args.gamma_p is not None: overrides["gamma_p"] = args.gamma_p
    if args.spectral_radius is not None: overrides["spectral_radius"] = args.spectral_radius
    if args.ridge is not None: overrides["ridge"] = args.ridge
    if args.N is not None: overrides["N"] = args.N

    run(realizations=args.realizations, master_seed=args.master_seed,
        cfg_overrides=overrides or None, n_free=args.n_free, tag=args.tag,
        rmse_full_grid=args.rmse_full_grid)
