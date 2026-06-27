"""
gate_sweep.py
=============
Walk the C1 gate resolution path of methodology section 8, in order:
    1. gamma_p          in [0.1, 1.0]
    2. spectral_radius  in [0.4, 1.2]
    3. ridge            in [1e-8, 1e-4]
    4. N                up to 1000
Each stage fixes the best value found so far and moves to the next lever only if
C1 still fails. Records exactly what moved for the progress log.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pickle
import lorenz
import training
from reservoir import ESNConfig

RHO_TRAIN = [24.56, 26.06, 27.56, 29.06]
RHO_GRID = np.round(np.arange(24.0, 32.0 + 1e-9, 0.1), 2)
L_TOTAL = 120_000
MASTER = 20260613


def load_truth():
    cache = os.path.join(os.path.dirname(__file__), "..", "data",
                         f"ground_truth_{MASTER+777}.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    truth = training.ground_truth_bifurcation(RHO_GRID, n_esn_steps=8000,
                                              transient_time=80.0, with_lyap=True,
                                              ic_seed=MASTER + 777)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(truth, f)
    return truth


def evaluate(cfg_over, truth, segments, R=6, n_free=4000, discard=1200):
    preds = []
    for k in range(R):
        cfg = ESNConfig(seed=MASTER + k, **cfg_over)
        esn = training.train_realization(cfg, segments)
        preds.append(training.predicted_bifurcation(
            esn, RHO_GRID, segments, n_free=n_free, discard=discard, seed=MASTER + k))
    agg = training.aggregate_realizations(preds)
    m = training.c1_metrics(agg, truth)
    passed, parts = training.c1_pass(m)
    return m, passed, parts, agg


def main():
    t0 = time.time()
    truth = load_truth()
    segments = training.build_segments(RHO_TRAIN, L_TOTAL, transient_time=80.0,
                                       ic_seed=MASTER + 1)
    print(f"setup ready ({time.time()-t0:.1f}s)\n")

    log = []
    base = {}

    stages = [
        ("gamma_p", [0.1, 0.3, 0.5, 0.7, 1.0]),
        ("spectral_radius", [0.4, 0.6, 0.8, 1.0, 1.2]),
        ("ridge", [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]),
        ("N", [500, 700, 1000]),
    ]

    for stage, values in stages:
        print(f"=== stage: {stage} ===")
        best = None
        for v in values:
            cfg_over = dict(base); cfg_over[stage] = v
            m, passed, parts, agg = evaluate(cfg_over, truth, segments)
            row = {"stage": stage, stage: v, "cfg": dict(cfg_over),
                   "class_acc": m["class_acc"], "rmse_frac": m["zmax_rmse_frac"],
                   "lyap_err": float(m["lyap_proxy_err"]), "passed": passed,
                   "n_wrong": m["n_class_wrong"]}
            log.append(row)
            print(f"  {stage}={v!s:<7} class={m['class_acc']*100:5.1f}% "
                  f"rmse={m['zmax_rmse_frac']*100:5.2f}% lyap={m['lyap_proxy_err']*100:5.1f}% "
                  f"{'PASS' if passed else 'fail'} ({time.time()-t0:.0f}s)")
            # rank: passing first, then lowest rmse_frac, then highest class_acc
            key = (not passed, m["zmax_rmse_frac"], -m["class_acc"])
            if best is None or key < best[0]:
                best = (key, v, m, passed)
        base[stage] = best[1]
        print(f"  -> fixed {stage} = {best[1]}  (passed={best[3]})\n")
        with open(os.path.join(os.path.dirname(__file__), "..", "data", "gate_log.json"), "w") as f:
            json.dump({"log": log, "base": base}, f, indent=2, default=float)
        if best[3]:
            print(f"C1 PASSES with {base}. Stopping resolution path.")
            break

    print(f"\nfinal base config: {base}  ({time.time()-t0:.0f}s)")
    return log, base


if __name__ == "__main__":
    main()
