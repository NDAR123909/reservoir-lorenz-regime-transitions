"""
gate_one.py
===========
Evaluate ONE C1 configuration and append the result to data/gate_log.json.
Driving the section-8 resolution path one config per process keeps each run
inside the time budget (background jobs don't persist here) while building up a
persistent, inspectable record of exactly what moved.

Usage:
    python gate_one.py --stage gamma_p --value 0.7 \
        --base '{"gamma_p":0.7}' --R 6
The --base dict carries hyperparameters already fixed by earlier stages.
"""
import sys, os, json, time, argparse
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
LOG = os.path.join(os.path.dirname(__file__), "..", "data", "gate_log.json")


def load_truth():
    cache = os.path.join(os.path.dirname(__file__), "..", "data",
                         f"ground_truth_{MASTER+777}.pkl")
    with open(cache, "rb") as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--value", required=True)
    ap.add_argument("--base", default="{}", help="JSON of already-fixed overrides")
    ap.add_argument("--R", type=int, default=6)
    ap.add_argument("--n-free", type=int, default=4000)
    args = ap.parse_args()

    base = json.loads(args.base)
    # coerce value to right type
    v = args.value
    val = int(v) if args.stage == "N" else float(v)
    cfg_over = dict(base)
    cfg_over[args.stage] = val

    t0 = time.time()
    truth = load_truth()
    segments = training.build_segments(RHO_TRAIN, L_TOTAL, transient_time=80.0,
                                       ic_seed=MASTER + 1)

    preds = []
    for k in range(args.R):
        cfg = ESNConfig(seed=MASTER + k, **cfg_over)
        esn = training.train_realization(cfg, segments)
        preds.append(training.predicted_bifurcation(
            esn, RHO_GRID, segments, n_free=args.n_free, discard=1200,
            seed=MASTER + k))
    agg = training.aggregate_realizations(preds)
    m = training.c1_metrics(agg, truth)
    passed, parts = training.c1_pass(m)

    row = {"stage": args.stage, "value": val, "cfg": cfg_over,
           "class_acc": float(m["class_acc"]),
           "rmse_frac": float(m["zmax_rmse_frac"]),
           "lyap_err": float(m["lyap_proxy_err"]),
           "n_wrong": int(m["n_class_wrong"]), "passed": bool(passed),
           "R": args.R, "elapsed_s": round(time.time() - t0, 1)}

    log = {"log": [], "base": base}
    if os.path.exists(LOG):
        with open(LOG) as f:
            log = json.load(f)
    log["log"].append(row)
    with open(LOG, "w") as f:
        json.dump(log, f, indent=2)

    print(f"{args.stage}={val!s:<7} class={m['class_acc']*100:5.1f}% "
          f"rmse={m['zmax_rmse_frac']*100:5.2f}% lyap={m['lyap_proxy_err']*100:5.1f}% "
          f"{'PASS' if passed else 'fail'}  ({row['elapsed_s']}s)")


if __name__ == "__main__":
    main()
