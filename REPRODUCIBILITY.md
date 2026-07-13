# Reproducibility

This repo is built to reproduce from a clean checkout. On the pinned environment, a fresh run rebuilds the shipped figures bit-for-bit and not just statistically. This file records the seeds, the determinism guarantees, and what is cached versus regenerated.

## Environment

Developed and validated on Python 3.12.3 with numpy 2.4.4, scipy 1.17.1, and matplotlib 3.10.8. The versions are pinned in `requirements.txt` and the interpreter is recorded in `.python-version`. The exact guarantee is specific to this stack. The two version-sensitive pieces are the sparse-matrix RNG that builds the reservoir and the ARPACK eigensolver that reads its spectral radius, so a different numpy or scipy can produce a statistically equivalent run that is not 100% identical.

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Seeds

Everything random in the study traces back to one master seed, `20260613`.

- Reservoir draws: realization `k` of any config uses reservoir seed `master + k`. This sets the sparse recurrent matrix, the input weights, and the bias.
- Spectral-radius solver: the ARPACK starting vector is drawn from a generator keyed to `seed + 777`, separate from the reservoir RNG so the matrix and weight draws keep their order. This is the determinism fix described below.
- Training segments: the per-config initial conditions are seeded from an md5 hash of the config id. This is deterministic across processes, which a salted Python `hash()` is not. It replaced an earlier salted version.
- C2-C4 ground truth: one seeded trajectory per ρ, initial condition from `master + 777`.
- C1 ground truth: same scheme, cached separately as `ground_truth_20261390.pkl`.
- C1 segments: initial conditions from `master + 1`.
- Bootstrap bands: the IQR resampling in the finalize step uses a fixed seed of `12345`, so the reported bands are themselves reproducible.

## Determinism

The pipeline is deterministic end to end on the pinned stack. There was one spot that was not, and the fix is as follows:

`scipy.sparse.linalg.eigs` computes the largest eigenvalue magnitude used to rescale the reservoir to its target spectral radius. Left to its default, ARPACK seeds its starting residual from an unseeded RNG, so the converged magnitude varies slightly between processes. At the master seed the spread was about 4e-4. That is enough to move the rescale factor and occasionally flip a single valid-prediction-time threshold step, which is the difference between a run that reproduces bit for bit and one that only reproduces in distribution. Supplying `eigs` with a seeded `v0` pins the magnitude to a fixed value per seed. With the fix, the spectral-radius spread across repeated calls is zero, and two independent from-scratch runs of the same config produce identical cells.

This change only affects determinism. It does not alter any hyperparameter or the architecture, which means that the target spectral radius is unchanged at 0.6.

To check determinism yourself, run any one sweep config twice into separate trees and compare the cell pickles. They are byte-stable, the VPT arrays included.

## Cached versus regenerated

Shipped in the repo:

- `src/` and `experiments/`
- the 480 sweep cells in `data/C2_cells`, `data/C3_cells`, `data/C4_cells`
- the 10 C1 prediction cells in `data/c1v2_preds`
- the four result JSONs and four figures in `figures/`

Rebuilt on a clean checkout, excluded by `.gitignore`:

- `data/truth_cache.pkl`, the C2-C4 ground-truth and valid-ground-truth-time cache, about four to five minutes to build
- `data/ground_truth_20261390.pkl`, the C1 ground-truth cache
- `data/segcache/`, the per-config training segments, built as a side effect of the first sweep run

The shipped cells let a reader rebuild any figure in seconds with the finalize-and-plot path. Rebuilding the cells from scratch is the slower full path and is what backs the claim regarding exact reproduction.

## Tolerance

On the pinned environment, there is no tolerance to quote, so the figures reproduce exactly. Off the pinned environment, expect the headline findings to hold but individual numbers to move at the level of a single δρ = 0.1 step. The known-fragile points are the C2 endpoints. C2 W=2 in particular has a bootstrap IQR of [0.80, 1.20], and a fresh run on a different stack can land anywhere in that band. The saturation in C3, the collapse position in C4, and the C1 pass are all robust to the seed and the stack.
