# Progress log — Session 8: C5 reproducibility

**Project:** Parameter-aware echo state networks for Lorenz regime transitions
**Author:** Noah Riego, Applied Physics, University of Arizona
**Date:** 15 June 2026
**Scope:** Settle the reproducibility story for the repo, make the README the single source of truth for reproduction, time each of Figures 1–4 from a clean checkout against the §7 bar, and confirm C5 passes. C1 stays locked, the §8 gate stays shut, and the architecture is not reopened. This is the last contribution; after it passes, the next move is the paper.

## Literature snapshot first

The §9 risk register asks for a re-scan at session start, so I did that before anything else. The gap still holds. The two 2026 items worth recording are a January 2026 NG-RC paper in *Chaos* that reconstructs bifurcation diagrams from four training values, and a February 2026 arXiv paper that runs a large ESN hyperparameter sweep on the M4 forecasting set. The first is the same neighbor I logged in Session 7: it notes that error grows once the model leaves the training range, but reports that as a qualitative limit rather than a measured distance against sampling. The second sweeps leakage, spectral radius, and reservoir size for univariate forecasting accuracy, which is a different question on a different substrate and never touches a bifurcation parameter. Neither measures extrapolation distance against sampling range, density, or position at fixed data on the Lorenz system. Verdict: no change. I continued.

## The reproducibility decision

I had two options on the table from the end of Session 7. Option A: regenerate the segments and cells under the deterministic seed so the shipped JSONs and figures match a clean re-run bit for bit. Option B: ship the cell stores as they are and document that a fresh run is statistically equivalent rather than identical. I went with A, because a repo that reproduces bit for bit from a clean checkout is the stronger claim and the one the charter's success criterion is really asking for. Getting there took one extra fix that I did not expect to need.

Before committing to A, I checked whether the pipeline was actually deterministic, because A is only worth doing if a second from-scratch run lands in the same place as the first. It did not, quite. The segment cache rebuilt bit-identically under the md5 seed I put in last session, and the per-cell class labels and in-window VPT matched, but the fine-grained VPT array drifted on the occasional realization. I traced it to the spectral-radius step in `reservoir.py`. The sparse reservoir is rescaled to a target spectral radius of 0.6, and the largest eigenvalue magnitude is read from `scipy.sparse.linalg.eigs`. With no starting vector supplied, ARPACK seeds its own residual from an unseeded RNG, so the converged magnitude wobbles from process to process. At the master seed the spread was about 4e-4, which is small but real, and it is enough to shift the rescale factor and flip a single valid-prediction-time threshold step now and then.

The fix is one deterministic starting vector. I draw `v0` from a generator keyed to `cfg.seed`, kept separate from the main reservoir RNG so the W, Win, and bias draws keep their order, and hand it to `eigs`. After that, the spectral-radius spread across repeated calls is exactly zero, and two fully independent from-scratch runs of the same config produce bit-identical cells, the VPT array included. I want to be clear about what this change is and is not. It does not touch a hyperparameter and it does not touch the architecture. The target spectral radius is still 0.6; only the numerical path to it is pinned. It belongs to the same family as the segment-IC hash fix from Session 7: reproducibility plumbing, not retuning. The §8 gate stays shut.

Regenerating under the fixed pipeline left the findings where Session 7 put them. C1 still passes. C3 still saturates at 0.30 beyond the second sample, with M=2 at 0.80. C4 still collapses at ρ = 24.10 at every window position, with the same 0.64 overshoot and zero across-Hopf depth. C2 reproduced with one borderline shift: the W=2 point estimate moved from 0.80 to 1.20. That value sits inside the bootstrap IQR of [0.80, 1.20] I already reported for W=2 last session, so it is a band-edge step flip on the most fragile point in the sweep rather than a new result. The non-monotone C2 shape and the cleanly climbing absolute ceiling (31.2 up to 35.5) are unchanged. I would rather record that shift honestly than paper over it, and it reinforces what the C2 finding already says: the endpoints W=2 and W=10 are the fragile ones, and W=10 is fragile because its lower edge falls below the Hopf, which is the §2.4 tension I come back to below.

## What is cached and what is regenerated

The split follows the principle that anything cheap and deterministic is rebuilt, and anything that takes real time to recompute is shipped so a reader can get to a figure fast.

Shipped in the repo: all of `src/` and `experiments/`, the 480 sweep cells under `data/C2_cells`, `data/C3_cells`, and `data/C4_cells`, the 10 C1 prediction cells under `data/c1v2_preds`, the four result JSONs and four figures, and the paper directory.

Rebuilt on a clean checkout, and kept out of the repo by `.gitignore`: the C2–C4 ground-truth cache (`data/truth_cache.pkl`), the C1 ground-truth cache (`data/ground_truth_20261390.pkl`), and the per-config training segments under `data/segcache`. These all regenerate from the logged seeds, the truth caches in a few minutes and the segments as a side effect of the first sweep run. I corrected the `.gitignore` this session: the old `data/*.pkl` rule caught the two stray truth caches but not the segment cache, which lives in a subdirectory, so I added it explicitly while leaving the shipped cell directories untouched.

## Verification: each figure from a clean checkout, timed

I reproduced all four figures end to end from a clean state on the pinned environment (Python 3.12.3, numpy 2.4.4, scipy 1.17.1, matplotlib 3.10.8) and timed each path. The §7 bar is that a graduate student reproduces any one figure in under an hour following the README. Every figure clears it with room to spare.

| Figure | Contribution | Fast path (cached cells) | Full path (clean checkout) |
|---|---|---|---|
| 1 | C1 reproduction | shipped PNG / cells, seconds | ~8 min (GT cache + 10 realizations) |
| 2 | C2 range width | finalize + plot, ~3 s | ~24 min (truth cache + 160 cells) |
| 3 | C3 sample density | finalize + plot, ~3 s | ~27 min (truth cache + 192 cells) |
| 4 | C4 window position | finalize + plot, ~3 s | ~22 min (truth cache + 128 cells) |

The fast path rebuilds a figure from the shipped cells with `run_sweep.py --mode finalize` followed by `make_figures.py`, which is a couple of seconds. The full path rebuilds the cells from scratch, which is what proves the bit-for-bit claim. The truth cache is the shared up-front cost for Figures 2–4 at roughly four to five minutes; Figure 1 does not use it and instead builds its own C1 ground-truth cache. The whole pipeline end to end, all four figures from nothing, is a little over an hour, but the criterion is any single figure, and the slowest single figure is under half an hour.

Figure 1 came back a pass under the determinism fix: regime-class accuracy 100% across the full grid, z-maxima RMSE 2.29% of the z-range on the chaotic band above the Hopf, and lyapunov-proxy agreement 6.1%, against the §7 thresholds of 95%, 5%, and 10%. The full-grid RMSE that includes the coexistence sliver is 7.61%, which is the same floor Sessions 5 and 6 reported and the reason the v2 criterion scores amplitude above the Hopf.

## README and layout

The README is now the single source of truth for reproduction. It states the pinned environment, gives the fast single-figure path and the full-pipeline path, lists the seeds and what they drive, and points at the reproducibility statement for the determinism details. I checked the tree against the ENSO layout the charter specifies and it matches: `README.md`, `LICENSE`, `requirements.txt`, `src/` with the five modules, `experiments/` with the three notebooks plus the runner scripts, and the `data/`, `figures/`, and `paper/` directories.

## The C2 W=10 / §2.4 tension, resolved

This was left open from Session 7, and after the reproducibility work was settled I closed it by running the experiment rather than guessing. Section 4's C2 grid puts the W=10 lower edge at ρ = 24, below the Hopf at 24.74, which collides with the §2.4 training exclusion. The question was whether the W=10 jump in the C2 curve, the thing that makes the curve non-monotone, is a real feature or just contamination from that one sub-Hopf sample.

I went in expecting contamination and a recommendation to cap the widths. I was wrong. I ran a clamped variant, the same W=10 window with its lowest training sample pushed up to the Hopf (training set {25, ..., 34} instead of {24, ..., 34}), at the full R=32 against the locked pipeline. It came back at Δρ = 1.50 with an absolute ceiling of 35.50, identical to the original W=10. The cells are genuinely different runs, with median in-window VPT 7.60 against the original 7.67 and per-realization values that differ cell by cell, so this is not a re-label. The extrapolation distance simply does not move when the sub-Hopf sample is removed.

That flips the call. Capping the widths to force a monotone trend would have hidden a real effect. The non-monotonicity is genuine: reach falls from W=2 down to a minimum at W=8, then rises again at W=10, while the absolute ceiling climbs the whole way. I am glad I ran this before drafting the C2 figure caption, because the honest caption is the opposite of the one I would have written from the Session 7 note.

The §2.4 tension is still worth fixing, but as a compliance point, not a result change. The methodology v3 note (`paper/03_methodology_v3_note.md`) records the amendment: the C2 grid is constrained so no window trains below the Hopf, the clamped W=10 is the compliant point, and it gives the same answer. The original W=10 cells stay in the repo as the pre-amendment record, and `figures/c2clamp_result.json` carries the clamped point. C2 is reported as non-monotone, with the clamped check as the evidence that the widest-window upturn is not a training artifact.

## Status

C5 meets its §7 criterion. With C1 through C4 already passed under methodology v2, all five contributions now pass and the project is methodologically complete. The C2 / §2.4 item that was open at the start of the session is closed: the methodology v3 note is written, the clamped robustness check is run and shipped, and C2 stands as a non-monotone result. The remaining work is the paper itself. No more sweeps.
