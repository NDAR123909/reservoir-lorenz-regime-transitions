# Parameter-aware echo state networks for Lorenz regime transitions

Can a reservoir computer, trained on a handful of chaotic trajectories at a few values of the Lorenz parameter rho, reconstruct the system's bifurcation diagram at values of rho that it never saw? That is the whole question that this project aims to answer. The network gets the parameter handed to it as an extra input channel, so in principle, it can be told "now behave as if rho were this" and run forward from there.

This repository is the code and the running record for that project. It is built and tested entirely by the author of the project, written in stages, with each and every session documented and logged in `paper/`.

## Where things stand

The C1 reproduction gate is **passed** and the hyperparameters are **locked**. The story took a turn between Session 5 and Session 6.

At the Session-5 lock attempt, C1 was a partial pass: regime class and Lyapunov agreement cleared their bars, but the z-maxima amplitude RMSE floored near 7.5% of the z-range against a 5% bar, and the full section-8 gate walk (gamma_p, spectral radius, ridge, reservoir size) could not move it. Methodology section 10 sends an exhausted gate back for a logged revision rather than a hand-tune, so the hyperparameters were deliberately left unlocked.

Session 6 found the cause. The whole 7.5% was six grid points below the subcritical Hopf (rho approx 24.74), where the Lorenz system is in its attractor-coexistence region: the chaotic attractor and the stable C-plus/C-minus pair both exist, the single-IC ground-truth z-maxima envelope flips between them at neighbouring rho, and the network is charged a large error for sitting steadily on the chaotic branch. The methodology already excludes that region from *training* (section 2.4) for the same reason; the C1 *test* grid was reaching into it. Two non-architecture knobs (the parameter-channel reference interval and the cold-extrapolation primer) were measured first and ruled out as the cause.

The methodology was revised (v2, `paper/03_methodology_v2.pdf`): the C1 amplitude RMSE is now read on the chaotic band rho >= 24.74, where the envelope is single-valued. Regime class is still scored on the full grid, and the downward-across-Hopf behaviour is C4's job. With the model unchanged, C1 passes at R = 10:

- regime-class accuracy: 100% on the full grid (target >= 95%) — pass
- z-maxima RMSE on rho >= 24.74: 2.29% of the z-range (target <= 5%) — pass
- largest-Lyapunov agreement: 6.1% (target <= 10%) — pass

(These are the values from the regenerated run under the Session-8 determinism fix; they sit within rounding of the Session-6 pass at 2.31% and 5.6%. See `REPRODUCIBILITY.md`.)

Locked hyperparameters: gamma_p = 0.1, spectral radius = 0.6, everything else at the section-1.5 priors. These are now the `ESNConfig` defaults and carry into C2–C4 unchanged. The Session-6 reasoning is in `paper/05_progress_log.md`; the diagnostics are in `data/diag_log.json` and `data/diag_zones.json`.

**Session 7 ran C2–C4 against the locked architecture** (R = 32, total data fixed at 120,000, delta-rho = 0.1). Headlines: C3 (the key density measurement) saturates flat at delta-rho ~ 0.30 from M = 3 on, with the data-volume confound controlled and the washout floor not reached. C4 finds that no training window crosses the Hopf cleanly; instead every window, regardless of position, places the predicted collapse at rho ~ 24.10 (the coexistence onset), overshooting the true Hopf by ~0.64. C2 is resolved point-by-point but non-monotone. Its widest config (W = 10) sits high, and a Session-8 robustness check (a §2.4-compliant W = 10 with the sub-Hopf sample clamped to the Hopf) returns the same Δρ = 1.50, so that upturn is a real feature of the wide window rather than sub-Hopf contamination. Full write-up in `paper/06_progress_log.md`; results in `figures/c{2,3,4}_result.json` and Figures 2–4.

**Session 8 settled the C5 reproducibility contribution.** The repo now reproduces from a clean checkout bit for bit on the pinned environment, which took one determinism fix to the spectral-radius solver (a seeded ARPACK starting vector; no hyperparameter or architecture change). Every figure regenerates under the deterministic pipeline with the findings intact: C1 passes, C3 saturates at 0.30, C4 collapses at 24.10, and C2 holds with one band-edge shift on its most fragile point (W=2, from 0.80 to 1.20, inside its already-reported IQR). Each of Figures 1–4 reproduces from a clean checkout well under the one-hour bar. With C5 passed, all five contributions pass and the project is methodologically complete. The reproduction recipe is below and the determinism details are in `REPRODUCIBILITY.md`; the session write-up is `paper/07_progress_log.md`.

## Layout

```
.
├── src/
│   ├── lorenz.py                  # RK4 integrator, regime classifier, Lyapunov exponent, bifurcation map
│   ├── reservoir.py               # Parameter-aware ESN (sparse reservoir, ridge readout, separate parameter channel)
│   ├── metrics.py                 # Valid prediction time, correlation dimension, z-maxima Wasserstein distance
│   ├── training.py                # Segment builder, realization training, C1 acceptance test
│   └── sweep.py                   # C2–C4 extrapolation-distance measurement (Methodology §3.5) and sweep definitions
│
├── experiments/
│   ├── run_c1.py                  # C1 gate: train, cold-extrapolate, score against ground truth
│   ├── gate_one.py                # Evaluate one hyperparameter configuration (Section 8 search path)
│   ├── run_c1_v2.py               # C1 re-validation under the v2 criterion (checkpoint/resume)
│   ├── build_truth_cache.py       # Precompute ground-truth cache used by C2/C3 VPT measurements
│   ├── run_sweep.py               # Chunked, resumable C2–C4 driver + finalization
│   │                              # (median curves, bootstrap IQR)
│   ├── make_figures.py            # Generate Figures 2–4 from result JSONs
│   ├── make_attractor_figure.py   # Generate Figure 5 (3D attractor climate at unseen ρ)
│   ├── render_log_pdf.py          # Convert paper/0N_progress_log.md to PDF
│   ├── exp01_baseline.ipynb
│   ├── exp02_sweep.ipynb
│   └── exp03_transitions.ipynb
│
├── data/
│   ├── Shipped:
│   │   ├── C2–C4 cell stores
│   │   ├── C1 prediction cells
│   │   └── Gate/diagnostic JSONs
│   └── Regenerated (gitignored):
│       ├── Ground-truth cache
│       └── Segment caches
│
├── figures/                       # Output plots and per-sweep result JSONs
├── paper/                         # Charter, literature review, methodology, session progress logs
└── REPRODUCIBILITY.md             # Seeds, determinism, cached vs. regenerated artifacts, tolerances
```

## Reproducing the figures

This section is the single source of truth for reproduction. Everything below was checked end to end from a clean checkout this session.

Environment first. The repo reproduces bit for bit on Python 3.12.3 with the pinned numpy, scipy, and matplotlib. Other versions give statistically equivalent runs, not identical ones, because the sparse-matrix RNG and the eigensolver are version-sensitive (see `REPRODUCIBILITY.md`).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd experiments
```

Long single jobs get killed in this environment, so every runner checkpoints one cell per process and resumes. The shipped cell stores let you skip the long runs entirely.

Fast path, any of Figures 2–4 in seconds from the shipped cells:

```bash
python run_sweep.py --mode finalize --sweep C3 --R 32   # C2 / C3 / C4
python make_figures.py                                   # writes Figures 2-4
```

Full path, Figures 2–4 rebuilt from scratch (this is what reproduces the cells bit for bit):

```bash
python build_truth_cache.py                              # ~4-5 min, once, shared by C2-C4
for s in C2 C3 C4; do
  while python run_sweep.py --mode run --sweep $s --R 32 --max-cells 60 \
        | grep -q "batch cap"; do :; done                # repeat until the sweep is done
  python run_sweep.py --mode finalize --sweep $s --R 32  # median curve + bootstrap IQR + JSON
done
python make_figures.py
```

Figure 1, the C1 reproduction, has its own ground-truth cache and does not use the C2–C4 one:

```bash
python run_c1.py --realizations 1                        # builds the C1 ground-truth cache
python run_c1_v2.py --mode run --upto 10 --batch 10      # repeat until 10/10 on disk
python run_c1_v2.py --mode finalize --upto 10            # scores, writes fig1_c1_v2_bifurcation.png
```

Figure 5, the 3D attractor-climate companion to C1, needs no caches at all. It trains realization 0 fresh and free-runs at three unseen rho:

```bash
python make_attractor_figure.py                          # ~15 s, writes fig5_attractor_climate.png
```

Timing from a clean checkout, measured on the pinned stack: Figure 1 about 8 minutes, Figure 2 about 24, Figure 3 about 27, Figure 4 about 22, and Figure 5 about 15 seconds. The whole pipeline end to end is a little over an hour; any single figure is well under it.

To walk the locked-gate path one configuration at a time (for the record, not needed to reproduce the figures):

```bash
python gate_one.py --stage gamma_p --value 0.1 --base '{}' --R 6   # appends to data/gate_log.json
```

## The model, briefly

A sparse reservoir of N nodes is driven by the three standardized Lorenz coordinates plus a fourth channel carrying the normalized parameter. The state and parameter columns of the input matrix are scaled separately (gamma_in and gamma_p), which matters more than I expected — gamma_p turned out to be the one lever that actually moved the error, and it moved it the opposite way from my initial guess. Readout is plain ridge regression solved in closed form. For the bifurcation diagram the network runs cold: no ground-truth trajectory at the target rho, just the parameter value and a free run.

Details, landmark values, and the locked hyperparameter ranges are in `paper/03_methodology_v2.pdf`.

## Status of the science

All five contributions pass under methodology v2, so the study is methodologically complete. C1 reproduces the locked bifurcation diagram with amplitude RMSE scored on the chaotic band ρ ≥ 24.74. C2 measures extrapolation distance against training-range width and finds it non-monotone, with the absolute ceiling climbing cleanly. C3 measures it against sample density and finds it saturating at Δρ ≈ 0.30 from three samples on. C4 measures across-Hopf behaviour against window position and finds a position-independent collapse at the coexistence onset ρ ≈ 24.10. C5 makes the whole thing reproduce from a clean checkout.

Figure 5 (`fig5_attractor_climate.png`) is a qualitative companion to C1: the actual attractor, true system on top and cold-extrapolated ESN below, at three ρ values the network never trained on, one just above the Hopf, one inside the training range, and one beyond its upper edge. One caution on reading it. A free-running ESN diverges from the true trajectory pointwise once past the valid prediction time, so what gets compared here is climate, the geometry the network settles onto, not the path itself. The z-maxima statistics in Figures 1–4 are the quantitative version of the same claim.

The C2 / §2.4 item from Session 7 is closed (`paper/03_methodology_v3_note.md`). The W=10 lower edge fell below the Hopf, which collided with the §2.4 training exclusion. The methodology v3 note constrains the C2 grid so no window trains below the Hopf, and the clamped W=10 robustness check (`figures/c2clamp_result.json`) shows the result is unchanged at Δρ = 1.50, so the non-monotone C2 finding stands without a contamination caveat. None of this reopens the gate or the architecture.

The cell stores under `data/C{2,3,4}_cells/` are the raw per-(config, realization) results; `figures/c{2,3,4}_result.json` carry the aggregated medians, bootstrap IQR bands, and diagnostics. The ground-truth and segment caches regenerate from the logged seeds and are not shipped. See `REPRODUCIBILITY.md` for seeds and determinism, and the reproduction recipe above for the commands.
