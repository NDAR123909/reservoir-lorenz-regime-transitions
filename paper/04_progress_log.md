# Progress log — Session 5: implementation and the C1 gate

**Project:** Parameter-aware echo state networks for Lorenz regime transitions
**Author:** Noah Riego, Applied Physics, University of Arizona
**Date:** 14 June 2026
**Scope of this session:** build the core code, run the C1 reproduction gate, and walk the section-8 resolution path if C1 fails. The C2–C4 sweeps are explicitly out of scope and belong to the next session.

## What I built

Four modules, plus two experiment drivers, all under `src/` and `experiments/`:

- `lorenz.py` — RK4 integrator at h = 0.01 with the ESN observing every second step (dt = 0.02), plus the bifurcation-map and regime-classifier from methodology section 2.3, and a Benettin tangent-space estimate of the largest Lyapunov exponent.
- `reservoir.py` — the parameter-aware ESN of section 1. Sparse reservoir (N = 500, degree 6, spectral radius 0.9), leak 1.0, a four-column input map whose state columns are scaled by gamma_in and whose parameter column is scaled separately by gamma_p, ridge readout solved in closed form, washout 1000. The parameter channel maps rho onto the reference interval [20, 36].
- `metrics.py` — valid ground-truth time, valid prediction time (capped at the ground-truth time, eps = 0.4), the qualitative class test, Grassberger–Procaccia correlation dimension, and the Wasserstein distance between z-maxima distributions.
- `training.py` — segment builder, per-realization training, ground-truth and predicted bifurcation diagrams, realization aggregation, and the C1 metric/acceptance functions.
- `run_c1.py` and `gate_one.py` — the gate runner and a single-config evaluator I leaned on heavily once the long background sweep kept getting killed (more on that below).

Two checks told me the physics core was sound before I touched the reservoir. The Lyapunov estimate came out at 0.910 at rho = 28, against the textbook 0.906. The classifier put everything below rho ≈ 24 at a fixed point and everything above into chaos, with the landmark crossings where methodology section 2.3 says they should be. Good enough to trust the ground truth.

## A note on how the gate actually got run

The plan was one background sweep walking all four levers. That did not survive the session — background jobs here get killed between steps, so the sweep printed its first stage header and died. I rewrote it as `gate_one.py`, which evaluates a single configuration at R = 6 and appends the result to `data/gate_log.json`. One config per process, each landing around 95–135 seconds, with the expensive ground-truth bifurcation diagram cached to disk so it only had to be computed once. Slower to babysit, but every row is on disk and nothing gets lost when a process dies. The full log is in `data/gate_log.json`.

## C1 at the locked hyperparameters: a fail, and a wrong guess about why

At the section-1.5 defaults, C1 misses two of its three criteria:

| criterion | result | threshold | verdict |
|---|---|---|---|
| regime-class accuracy | 100.0% | ≥ 95% | pass |
| z-maxima RMSE (frac. of z-range) | 10.98% | ≤ 5% | fail |
| Lyapunov / spread agreement | 13.6% | ≤ 10% | fail |

The error sits at the extrapolation edges: the network over-predicts the small near-Hopf attractor just below the lowest training point and under-predicts the amplitude well above the highest. Going in, I had written down that this looked like gamma_p being too low, the parameter channel too quiet to push the reservoir into the right basin. That guess was wrong, and the gate is what showed me.

## Walking the section-8 path

The methodology fixes the order: gamma_p first, then spectral radius, then ridge, then N. Fix the best value found at each lever, move on only if C1 still fails.

### Lever 1 — gamma_p, range [0.1, 1.0]

| gamma_p | class | rmse | lyap |
|---|---|---|---|
| 0.1 | 100.0% | 7.59% | 7.1% |
| 0.3 | 100.0% | 8.70% | 9.3% |
| 0.5 (default) | 100.0% | 10.98% | 13.6% |
| 0.7 | 100.0% | 13.42% | 33.9% |
| 1.0 | 87.7% | 18.24% | 89.5% |

Cleanly monotonic, and pointing the opposite way from my prediction. Lowering gamma_p helped; raising it hurt fast. At 1.0 the parameter channel swamps the state, regime accuracy falls off the table (87.7%), and the amplitude error nearly doubles. That is the "too high drowns the state" failure mode of section 1.2, not the "too low can't extrapolate" one I expected. Best within range is the floor, 0.1, which gets rmse to 7.59% and slides the Lyapunov agreement under its threshold. Still short on rmse, so I fixed gamma_p = 0.1 and kept going.

### Lever 2 — spectral radius, range [0.4, 1.2], with gamma_p = 0.1

| spectral radius | class | rmse | lyap |
|---|---|---|---|
| 0.4 | 100.0% | 7.46% | 6.4% |
| 0.6 | 100.0% | 7.47% | 5.5% |
| 0.8 | 100.0% | 7.52% | 5.6% |
| 1.0 | 100.0% | 7.82% | 7.2% |
| 1.2 | 100.0% | 16.23% | 43.2% |

Below 0.8 the rmse flattens into a plateau around 7.5% and stops caring what the spectral radius is. Above 1.0 it falls apart (1.2 sends rmse to 16% and the Lyapunov agreement to 43%). The whole [0.4, 0.8] stretch is a wash within R = 6 noise, so I picked 0.6, where the Lyapunov agreement is best at 5.5%. It does not break the rmse barrier and I did not expect it to.

### Lever 3 — ridge, range [1e-8, 1e-4]

| ridge | class | rmse | lyap |
|---|---|---|---|
| 1e-8 | 100.0% | 7.58% | 6.8% |
| 1e-7 | 100.0% | 7.46% | 6.5% |
| 1e-6 (default) | 100.0% | ~7.47% | ~5.5% |
| 1e-5 | 100.0% | 7.50% | 6.8% |
| 1e-4 | 100.0% | 7.48% | 6.0% |

Four orders of magnitude in the regularization, and the rmse does not move off 7.5%. That fits what I already knew: the one-step readout error is around 1e-5, so the readout is not where the amplitude error comes from. No reason to disturb the default, so ridge stayed at 1e-6.

### Lever 4 — N, up to 1000

| N | class | rmse | lyap |
|---|---|---|---|
| 500 (default) | 100.0% | 7.47% | 5.5% |
| 700 | 100.0% | 7.45% | 6.0% |
| 1000 | 100.0% | 7.43% | 6.4% |

Doubling the reservoir buys 0.04 of a percentage point. If the floor were a capacity problem, a 2x reservoir should have dented it, and it did not. So the limit is not how many neurons there are. I kept N = 500 rather than pay 4x the compute for nothing.

## Where C1 actually lands

Best configuration: gamma_p = 0.1, spectral radius = 0.6, ridge = 1e-6, N = 500. I re-ran it at R = 10 with a longer free-run to make sure the floor was not a six-realization fluke:

| criterion | result | threshold | verdict |
|---|---|---|---|
| regime-class accuracy | 100.0% (0/81 wrong) | ≥ 95% | pass |
| z-maxima RMSE | 7.49% of z-range | ≤ 5% | fail |
| Lyapunov / spread agreement | 6.3% | ≤ 10% | pass |

Two of three, solidly. The R = 10 rmse of 7.49% matches the R = 6 sweep value, so this is a real floor and not sampling noise.

![C1 reproduction: predicted z-maxima (red) against the true bifurcation diagram (grey) over rho in [24, 32]. Dotted lines mark the four training values; the dashed line is the subcritical Hopf onset at rho approx 24.74. The fit is tight through the interior and frays at the extrapolation edges.](../figures/fig1_c1_locked_bifurcation.png)

The figure above is also saved at `figures/fig1_c1_locked_bifurcation.png`: the predicted z-maxima track the true bifurcation diagram across the chaotic band, the topology and onset are right, and the visible gap is amplitude error bunched at the two extrapolation edges, rho below 24.56 and rho above 29.06. The interior, where the model interpolates between training points, is tight.

## What this means, and what I am not doing about it

I walked all four gate levers and the z-maxima RMSE would not go below about 7.4% of the z-range anywhere in the locked ranges. It is roughly 1.5x the 5% bar. The other two criteria pass comfortably and stay passed across most of the parameter space.

So C1 is a partial pass: the network reproduces the *structure* of the bifurcation — regime class, onset, Lyapunov exponent — but not the *amplitude* of the z-maxima at the far edges of the parameter sweep, and no setting inside the gate fixes that. Methodology section 10 is explicit about what happens here: an exhausted gate means the methodology gets revised and re-issued, with the revision logged, rather than hand-tuned around. I am not going to redesign the locked computational plan in an implementation session to force a pass. The honest result is more useful to the section-10 process than a massaged one.

A couple of things I would put in front of that revision, as starting points rather than conclusions:

- The error is amplitude, not classification, and it lives entirely in extrapolation. That points at the parameter channel's encoding or the training window placement, not at reservoir size or regularization, both of which I have now ruled out.
- The 5% z-maxima target may be the wrong yardstick for cold extrapolation specifically. Worth asking whether the acceptance criterion should separate interpolation from extrapolation, since the interior already clears it.

## Status

- Core code built and the physics core validated.
- C1 run, gate walked in full, every configuration logged to `data/gate_log.json`.
- Hyperparameters not locked, because C1 did not fully pass. Best-found configuration recorded above and flagged for the section-10 revision.
- Stopping here. The C2–C4 sweeps are next session, as planned.
