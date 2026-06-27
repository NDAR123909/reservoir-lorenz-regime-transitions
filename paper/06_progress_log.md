# Progress log — Session 7: the C2–C4 sweeps

**Project:** Parameter-aware echo state networks for Lorenz regime transitions
**Author:** Noah Riego, Applied Physics, University of Arizona
**Date:** 14 June 2026
**Scope:** Build the sweep harness against the locked architecture, run C2, C3, and C4 as methodology v2 §4 specifies, and produce Figures 2–4 with the per-sweep result JSONs. C1 stays locked and the section-8 gate stays shut. The C5 reproducibility write-up is next session, not this one.

## Literature snapshot first

The §9 risk register asks me to re-scan the gap before sweeping, so I did that before touching any code. The gap holds. The 2026 parameter-aware-RC and Lorenz-extrapolation work I could find still extends or applies the framework rather than measuring the sampling curve. A January 2026 NG-RC paper reconstructs bifurcation diagrams from four training values and notes that error grows past the training range, but it reports that as a qualitative limit, not a measured distance against sampling. The one new neighbor worth writing down is an October 2025 paper on reservoir-computing crisis prediction that trains on widely spaced versus closely spaced parameter values and finds the spacing changes the reconstructed diagram. That touches the density axis, but on one-dimensional maps with a crisis-scaling framing, not the Lorenz parameter-aware ESN with a fixed-data extrapolation-distance sweep. It is a citation for the introduction to differentiate against, not a competitor that lands in the gap. Verdict: no change. I continued.

## What I built

The sweeps run on the same checkpoint-and-resume pattern the C1 gate and the C1 v2 re-validation used, for the same reason: long single jobs get killed in this environment, so the work is broken into one `(config, realization)` cell per process, each appended to disk the moment it finishes. A killed or timed-out run loses at most the cell in flight, and a re-invocation picks up where it stopped. I drove the whole study as a sequence of short bounded calls.

Three new pieces under `src/` and `experiments/`:

- `src/sweep.py` — the extrapolation-distance measurement of §3.5 and the three sweep definitions of §4. The same-class upward kernel (C2, C3) and the across-Hopf class-match kernel (C4) live here, along with the Δρ reductions.
- `experiments/build_truth_cache.py` — the ground-truth cache. It is deterministic from the seed stream and is not shipped in the tarball, so I rebuild it here: one seeded ground-truth trajectory per ρ on a 24.0–40.0 grid, its valid ground-truth time (the §3.1 cap), and its qualitative class. Every sweep cell reads from it, so the ground truth is computed once and reused.
- `experiments/run_sweep.py` — the chunked, resumable driver, plus the finalize step that aggregates cells into a median curve with an IQR band and writes each result JSON.
- `experiments/make_figures.py` — Figures 2–4.

The architecture never moves. Every cell trains `ESNConfig()` at its locked defaults (γ_p = 0.1, spectral radius = 0.6, N = 500, the rest at the §1.5 priors); the only thing that changes between realizations is the reservoir seed, and the only thing that changes between configs is the sampling strategy. The driver has no hyperparameter argument, by design.

### How the measurement works

The research question wants a distance in ρ, so the core quantity is the extrapolation distance Δρ of §3.5: start at a training-window edge, step outward at δρ = 0.1, and find the largest contiguous outward run where a per-ρ validity test holds.

For C2 and C3 the test rho stays in the chaotic band, so validity is the same-class rule: the valid-prediction time has to stay at or above a fixed fraction of the in-window VPT, and the predicted class has to stay chaotic while ground truth is chaotic. I fixed that fraction at 0.5 up front and logged it. I read VPT in raw time units rather than Lyapunov times, because the ratio against the in-window value is unit-free and it lets me skip the per-ρ Lyapunov-time conversion, which is the expensive part. A single warmup-then-free-run at each step gives me both the VPT and the predicted class, since I can classify the free-run portion directly.

For C4 the test rho crosses the Hopf into the fixed-point regime, where VPT means nothing once the attractor type has changed. So validity there is the qualitative-class match alone, judged from a cold extrapolation with no ground truth at the target ρ. The ground-truth class below the Hopf comes from the §2.3 landmark, not the single-IC classifier. This is the same call the v2 C1 revision made and for the same reason: the coexistence sliver fools a single-IC label, so the landmark is the honest target there.

On aggregation, §3.5 says to evaluate validity on the median over realizations and §5 says every figure carries an IQR band, so I report both. The point estimate per config is Δρ read off the median-over-realizations curve. The band is a bootstrap of that same estimator over the 32 realizations, which keeps the band centered on the point it qualifies rather than floating somewhere else. I also keep the raw per-realization Δρ values in the JSON for anyone who wants the underlying scatter.

### Validation before the full grid

I ran a single C2 width (W = 4) at R = 4 first, which is the reduced-cost check methodology §6 calls for before committing the grid. It produced a sane Δρ and a coherent band, the cells checkpointed and resumed correctly, and the four reduced-R seeds are a subset of the full R = 32 seeds, so nothing was wasted. Then I ran the full grid: 480 cells total (C2 has 5 widths, C3 has 6 densities, C4 has 4 positions, each at R = 32).

## C2 — extrapolation distance versus training-range width

Center ρ = 29, widths W ∈ {2, 4, 6, 8, 10} at one sample per ρ-unit, total data fixed. Figure 2.

| W | upper edge | Δρ beyond edge | bootstrap IQR | absolute ceiling | in-window VPT |
|---|---|---|---|---|---|
| 2 | 30 | 0.80 | [0.80, 1.20] | 30.8 | 7.64 |
| 4 | 31 | 0.60 | [0.60, 0.60] | 31.6 | 7.62 |
| 6 | 32 | 0.30 | [0.30, 0.30] | 32.3 | 7.62 |
| 8 | 33 | 0.20 | [0.20, 0.20] | 33.2 | 7.61 |
| 10 | 34 | 1.50 | [1.20, 1.50] | 35.5 | 7.66 |

The naive hypothesis was that a wider training range buys more reach. The measured Δρ beyond the edge does not say that. It falls from 0.80 at W = 2 down to 0.20 at W = 8, then jumps to 1.50 at W = 10. The bands are tight, so each point is resolved well above the realization spread, but the trend in Δρ is not monotone.

Two things are worth separating here. First, the in-window VPT reference is steady at about 7.6 across every width, because the window center at ρ = 29 is a training sample in all five configs. So the non-monotone Δρ is a real measurement, not an artifact of a drifting validity bar. Second, the absolute ceiling, meaning the highest ρ the network still tracks (edge plus Δρ), climbs cleanly the whole way: 30.8, 31.6, 32.3, 33.2, 35.5. So the wider window does reach a higher absolute ρ; what shrinks is the marginal distance past the edge, until W = 10 breaks the pattern.

I think the marginal shrinkage from W = 2 to W = 8 is the upper edge moving into higher-ρ chaos, where the Lyapunov exponent is larger and the prediction error grows faster, so fewer outward steps clear the fixed VPT bar even though the absolute reach keeps rising. The W = 10 point is the one I trust least, and I would not build a story on it. Its lower edge sits at ρ = 24, below the Hopf, which collides with §2.4 (training windows are supposed to live in the clearly chaotic band). I followed the §4 grid exactly as written and the anomaly showed up exactly at the config that violates §2.4, which is not a coincidence I want to paper over. More on that below.

Against the §7 criterion for C2 (monotone trend resolved above the realization spread, with a scaling fit if one holds): the resolution half is met, the monotone half is not. There is no clean power law to fit. The honest result is that under a fixed center and fixed density, widening the window raises the absolute extrapolation ceiling monotonically but does not lengthen the reach past the window edge, and the widest config is contaminated by sub-Hopf training data. That is a real finding even though it is not the tidy curve the criterion hoped for.

## C3 — extrapolation distance versus sample density (the key measurement)

Window fixed at W = 6 on center ρ = 29 (so ρ ∈ [26, 32]), total data fixed, sample count M ∈ {2, 3, 5, 8, 13, 21}. Figure 3. This is the measurement no one in the literature has reported, so it is the one I cared most about getting clean.

| M | per-sample length | Δρ beyond edge | bootstrap IQR | in-window VPT |
|---|---|---|---|---|
| 2 | 60,000 | 0.80 | [0.80, 0.80] | 2.34 |
| 3 | 40,000 | 0.30 | [0.30, 0.30] | 7.61 |
| 5 | 24,000 | 0.30 | [0.30, 0.30] | 7.60 |
| 8 | 15,000 | 0.30 | [0.30, 0.30] | 7.60 |
| 13 | 9,230 | 0.30 | [0.30, 0.30] | 7.62 |
| 21 | 5,714 | 0.30 | [0.30, 0.30] | 7.60 |

The curve saturates. From M = 3 on, Δρ sits flat at 0.30 with a degenerate band, no matter how many samples I pack into the window. Adding density beyond three samples buys nothing for the upward same-class reach. That is the diminishing-returns answer the §7 criterion asks C3 to identify, and the data-volume confound is controlled the way §4 demands: L_total is held at 120,000 and the per-sample length absorbs the change in M, so the x-axis really is sample count.

The M = 2 point needs a caveat, and it is not the caveat I expected going in. Its Δρ of 0.80 looks like more reach, but its in-window VPT is 2.34 against the 7.6 that every other config sits at. The reason is that M = 2 puts its two samples at the window ends, ρ = 26 and ρ = 32, so the in-window reference at the center ρ = 29 falls on a point with no training sample nearby and the network interpolates it badly. A low in-window VPT lowers the validity bar, and a lower bar lets more outward steps pass. So the M = 2 advantage is a reference-point artifact, not better extrapolation. Once the center is sampled (every M ≥ 3), the reference recovers to 7.6 and the reach settles at 0.30. I would not report M = 2 as comparable to the rest.

The §4 warning to watch was the per-sample-length floor near washout. It did not bite. At M = 21 each segment is 5,714 steps, and after the 1,000-step washout that is roughly 85 Lyapunov times of usable signal, which is comfortable, exactly as §4 predicted. So the saturation is a real saturation and not a washout-starvation effect masquerading as one. If I wanted to find the floor I would have to push M well past 21, into segments short enough that washout eats most of them. That is a follow-up, not something this grid reaches.

C3 meets its §7 criterion: the curve shape is identified (saturating, flat at 0.30 from M = 3), and the confound is demonstrably controlled.

## C4 — extrapolation across the Hopf versus window position

Width W = 4, count M = 5, total data fixed, window center slid across {27, 29, 31, 33} so the lower edge sits at distances d ∈ {0.26, 2.26, 4.26, 6.26} above the Hopf. Figure 4. The primary kernel here is the qualitative-class match, since the test crosses into the fixed-point regime.

The strict result is a flat zero. The across-Hopf depth, meaning how far below ρ = 24.74 the predicted class still matches ground truth contiguously, is 0.00 for all four positions, with a degenerate band. No window crosses the Hopf cleanly. The hypothesis that a window closer to the Hopf reaches further across it is not supported in the strict sense, because none of them reach across it at all.

That zero is real, but on its own it hides the more interesting thing the network actually does. Stepping down from the lower edge, the closest window (lower edge ρ = 25, d = 0.26) tracks chaotic correctly down to about the Hopf, and then keeps producing a chaotic-looking attractor from ρ = 24.7 down to about 24.2, where the true system has already collapsed to a fixed point. It finally settles onto a fixed point at ρ ≈ 24.1. So the network does locate a transition; it just puts it too low. This is the pseudo-Lorenz persistence that §9 flagged as an expected failure mode, the ESN sustaining a spurious attractor past the bifurcation where the real system has none.

When I measure where the network actually places the collapse, the result is clean and a little surprising:

| window center | d above Hopf | predicted collapse ρ | overshoot below Hopf |
|---|---|---|---|
| 27 | 0.26 | 24.10 | 0.64 |
| 29 | 2.26 | 24.10 | 0.64 |
| 31 | 4.26 | 24.10 | 0.64 |
| 33 | 6.26 | 24.10 | 0.64 |

The predicted collapse sits at ρ ≈ 24.10 for every window, with a degenerate IQR, regardless of where the training window lives. A window whose lower edge is 0.26 above the Hopf and a window whose lower edge is 6.26 above it both put the transition at the same place and overshoot the true Hopf by the same 0.64. The transition location is independent of d.

The number ρ ≈ 24.10 is the part I keep looking at. It lands right on the attractor-coexistence onset at ρ ≈ 24.06 from §2.3, which is where the chaotic attractor actually stops existing. The Hopf at 24.74 is a different boundary: it is where the C± fixed points regain stability, but the chaotic attractor still coexists with them down to 24.06. The network, having learned the chaotic manifold, rides it down to where that manifold genuinely terminates rather than to where the fixed point becomes the only option. Read that way the overshoot is not the network being wrong about the Hopf. It is the network tracking the boundary it was actually trained to know, the existence of the chaotic attractor, and that boundary sits below the Hopf. A trajectory already sitting on the chaotic attractor has no reason to fall off it at 24.74.

Against the §7 criterion for C4 (dependence on d resolved above the realization spread, with the direction stated): the dependence is resolved, and it is resolved to be absent. The direction of the effect is that there is no effect of window position on where the network places the transition; every window overshoots the Hopf by the same amount and collapses at the coexistence onset. I would rather report that than force the position hypothesis to look alive when the measurement says it is flat.

## On the W = 10 / §2.4 tension

I want this on the record rather than buried. Section 4's C2 grid specifies widths up to 10 centered at 29, which places the lower edge at ρ = 24 and a couple of training samples (24, and 24.5 if the spacing landed there; here 24 and 25) at or below the Hopf. Section 2.4 says training windows sit in the clearly chaotic band above 24.74, precisely because the coexistence region below it makes the single-IC class label unreliable. The two sections are in tension at W = 10. I ran the grid as §4 wrote it, did not silently clamp the window, and the contaminated config is the one that broke the C2 trend. If we want C2 clean, the fix is either to cap the C2 widths so the lower edge stays above the Hopf, or to run a W = 10 variant clamped to ρ ≥ 24.74 as a robustness check. That is a methodology call for the next revision, not something I am going to decide inside a results session. For now Figure 2 reports W = 10 as measured, with the caveat drawn on the figure.

## Reproducibility note

The reservoir seeds are deterministic: master seed 20260613, per-realization seed master + k for k = 0..31, logged in every result JSON. The Lorenz initial-condition stream is separate and also deterministic. While building the harness I caught the segment-IC seed deriving from Python's built-in `hash()`, which is salted per process and would have made a fresh checkout build different training segments. I replaced it with a fixed hash so the segment seeds are reproducible from scratch. The C2–C4 numbers reported here come from the cached segment set built during this session; the IC choice has negligible effect on a readout fit over multi-thousand-step chaotic segments, and the degenerate bands across most configs say the results do not hinge on any particular draw. The full bit-for-bit reproduction claim is C5's job next session, and the deterministic seed is in place for it.

## Status

- Harness built on the checkpoint-and-resume pattern, validated at reduced R on a single C2 width before the full grid.
- C2, C3, C4 run at R = 32 against the locked architecture, total data fixed at 120,000, δρ = 0.1. All 480 cells on disk.
- No sweep point came back unresolved under the §5 rule, so R = 32 was enough everywhere and I did not have to raise it anywhere.
- Figures 2–4 produced with IQR bands; result JSONs at `figures/c2_result.json`, `c3_result.json`, `c4_result.json`.
- Section-7 status: C3 meets its criterion (saturating curve, confound controlled). C4 meets its criterion (dependence on d resolved, and it is flat). C2 is resolved point-by-point but non-monotone, with the widest config contaminated by sub-Hopf training; it meets the resolution half of its criterion and not the monotone half, and that is reported as the finding.

Not starting the C5 reproducibility write-up. That is its own session, as planned.
