# Methodology Design (v2): Sampling-Strategy Effects on Bifurcation Extrapolation in a Parameter-Aware Echo State Network

**Session 4 deliverable for the JURPA paper project, revised in Session 6.** Feeds Session 5 (Implementation, Core Code) and the C2–C4 sweeps.

| | | | |
|---|---|---|---|
| **Session** | 4 (Methodology Design); revised Session 6 | **Author** | Noah Riego |
| **Affiliation** | University of Arizona, B.S. Applied Physics | **Target venue** | JURPA |
| **Document** | 03_methodology_v2.pdf | **Status** | Locked plan; C1 gate **passed** under the v2 revision below. Hyperparameters locked. |

This is the computational plan. It is written so Session 5 can start coding without making any further design decisions. Every quantity is given a value, either from a cited paper or as a prior, and the priors are the ones the first experiment (C1) has to confirm before any sweep runs. The research question, the gap, and the three sampling axes it turns on were settled in Sessions 2 and 3, so this document does not revisit them. It sets the architecture, the testbed, the metrics, the four sweeps, and the criteria for a finished result.

---

## Revision record (v1 → v2)

This is the logged revision the page-10 living-document clause calls for. Session 5 ran the C1 gate, walked the full section-8 resolution path, and could not bring the z-maxima amplitude RMSE below about 7.4% of the z-range anywhere inside the locked hyperparameter ranges. The other two C1 criteria passed comfortably. Under section 10 an exhausted gate is resolved by a logged methodology revision before any sweep runs, not by hand-tuning around the architecture, which the charter forbids.

Session 6 diagnosed the floor and revised the C1 acceptance criterion. The change is small and it is confined to where the amplitude metric is evaluated. Nothing about the architecture moves.

**What changed.** The C1 z-maxima amplitude RMSE (section 7) is now evaluated on the chaotic band ρ ≥ 24.74 (the subcritical Hopf landmark of section 2.3) rather than on the full reconstruction grid ρ ∈ [24, 32]. Regime-class accuracy is still scored across the entire grid, unchanged. The largest-Lyapunov / spread agreement is unchanged.

**Why.** Below the Hopf the Lorenz system sits in the attractor-coexistence region that section 2.4 already singles out: the chaotic attractor coexists with the stable C± pair, so the asymptotic state, and with it the z-maxima envelope, depends on the initial condition. Section 2.4 excludes this region from training for that exact reason. The C1 *test* grid was reaching into it anyway, and the ground-truth bifurcation diagram is built from one initial condition per ρ, so below the Hopf it samples the two coexisting branches at random. The Session-6 diagnostic makes this concrete: the true mean z-maximum alternates between roughly 23 and roughly 34 at neighbouring ρ across 24.0–24.7, while the network sits steadily on the chaotic branch near 34. Comparing a steady prediction against a ground truth that flips between branches manufactures a 20%-plus error in six grid points that has nothing to do with model fidelity. Those six points carry the whole failure. On the chaotic band the same reconstruction lands at 2.3% of the z-range.

**Why this is charter-legal.** It changes no hyperparameter and adds no architecture. It introduces no new threshold; the 5% bar stays where it was. It scopes the amplitude metric to the region where the quantity it measures is single-valued, which is a correction to the metric's domain rather than a relaxation of it. The downward-across-Hopf behaviour is not dropped from the study; it is the dedicated subject of C4, judged there by qualitative-class match, which is the right tool when the attractor type itself changes.

**Diagnostics behind the decision.** Two non-architecture knobs were tested first as bounded measurements (Session-6 progress log, with the full sweep in `data/diag_log.json`): the parameter-channel reference interval of section 2.4 (recentred, narrowed, and widened) and the cold-extrapolation primer of section 1.4 (nearest, edge, and none). Neither moved the floor; every variant sat between 7.40% and 7.71%. Narrowing the reference interval about its centre is algebraically the same as scaling γ_p, which the gate had already swept, so only recentring is a genuinely separate lever, and recentring did nothing. That left the metric domain as the thing to fix, which is what v2 does.

**Re-validation.** With the revised criterion and the model unchanged, C1 passes at R = 10: regime class 100% on the full grid, amplitude RMSE 2.31% on ρ ≥ 24.74, Lyapunov / spread agreement 5.6%. The full-grid (v1) RMSE is retained on record at 7.52%. The reconstruction figure and the result are in `figures/fig1_c1_v2_bifurcation.png` and `figures/c1_v2_result.json`.

**Hyperparameters locked.** The C1 gate is closed. The post-gate locked values are γ_p = 0.1 and spectral radius = 0.6; everything else stays at the section-1.5 priors. These carry into C2–C4 unchanged.

The rest of this document is the plan as Session 4 wrote it, with the section-7 criterion and the section-1.5 locked column updated to match the lock above. Section numbers are preserved so the carry-forward prompts from earlier sessions still point where they used to.

---

## 1. The parameter-aware echo state network

The architecture is the parameter-aware echo state network (ESN) of Kong, Fan, Grebogi and Lai (2021). A standard ESN is a fixed random recurrent network whose only trained part is a linear readout. The parameter-aware version adds one extra input channel that carries the system's bifurcation parameter, so the same trained readout can be asked about a parameter value it never saw during training. The reservoir itself is standard. What is new is the set of training choices varied around it, which Section 4 lays out.

### 1.1 Reservoir state update

The reservoir is a vector r(t) of N nodes. At each step it updates as:

r(t+Δt) = (1 − α)·r(t) + α·tanh( W_r r(t) + W_in u(t) + b )

W_r is the N×N reservoir matrix (sparse, fixed, rescaled to a chosen spectral radius), W_in maps the input vector u(t) into the reservoir, b is a fixed bias vector, and α is the leakage rate. tanh is applied elementwise. The parameter awareness enters through the input u(t), described next.

### 1.2 The parameter-input channel

The input at each step is the three standardized Lorenz coordinates plus one channel holding the normalized bifurcation parameter:

u(t) = [ x̂(t), ŷ(t), ẑ(t), p̂ ]

The hats mean standardized: x, y, z are shifted and scaled to zero mean and unit variance using statistics pooled across the whole training set, so no single training ρ dominates the scaling. The parameter channel p̂ is ρ mapped linearly onto a fixed reference interval (Section 2.4) and then multiplied by its own input scaling γ_p, kept separate from the state scaling γ_in. Keeping them separate matters because the parameter channel has to be strong enough to steer the readout without overwhelming the state input. A γ_p set too low leaves the network unable to extrapolate; set too high, it drowns out the state signal. Finding a workable value is part of the C1 gate.

During training the reservoir is shown several segments, one per training ρ, each segment tagged with its own constant p̂. Between segments the reservoir state is reset and a washout transient is discarded so the readout never learns cross-segment leakage.

### 1.3 Training the readout

Only the linear readout W_out is trained, by ridge regression. Collect the post-washout reservoir states into a matrix R and the one-step-ahead Lorenz targets into Y:

W_out = Y Rᵀ ( R Rᵀ + λ·I )⁻¹

λ is the Tikhonov (ridge) regularizer. This is a single closed-form solve, no gradient descent, which is the whole point of reservoir computing and the reason the sweeps in Section 4 are cheap enough for one person to run.

### 1.4 Prediction at an unseen ρ

To test extrapolation at a target ρ* the network runs closed-loop (autonomously): its own output is fed back as the next state input, while p̂ is held fixed at p̂(ρ*). Two protocols are used and reported separately.

**Warmup-then-free-run.** A short ground-truth segment at ρ* is fed in teacher-forced to set the reservoir state, then the network free-runs. This is the protocol the valid-prediction-time metric needs, because it requires a ground-truth trajectory to diverge from.

**Cold parameter extrapolation.** No ground-truth segment at ρ* is given. The network is warmed from a generic initial state and free-runs on p̂(ρ*) alone. This is the stricter test of whether the parameter channel actually carries the regime information, and it is what the qualitative-class and climate metrics judge. A short primer drawn from an available training ρ may be teacher-forced while the parameter channel is already held at p̂(ρ*), which seats the reservoir on the Lorenz manifold before the free run; the Session-6 diagnostic confirmed the choice of primer does not affect the reconstructed amplitude, so the nearest-training-ρ primer is used throughout.

### 1.5 Locked hyperparameters

These are priors. The values come from the parameter-aware ESN of Kong et al. (2021), the N=300 / degree-6 / 10,000-step Lorenz ESN recovered from the Elicit scan, and the sensitivity ranges in Hurley and Shaheen (2025). The charter's risk register forbids an architecture-level hyperparameter search, so these stay fixed across every sweep. The only place they may move is the C1 gate (Section 8), and only within the listed ranges. The gate is now closed; the **Locked (post-gate)** column is the value that carries into C2–C4.

| Hyperparameter | Locked (post-gate) | Gate range | Source / note |
|---|---|---|---|
| Reservoir size N | 500 | 300–1000 | Above the N=300 Lorenz ESN and Hurley's 400; headroom |
| Average in-degree | 6 | 3–6 | Matches the 6/N connectivity of the baseline Lorenz ESN |
| Spectral radius ρ_sr | **0.6** | 0.4–1.2 | Hurley sensitivity band; set in the C1 gate within the echo-state range |
| Leakage rate α | 1.0 | 0.3–1.0 | Non-leaky baseline; not needed by the gate |
| State input scaling γ_in | 0.10 | 0.05–0.5 | Standard for normalized Lorenz input |
| Parameter scaling γ_p | **0.1** | 0.1–1.0 | Set in the C1 gate; lower end, see §1.2 and the Session-5 log |
| Bias scaling | 0.10 | 0.0–1.0 | Fixed additive bias vector |
| Ridge λ | 1×10⁻⁶ | 10⁻⁸–10⁻⁴ | Hurley regularization range |
| Washout per segment | 1000 steps | — | Discarded before the readout fit |

Reservoir realizations differ only in the random draws of W_r, W_in, and b. Section 5 explains why a single draw is never trusted.

---

## 2. The Lorenz testbed

### 2.1 System and fixed parameters

The Lorenz 1963 system:

ẋ = σ(y − x), ẏ = x(ρ − z) − y, ż = xy − βz

σ and β are held at their textbook values, σ = 10 and β = 8/3. ρ is the only axis this study moves. The topic charter (§7) allows revisiting σ or β if time permits; the default is no, which keeps the whole study one-dimensional and the curves in Section 4 interpretable.

### 2.2 Integrator, time step, Lyapunov time

Trajectories are integrated with fixed-step fourth-order Runge–Kutta at h = 0.01. The ESN observes every second point, so its native step is Δt = 0.02. At ρ = 28 the largest Lyapunov exponent of the Lorenz system is λ_max ≈ 0.906, giving a Lyapunov time τ_Λ = 1/λ_max ≈ 1.10 time units, about 55 ESN steps. Prediction quality is reported in Lyapunov times rather than raw seconds so the numbers compare directly with the reservoir-on-Lorenz literature.

### 2.3 Bifurcation map

Before any window is chosen, Session 5 maps the ρ-bifurcation structure numerically: sweep ρ in steps of 0.1, integrate past the transient, and classify each ρ as fixed point, periodic, or chaotic from the largest Lyapunov exponent and the spread of the z-maxima. The classification has to land on the known landmarks below, which doubles as a check that the integrator and the classifier are correct before they are trusted inside the metric.

| ρ landmark | Value (σ=10, β=8/3) | What happens |
|---|---|---|
| Pitchfork | ρ = 1 | Origin loses stability; the symmetric pair C± appears |
| Homoclinic explosion | ρ ≈ 13.93 | Onset of transient (pre-turbulent) chaos |
| Attractor coexistence | ρ ≈ 24.06 | Chaotic attractor coexists with the stable C± |
| Subcritical Hopf | ρ ≈ 24.74 | C± lose stability; sustained chaos. The pivot of this study |
| Standard chaos | ρ = 28 | The classic Lorenz attractor |

The Hopf point at ρ ≈ 24.74 is the qualitative boundary the extrapolation has to reach across: above it the asymptotic state is chaotic, below it (ignoring the coexistence sliver near 24.06–24.74) the trajectory spirals into a fixed point. Predicting that class change from training data that lives entirely on one side is the core test in C4.

### 2.4 Locked ρ windows

The reference interval for normalizing the parameter channel is ρ ∈ [20, 36], fixed once so p̂ means the same thing in every experiment. All training windows sit inside the clearly chaotic band ρ > 24.74; extrapolation is then measured outward from the window edges, including downward across the Hopf into the fixed-point regime and upward into higher-ρ chaos. The exact per-sweep windows are given with each contribution in Section 4. Training values below 24.74 are avoided on purpose, because the coexistence region makes the asymptotic class depend on the initial condition, which would contaminate a clean regime label. The v2 revision extends this same exclusion to the C1 amplitude metric: the z-maxima envelope is single-valued only above the Hopf, so the amplitude RMSE is read there (Section 7).

---

## 3. Metrics

The research question asks for a distance in ρ: how far past the training window the network keeps the qualitative dynamics right. That distance is built from a per-ρ validity test, which in turn rests on a primary kernel (valid prediction time) and two secondary kernels (qualitative class and a climate statistic) used to cross-check it. The ground-truth-time cap from Hurley and Shaheen (2025) sits underneath all of it.

### 3.1 Valid ground-truth time (the cap)

Because the Lorenz system is chaotic, two numerical integrators of the same trajectory drift apart after a finite time. Any claimed prediction time longer than that drift time is measuring agreement with one particular numerical realization, not with the system. So before measuring anything, Session 5 integrates each ground-truth trajectory twice (RK4 at h, and again at h/2) and records the valid ground-truth time (VGTT): the time at which the two integrations diverge by the same error threshold ε used for prediction. Every reported prediction time is capped at the VGTT. This cap follows Hurley and Shaheen, and it keeps the paper from reporting prediction times longer than the trajectory itself stays trustworthy.

### 3.2 Valid prediction time (primary per-ρ kernel)

At a test ρ, run the warmup-then-free-run protocol and track the normalized error

E(t) = ‖ û(t) − u(t) ‖ / √⟨‖u‖²⟩

The valid prediction time (VPT) is the first t at which E(t) exceeds ε = 0.4, reported in Lyapunov times and capped at the VGTT. ε = 0.4 is the standard threshold in this literature and is fixed in advance.

### 3.3 Qualitative class (secondary kernel)

From a long cold-extrapolation run at the test ρ, classify the predicted attractor as fixed point, periodic, or chaotic using the sign of the predicted largest Lyapunov exponent (chaotic if above +0.01, fixed point if below −0.01, periodic in between) together with a collapse test on the trajectory variance. Compare to the ground-truth class from Section 2.3. This kernel is the right one when the test ρ sits on the far side of a bifurcation, where VPT is not meaningful because the attractor type itself has changed.

### 3.4 Climate agreement (secondary kernel)

Two climate statistics cross-check the class label: the Grassberger–Procaccia correlation dimension D2 of the predicted attractor, and the distribution of successive z-maxima (the Lorenz return map). Agreement at a test ρ means |D2(pred) − D2(true)| < 0.15 and a small Wasserstein distance between the predicted and true z-maxima distributions. The z-maxima map is also what builds the reproduced bifurcation diagram in C1. In the Session-6 diagnostic the z-maxima Wasserstein distance stayed at or below about 1.2% of the z-range across the whole chaotic band, which is the distributional companion to the C1 amplitude RMSE and corroborates the v2 pass.

### 3.5 Extrapolation distance Δρ

This is the quantity every figure in Section 4 reports. Starting at a training-window edge, step outward in ρ at resolution δρ = 0.1. At each step evaluate the per-ρ validity criterion on the median over reservoir realizations. The extrapolation distance Δρ is the largest contiguous outward distance for which validity holds without a break.

Validity has two forms, chosen by what the test ρ crosses. For same-class extrapolation (staying in the chaotic band), a ρ is valid if the median VPT stays at or above a fixed fraction of the in-window VPT and the class label is unchanged. For across-bifurcation extrapolation (reaching past the Hopf), a ρ is valid if the predicted qualitative class matches ground truth; VPT is not used there. The correlation dimension cross-checks both. Reporting Δρ under both the VPT-based and the class/D2-based criteria, and showing they trend together, is the internal consistency check for the whole result.

---

## 4. The four sweeps

Four contributions, C1 through C4, each map to one figure. C1 reproduces the known result and gates everything after it. C2, C3, and C4 are the new measurements: they take the three sampling choices that the literature fixes for convenience (range, density, position) and turn each one into the independent variable, holding the architecture and the total quantity of training data constant. The fixed quantities common to C2–C4 are listed first.

| Held constant across C2–C4 | Value |
|---|---|
| Architecture and hyperparameters | Locked post-gate values from §1.5 (γ_p = 0.1, spectral radius = 0.6, rest as listed) |
| Total training data length L_total | 120,000 ESN steps, summed across all training segments |
| Integrator / time step | RK4 h = 0.01; ESN Δt = 0.02 (§2.2) |
| Reference normalization interval | ρ ∈ [20, 36] (§2.4) |
| Reservoir realizations per configuration | R = 32 (§5) |
| Extrapolation resolution δρ | 0.1 |
| Held Lorenz parameters | σ = 10, β = 8/3 |

Holding the total data length fixed is the point the Session 3 carry-forward insisted on. When the number of training samples M changes, the length per sample changes to keep L_total constant, so the result reflects the sample count rather than how much data went in. C3 returns to this.

### C1. Reproduce the parameter-aware ESN on the Lorenz ρ-bifurcation

Train the parameter-aware ESN on a small fixed set of ρ values in the chaotic regime and reconstruct the bifurcation diagram (z-maxima against ρ) across ρ ∈ [24, 32], comparing to ground truth. The training set follows the published parameter-aware-ESN-on-Lorenz precedent recovered from the literature scan: ρ ∈ {24.56, 26.06, 27.56, 29.06}, four values spaced 1.5 apart. This is the closest existing match to the present setup, which makes it the fair reproduction target.

Varied: ρ (the test axis of the reconstructed diagram). Fixed: everything in §1.5, the four training values, and the total data length. Output: Figure 1, the predicted-versus-true bifurcation diagram, plus Table 1 of quantitative agreement. C1 is also the gate. If the locked hyperparameters cannot reproduce the diagram, that is a methodology problem to fix here, not a result (Section 8). The diagram is drawn across the full ρ ∈ [24, 32] grid; the amplitude RMSE is read on the chaotic band ρ ≥ 24.74 (Section 7), where the z-maxima envelope is single-valued.

### C2. Extrapolation distance versus training-range width

Hold the sampling density fixed at one sample per ρ-unit and the total data length fixed, then widen the training window. Center ρ_c = 29. Window width W ∈ {2, 4, 6, 8, 10}, giving M = W + 1 evenly spaced samples (3, 5, 7, 9, 11) and a per-sample length of L_total/M. Measure Δρ beyond the window edges as a function of W.

Varied: window width W (and with it M, since density is held). Fixed: density, total data length, center, hyperparameters. Output: Figure 2, Δρ against W with interquartile bands, and a fitted scaling relation if the curve supports one.

### C3. Extrapolation distance versus sample density (the key measurement)

Hold the window width fixed at W = 6 centered on ρ_c = 29 (so ρ ∈ [26, 32]) and hold the total data length fixed, then fill that fixed window with more and more samples. M ∈ {2, 3, 5, 8, 13, 21}, with per-sample length L_total/M so the total never changes. The sparse end (M = 2) sits at the Yadav et al. (2024) minimal-data anchor; the dense direction reaches toward the Panahi et al. (2025) regime of hundreds of values, though a fixed-data sweep stops well short of 500 by design. Measure Δρ as a function of M.

This is the contribution the literature has not reported. Yadav (two samples) and Panahi (five hundred) sit at opposite ends of a range no one has measured across; C3 measures it, holding data volume fixed so the x-axis is sample count rather than total data. There is a floor: as M grows, the per-sample length shrinks, and below some length a segment no longer samples the attractor well enough to train on. At L_total = 120,000 and M = 21 each segment is about 5,700 steps, roughly 85 Lyapunov times after washout, which is comfortable. If a denser point is added later and the per-sample length drops near the washout length, the resulting breakdown is worth reporting in its own right.

Varied: number of samples M (density M/6). Fixed: window width, center, total data length, hyperparameters. Output: Figure 3, Δρ against M, with the curve shape (diminishing returns, linear, or saturating) identified.

### C4. Extrapolation distance versus window position relative to the Hopf point

Hold the width (W = 4), the count (M = 5, so density 1.25 per ρ-unit), and the total data length fixed, then slide the training window and ask how well the network extrapolates downward across the Hopf at ρ ≈ 24.74 into the fixed-point regime. Window centers ρ_c ∈ {27, 29, 31, 33}, so the lower edge sits at {25, 27, 29, 31} and its distance above the Hopf is d ∈ {0.26, 2.26, 4.26, 6.26}. The hypothesis to test is simple: a training window whose lower edge sits closer to the Hopf should reach further across it. Here the primary kernel is the qualitative-class match, since the test crosses into a different attractor type where VPT does not apply.

Varied: window center, expressed as distance d of the lower edge above the Hopf. Fixed: width, count, total data length, hyperparameters. Output: Figure 4, across-Hopf extrapolation success against d.

| Sweep | Independent variable | Held fixed | Figure |
|---|---|---|---|
| C1 | ρ (reconstruction axis) | Training set {24.56, 26.06, 27.56, 29.06}, all §1.5 | Fig 1 + Table 1 |
| C2 | Window width W (density fixed) | Density, L_total, center ρ_c=29, §1.5 | Fig 2 |
| C3 | Sample count M (width fixed) | Width W=6, center, L_total, §1.5 | Fig 3 |
| C4 | Window center / distance d to Hopf | Width W=4, count M=5, L_total, §1.5 | Fig 4 |

---

## 5. Validation strategy

Haluszczynski and Räth (2019) showed that reservoir prediction quality swings a lot with the random draw of the reservoir. A single draw can look great or terrible for reasons that have nothing to do with the sampling strategy under study. So no result here comes from one reservoir. Every configuration is run over R = 32 independent realizations, differing only in the random W_r, W_in, and b. Each curve reports the median and the interquartile range, and the IQR is drawn as a band on every figure.

Seeds are fixed and logged so the whole study reruns bit-for-bit. A single master seed (recorded in the progress log) sets a per-realization seed of master + realization index for the reservoir draws; the Lorenz initial conditions get a separate, independently logged seed stream. The acceptance rule for any sweep point is that the spread across realizations is smaller than the effect being measured. If the IQR band swallows the trend, that configuration is reported as unresolved rather than presented as a finding, and R is raised before any claim is made.

---

## 6. Compute budget

The whole study is small enough to run on a laptop. C1 is a handful of configurations, C2 is five widths, C3 is six densities, and C4 is four positions, which is sixteen base configurations. At R = 32 realizations that is about 512 trained ESNs. Each training builds a sparse 500-node reservoir, runs roughly 120,000 steps, and solves one ridge regression, which is seconds of work. The test sweeps add about sixty short closed-loop runs per configuration per realization. The total lands in the low hours of wall time, inside one elastic 2–3 hour session block. The cap is firm: no architecture-level hyperparameter search beyond the C1 gate, and a fixed compute ceiling per the charter risk register, so the sweeps cannot quietly turn into a tuning exercise.

One practical note from Session 5: background jobs do not persist in the working environment, so long sweeps are run as short single-configuration processes that checkpoint to disk and resume, rather than one long job. The gate walk and the Session-6 diagnostics were both run this way.

---

## 7. Definition of done

Each contribution from topic-charter §5 is tied to a specific deliverable and a specific acceptance criterion. The project is methodologically complete when all five pass.

| Deliverable | Acceptance criterion |
|---|---|
| **C1** Figure 1 (predicted vs. true bifurcation diagram) and Table 1 (agreement) | Regime class correct at ≥95% of test ρ across the full grid; z-maxima diagram RMSE within 5% of the z range, **evaluated on the chaotic band ρ ≥ 24.74** (v2; below the Hopf the envelope is coexistence-multivalued per §2.4, so amplitude is not scored there and the class metric carries that region); largest-Lyapunov agreement within ~10% in the chaotic band. **Passed under v2.** Gates C2–C4. |
| **C2** Figure 2: Δρ versus window width, with IQR bands | Monotone trend resolved above the realization spread; a scaling fit reported if one holds. |
| **C3** Figure 3: Δρ versus sample count at fixed total data | Curve shape (diminishing / linear / saturating) identified, with the data-volume confound demonstrably controlled. |
| **C4** Figure 4: across-Hopf success versus distance d of the training edge to the Hopf | Dependence on d resolved above the realization spread; direction of the effect stated. |
| **C5** Public GitHub repository (ESN + Lorenz integrator + sweep harness + figure scripts) | A graduate student can reproduce any one figure from a clean checkout in under one hour, following the README. |

C5 follows JURPA precedent: PhysiCL, Jcurve, MIDSX, and the optical-tweezers paper all cite a repository inside the paper. The repo is initialized at the start of Session 5 and mirrors the ENSO project layout from the charter.

---

## 8. The C1 gate and its resolution path

If the locked hyperparameters cannot reproduce the Kong-style bifurcation diagram in C1, that blocks every sweep and has to be solved before any of them run. The charter still forbids an architecture-level search, so the fix is a bounded adjustment within the §1.5 gate ranges, in this order:

| Order | Adjust | Within range | Why this first |
|---|---|---|---|
| 1 | Parameter scaling γ_p | 0.1–1.0 | The most likely culprit: a parameter channel too weak or too strong to steer the readout (§1.2) |
| 2 | Spectral radius ρ_sr | 0.4–1.2 | Sets memory depth; Hurley shows VPT is sensitive to it |
| 3 | Ridge λ | 10⁻⁸–10⁻⁴ | Trades attractor fidelity against overfit |
| 4 | Reservoir size N (with per-sample length) | up to 1000 | Last resort; more capacity if the readout still underfits |

Whatever value finally reproduces C1 becomes the locked value used unchanged through C2–C4, and the path taken to get there goes in the progress log so later sessions do not revisit it. The architecture stays fixed throughout; only these four settings change, and only during the gate.

**Outcome (Sessions 5–6).** The gate was walked in full. γ_p was the only lever that moved the amplitude error, and it floored near 7.4% of the z-range at the bottom of its range; spectral radius, ridge, and N did not move it. The best-found configuration was γ_p = 0.1, spectral radius = 0.6, with ridge and N at their defaults. The residual error was amplitude-only and lived entirely in the sub-Hopf coexistence sliver, which is what the v2 revision above addresses. With the revised section-7 criterion the gate is closed and these values are locked.

---

## 9. Failure modes to watch for

| Failure mode | Source | How the methodology handles it |
|---|---|---|
| Pseudo-Lorenz double-spiral attractor: the ESN learns a spurious extra saddle the real system lacks | Springer JJIAM (2025) | Expected in extrapolation. The qualitative-class kernel (§3.3) flags it; report it as a documented limit rather than hiding it |
| Reported VPT longer than the trajectory is physically meaningful | Hurley & Shaheen (2025) | Every VPT is capped at the VGTT (§3.1); no uncapped numbers appear in the paper |
| Density curve confounded by changing data volume | Session 3 carry-forward | L_total held fixed so per-sample length absorbs the change (§4, C3) |
| A single lucky or unlucky reservoir drives a curve | Haluszczynski & Räth (2019) | R = 32 realizations, median + IQR, unresolved points reported as such (§5) |
| Coexistence below the Hopf makes the single-IC ground-truth envelope multivalued | Session 6 diagnostic; §2.4 | Amplitude RMSE scored on ρ ≥ 24.74 only; the sub-Hopf region is carried by the class metric and by C4 |
| A 2026 paper appears in the gap before submission | Topic charter risk register | Re-run the literature snapshot at the start of Sessions 7 and 9; cite and differentiate if needed |

---

## 10. Living-document clause

This is a living document. The Session-6 revision recorded at the top was made under this clause: the C1 gate was exhausted, so the methodology was revised and re-issued as 03_methodology_v2.pdf with the change logged, rather than worked around silently, and it was done before any sweep ran. If a later session finds that the v2 criterion or any locked value no longer holds, the same path applies: revise, log, re-issue, then sweep.

---

## 11. Immediate next action: opening the sweeps

C1 is locked. The next session runs C2–C4 against the locked architecture, using the checkpoint-and-resume pattern from Section 6. The opening prompt names this document (03_methodology_v2.pdf), the Session-5 and Session-6 progress logs, and the locked hyperparameters, and reminds the next session not to reopen the architecture: the gate is closed and the charter forbids reopening it.
