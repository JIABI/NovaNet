# Code–paper alignment audit

This file records what the revised repository implements and what is still
needed to reproduce the manuscript's numerical tables. The canonical file is
schema 5. Its current fingerprint is `d53396a525bf68cc`; run
`python scripts/check_consistency.py` to print the fingerprint computed from
the selected configuration instead of relying on this recorded value after an
edit.

| Reviewer concern | Implemented artifact | Verification |
|---|---|---|
| Public code used `SoftDP(horizon=1)` | `novanet/soft_dp.py` performs a masked, cost-form `H=6` recursion on `(incumbent, freeze counter)`. The actual first-step counter is an input, a visible incumbent is locked while frozen, and an unavailable incumbent can recover. | brute-force path, future-cost, mask, gradient, freeze-lock, and forced-recovery tests |
| Future planner inputs were unspecified | `novanet/forecast.py` aligns the horizon candidate sets by satellite ID. SGP4 supplies future geometry and direct TTL. `novanet/model.py` graph-encodes the current 6D snapshot and rolls the first five geometry/dwell fields forward through the GRU using the input layer's tied first-five-column projection. Deterministic future SINR is a separate energy-head input; future realized channel samples are not model inputs. | model test perturbs the unused future sixth field and checks that freeze reaches DP |
| Table I and code disagreed | `configs/paper.yaml` is the single source for 10°, top-8, 2400 s, 100/150 ms, 30 s freeze, 1500 bytes, 20/100 MHz, channel, traffic, and model values. | `tests/test_config.py` and `scripts/check_consistency.py` |
| Density test kept top-8 fixed | `experiments/density_convergence.py` runs the 60/120/240 × 8/16/32 design on shell-stratified nested orbit-balanced TLE subsets, saves epoch losses, and emits the defined convergence epoch. | CLI, shell-mixture, nested-selection, and experiment-contract tests |
| LCB swept only κ | `experiments/lcb_variance_sensitivity.py` has separate `nominal-ground` and `joint-high-mobility` protocols. Both vary measurement standard deviation and staleness while keeping the configured κ fixed, with LCB on/off rows. | experiment CLI/contract tests |
| Uncertainty calibration lacked a reproducible path | `experiments/lcb_calibration.py` regenerates the canonical 3,600-sequence stream and evaluates the residual mean/scale head on its exact final 20% validation partition. It writes row-level predictions, equal-count calibration bins, one-sided κ coverage, and protocol/provenance metadata. A separately named independent mode is diagnostic only. | calibration split/math/parser tests and CLI contract |
| 300 km/h setup lacked a physical model | `experiments/aerial_doppler.py` records 1 km altitude, 60/300 km/h, four headings, radial-velocity Doppler, tracking efficiency, residual error, and coherent interval. | channel Doppler and experiment-contract tests |
| Realized fading/noise could drift across consumers | `novanet/channel.py` provides `RealizedChannelTrace`, whose channel, measurement, Doppler-error, and blockage samples are keyed by UE, satellite, time bin, and stream rather than call order. Dataset labels, CHO replay, and service evaluation query the same trace contract. | keyed-sample order-independence plus dataset/CHO replay tests |
| HOF definition and head were not closed | `novanet/handover.py` monitors the stored target over the decision interval, starts at the first sustained TTT, and replays source/target SINR through execution on a 10-ms grid. HOF is conditional on an attempted execution and uses the configured bad-sample fraction; `novanet/model.py` and `novanet/losses.py` implement the pairwise head and attempted-pair BCE mask. | delayed-trigger, event-grid, conditional-mask, and head-gradient tests |
| Learned baseline rows lacked implementation | `novanet/baselines.py` defines source-conditioned GNN-only and DQN+GNN networks. DQN training uses sequential replay, Bellman TD targets, and a target network. Checkpoint loading requires training/validation/source-fidelity qualification. The local DHO-shaped MLP is explicitly marked as a non-table-grade surrogate rather than being presented as the cited interaction-trained protocol. | incumbent-sensitivity, replay/TD/target-isolation, checkpoint-qualification, and policy tests; qualified paper checkpoints remain external |
| Multi-UE mechanics were missing | `experiments/multi_ue.py` and `novanet/multi_ue.py` specify a 500 km disk, capacity/admission constraints, synchronous proposal/retention rounds, work-conserving PF reallocation after admission rejection, realized-load snapshots, freeze counters, and event-boundary CHO service without source/target double occupancy. | numerical scheduler, late-trigger, cross-epoch, and experiment-contract tests; a paper run still requires validation-selected `c_blk` |
| Latency model was not reproducible | `novanet/latency.py` implements Poisson arrivals, finite drop-tail FIFO, work-conserving service, preemptive CHO blackouts, serialization, 33.5 ms network delay, and shared 1 ms protocol processing. | deterministic, blackout, and overflow tests |
| Risk energy and losses were unclosed | `novanet/model.py`, `novanet/handover.py`, and `novanet/losses.py` implement the 6D current interface, 5D future rollout, direct propagated TTL, fixed-reference utility, transition cost, pairwise HOF head/BCE, residual NLL, path KL, and L2 regularization. | direct-TTL, exact-transition, head-gradient, and HOF event-grid tests |
| Communication assumptions were incomplete | `novanet/channel.py` implements the 12 GHz link budget, SINR with optional exogenous interference and receive-array gain, EIRP density, gas/rain approximation, Rician/shadowing terms, measurement filtering, Doppler compensation, and tracking loss. | geometry/channel, interference, and array-interface tests |
| Additional revision experiments lacked entry points | `experiments/` includes nominal evaluation, validation-only heuristic tuning, ablation, rain/bandwidth, variance/staleness, uncertainty calibration, aerial/Doppler, blockage, freeze, density/convergence, multi-UE, trace-based latency, and ephemeris-aging runners. | experiment CLI/contract tests; scientific rows still require the artifacts listed below |

## Verification scope

The test count below is updated only from a fresh `python -m pytest -q` run for
this working tree. Diagnostic runs exercise software paths; they are not
evidence that a manuscript table has been numerically reproduced.

- The final frozen-tree release gate collected and passed 132 tests with
  `python -m pytest -q`.
- `scripts/check_consistency.py` prints schema/fingerprint and deliberately
  blocks the bundled 2025 TLE with the paper's 2023 experiment date.
- The inference benchmark measures the neural encoder, learned heads, and
  Soft-DP only. It excludes SGP4/feature construction. The current measured
  artifact is `results/benchmark/inference_current.json`; it records the exact
  machine/software context and whole-worker peak-RSS semantics. The preserved
  `results/benchmark/inference.json` remains explicitly legacy-incompatible.

## Manuscript synchronization still required

The following items are not missing software. They are places where
`0814_v2.tex` must describe the implementation actually used, or where a real
author artifact/run is still needed. Changing the implementation merely to
match the shorter printed formula would alter the declared parameter count and
would require retraining.

1. `K_g=4` is the graph-neighbor count. The top-8 nominal candidate cap and
   the 8/16/32 density sweep must use the separate symbol `K_cand`.
2. The 249,603-parameter model uses input LayerNorm, two-linear residual graph
   blocks with LayerNorm, and a future projection tied to the first five
   columns of the current input projection. The method equations should state
   these details; they currently show a simpler untied architecture.
3. The sixth node field is the measured SINR in dB divided by the configured
   30-dB reference, not linear SINR divided by a dB quantity.
4. The channel code applies a 0.12-dB zenith gas term, an elevation-dependent
   8-mm/h base rain model, controlled 0/5/10-dB *excess* attenuation,
   Rician-15-dB plus 1.5-dB shadowing, and
   `sinc^2(f_res*T_coh)` Doppler loss with `T_coh=1e-4 s`. The manuscript must
   disclose these implemented terms if the reported rows were generated with
   them. Service rate/outage is sampled on the configured 5-s geometry grid;
   only CHO qualification/execution uses the 10-ms event grid.
5. Packet service is a piecewise-constant work-conserving bit server: each
   packet requires 12,000 service bits and pauses during a zero-rate or CHO
   interval. This is more precise than fixing serialization once at arrival
   and should be the wording used in the traffic description.
6. `--OrbitPrior` removes the five orbital/geometric *encoder channels* at
   inference time; candidate construction, direct TTL, geometric adjacency,
   and nominal physical SINR remain. All listed ablations are fixed-checkpoint
   inference-time bypasses, not separately retrained networks.
7. The array/orientation expression is executable through
   `receive_array_gain_linear` and the simulation-wide receive-gain callback.
   The reported `M=1` setting leaves it at unity, so no directional-array or
   pointing-error performance has been validated.
8. The training windows start with zero recurrent cache, while deployment can
   carry the satellite-ID cache across decisions. The GRU is trained through
   the six-step horizon, but no chronological/TBPTT training artifact exists;
   the manuscript should not imply that such episode-level training was run.
9. The weight-Pareto, joint 300-km/h variance/LCB, and current-resource
   benchmark runners now exist. Their measured outputs must still be run and
   reported before a response letter says those empirical requests are
   answered.

## Reproduction blockers

The repository does not contain all scientific artifacts and selected
constants needed to regenerate the paper's tables:

1. The paper starts at `2023-05-01T09:30:00Z`, whereas the bundled TLE median
   epoch is `2025-08-31T03:18:20.777472Z` (852.7 days apart).
2. `checkpoints/novanet_paper.pt` and the GNN-only, DQN+GNN, and DHO paper
   checkpoints are absent. The architectures and training code are present,
   but newly trained diagnostic models are not those checkpoints.
3. The per-method rate/packet/blackout traces behind the reported latency
   distributions are absent. New evaluations generate their own traces and
   the offline oracle internally, but cannot recover the original packet
   quantiles from aggregate values.
4. The manuscript names a validation-selected dummy blocking cost `c_blk` but
   does not provide its numeric value. The canonical config therefore leaves
   it `null`; a manuscript-grade multi-UE run must supply it explicitly.
5. The exact reference statistics/rule used for the normalized
   reward-equivalent score in the freeze sensitivity paragraph are absent. The
   code reports event-derived metrics but does not infer this normalization
   from printed values.
6. The density table's reported 60-satellite candidate-union statistics have
   not been regenerated from the missing historical TLE. They must be checked
   from the runner output before publication; the code does not copy the
   printed 8.4/48% values into a result file.
7. A common dummy threshold is only meaningful after all multi-UE method
   scores are calibrated to the same validation objective. The missing
   `c_blk` selection artifact is therefore also required to support the
   blocking comparison, not just to launch the runner.

The code rejects the date-mismatched TLE by default and rejects checkpoints
without matching configuration and TLE fingerprints. `--allow-stale-tle` and
`--diagnostic-no-blocking-cost` are for software diagnostics only.

The five aggregate Fig. 7 means identify the shared 1 ms component but do not
identify a common queue. Fitting one shared M/M/1 arrival rate plus one shared
offset leaves a 1.868 ms cross-method RMSE. Consequently the repository does
not introduce method-specific arrival rates or offsets to force the bars to
match. Exact numerical reproduction requires the historical TLE, all trained
paper checkpoints, the original rate/packet/blackout traces, the selected
`c_blk`, and the freeze-score normalization artifact.
