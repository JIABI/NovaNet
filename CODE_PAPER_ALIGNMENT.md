# Code–paper alignment audit

This file records what the revised repository implements and what is still
needed to reproduce the manuscript's numerical tables. The canonical
configuration fingerprint is `814db9d6a32562e0`.

| Reviewer concern | Implemented artifact | Verification |
|---|---|---|
| Public code used `SoftDP(horizon=1)` | `novanet/soft_dp.py` performs a masked, incumbent-conditioned, cost-form `H=6` recursion | brute-force path, future-cost, masking, and gradient tests |
| Future planner inputs were unspecified | `novanet/forecast.py` aligns the six candidate sets by satellite ID; future geometry comes from SGP4 and channel fields use only a decayed residual from the latest causal report | end-to-end diagnostic evaluation |
| Table I and code disagreed | `configs/paper.yaml` is the single source for 10°, top-8, 2400 s, 100/150 ms, 30 s freeze, 1500 bytes, 20/100 MHz, channel, traffic, and model values | `tests/test_config.py` and `scripts/check_consistency.py` |
| Density test kept top-8 fixed | `experiments/density_convergence.py` runs the full 60/120/240 × 8/16/32 factorial design on shell-stratified nested orbit-balanced TLE subsets, saves every epoch's losses, and emits the defined convergence epoch | CLI, import, shell-mixture, and nested-selection tests |
| LCB swept only κ | `experiments/lcb_variance_sensitivity.py` varies measurement variance/staleness while keeping the configured κ fixed | CLI and import checks |
| 300 km/h setup lacked a physical model | `experiments/aerial_doppler.py` records 1 km altitude, four headings, radial-velocity Doppler, tracking efficiency, residual error, and coherent interval | channel Doppler tests |
| Multi-UE mechanics were missing | `experiments/multi_ue.py` and `novanet/multi_ue.py` specify a 500 km disk, capacity/admission constraints, synchronous ranking/commit, PF allocation, load snapshots, and per-UE report updates | scheduler unit test and a five-UE diagnostic run |
| Latency model was not reproducible | `novanet/latency.py` implements Poisson arrivals, finite drop-tail FIFO, work-conserving service, preemptive CHO blackouts, serialization, 33.5 ms network delay, and shared 1 ms protocol processing | deterministic, blackout, and overflow tests |
| Risk energy and losses were unclosed | `novanet/model.py`, `novanet/handover.py`, and `novanet/losses.py` implement switch-gated dimensionless cost, pairwise HOF head/BCE, independent angular-speed cost, all claimed losses, and executable ablations | head-gradient, HOF event-grid, and `TransVel` tests |
| Communication assumptions were incomplete | `novanet/channel.py` implements the 12 GHz link budget, common EIRP density, gas/rain approximation, Rician/shadowing terms, measurement filtering, Doppler compensation, and tracking loss | geometry/channel tests |

## Checks completed

- 17 tests pass.
- One-epoch training, checkpoint save, provenance validation, checkpoint load,
  and one-user evaluation complete successfully in diagnostic mode.
- The multi-UE pipeline completes a five-UE/one-epoch diagnostic run.
- Every experiment entry point accepts `--help`, and the package builds as a
  wheel without downloading dependencies.
- The measured one-thread neural forward means are 1.64/3.00/9.02 ms for
  K=8/16/32 (100 warm-up, 1,000 measured passes); metadata is stored in
  `results/benchmark/inference.json`.

## Reproduction blockers

The source logic is closed, but the repository does not contain the immutable
scientific artifacts needed to regenerate the paper's tables:

1. The paper starts at `2023-05-01T09:30:00Z`, whereas the bundled TLE median
   epoch is `2025-08-31T03:18:20.777472Z` (852.7 days apart).
2. `checkpoints/novanet_paper.pt` is absent.
3. The per-method packet arrival/rate/CHO traces behind the reported latency
   distributions are absent.

The code rejects the date-mismatched TLE by default and rejects checkpoints
without matching configuration and TLE fingerprints. `--allow-stale-tle` is
for software diagnostics only.

The five aggregate Fig. 7 means identify the shared 1 ms component but do not
identify a common queue. Fitting one shared M/M/1 arrival rate plus one shared
offset leaves a 1.868 ms cross-method RMSE. Consequently the repository does
not introduce method-specific arrival rates or offsets to force the bars to
match. Exact numerical reproduction requires the historical TLE, trained
checkpoint, and original rate/packet/CHO traces.
