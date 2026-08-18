# Experiment entry points

Experiment runners read `configs/paper.yaml` (currently schema 5, fingerprint
`d53396a525bf68cc`) and compute summaries from generated raw rows. They never
copy values from manuscript tables into outputs. Diagnostic runs with stale
artifacts are labeled as diagnostics, not reproductions.

The exact revision experiments require the historical 2023 TLE snapshot and
the corresponding `checkpoints/novanet_paper.pt`.  The checked-in newer TLE can
be used with `--allow-stale-tle` only for software checks.

The nominal evaluator implements the specified deterministic baselines:
Max-Elevation, Max-ServeTime, Periodic-HO-16, Skip-1/2, Dwell-Aware, and
Rate-Dwell. `novanet/baselines.py` and `train_learned_baseline.py` provide a
source-conditioned GNN-only regressor and an offline sequential DQN+GNN trained
with replay, TD targets, and a target network. The local DHO-shaped model is
explicitly a surrogate and is not eligible for the cited DHO table row. The
qualified author-trained checkpoints for those rows are still absent. The
evaluator generates the offline oracle internally by replaying the same
held-out realized link and CHO events under the same energy and freeze rules.
Missing learned rows are not approximated with NovaNet ablations or copied
from the manuscript.

```bash
python -m experiments.evaluate_paper \
  --checkpoint checkpoints/novanet_paper.pt \
  --gnn-checkpoint checkpoints/gnn_only_paper.pt \
  --dqn-checkpoint checkpoints/dqn_gnn_paper.pt \
  --dho-checkpoint checkpoints/dho_paper.pt

python -m experiments.evaluate_paper --baselines-only --allow-stale-tle
```

The second command is a software diagnostic for deterministic baselines when
the bundled date-mismatched TLE is still present.

## Validation-only heuristic tuning

Periodic-HO, Dwell-Aware, and Rate-Dwell are selected before test evaluation
on common held-out scenarios and channel seeds. Every candidate is scored by
the lower-is-better realized `mean_target_cost` returned by the same-energy
oracle replay; throughput or a test-set metric is not used for selection.

```bash
python -m experiments.tune_heuristics \
  --periods 8,16,24 --dwell-thresholds 0.05,0.10,0.15 \
  --rate-dwell-dwell-weights 0.25,0.50,0.75 \
  --rate-dwell-switch-penalties 0.10,0.20,0.30 \
  --users-per-seed 20
```

The default validation seeds are distinct from both the training seed and the
reserved test seed. The runner writes `raw.csv`, `summary.csv`, `selected.csv`,
and `protocol.json` under `results/heuristic_tuning/`; no selected value is
preloaded from the manuscript.

## Planning and training ablations

```bash
python -m experiments.ablation \
  --variants Full,OrbitPrior,DynAdj,Temporal,Planner,UncLCB,TransTTL,TransHOF

python train_learned_baseline.py --kind gnn_only
python train_learned_baseline.py --kind dqn_gnn
python train_learned_baseline.py --kind dho --allow-surrogate-dho \
  --output checkpoints/dho_surrogate_diagnostic.pt
```

NovaNet uses the current six-dimensional graph snapshot, a tied
five-dimensional geometry/dwell projection for future GRU steps, and a
freeze-aware `(incumbent, counter)` Soft-DP state. A newly trained diagnostic
checkpoint is not interchangeable with a missing paper checkpoint.
The default evaluator rejects checkpoints without complete training and
validation metadata and rejects the DHO surrogate unless the explicitly
diagnostic `--allow-surrogate-baselines` flag is supplied.

## Planner-weight sensitivity and Pareto validation

The weight sweep is run only on independent validation scenarios. It loads the
canonical checkpoint with an exact configuration fingerprint, strict model
shape, and matching TLE/subset provenance before changing any coefficient.
After that check, only the fixed planner weights `alpha,beta,c0,c1,c2` are
varied; `--include-lambda-u` also varies the manuscript LCB coefficient
`lambda_u` (named `lcb_kappa` in the YAML file). Learned parameters are never
changed or retrained by this runner.

```bash
python -m experiments.weight_pareto \
  --checkpoint checkpoints/novanet_paper.pt \
  --design one-at-a-time --multipliers 0.5,1.0,1.5 \
  --validation-seeds 3025,4025,5025 --users-per-seed 20 \
  --include-lambda-u
```

The paired validation scenarios are identical across weight settings and must
not reuse the configured training seed. The command writes `raw.csv`,
`summary.csv`, `pareto.csv`, and `protocol.json` under
`results/weight_pareto/`. The Pareto file is computed from measured validation
outputs, maximizing effective throughput while minimizing handovers, pooled
HOF, and outage. The default one-at-a-time design has a nominal point and two
perturbations per selected coefficient. A full Cartesian design is available
only when requested explicitly because its run count grows exponentially.
No values from a manuscript table or figure are copied into these files.

## Stress and sensitivity runs

```bash
python -m experiments.rain_fade \
  --bandwidth-mhz 20,100 --attenuation-db 0,5,10

python -m experiments.lcb_variance_sensitivity \
  --protocol-mode nominal-ground \
  --measurement-std-db 0,1,2,3,4 --staleness-steps 0,1,2

python -m experiments.lcb_variance_sensitivity \
  --protocol-mode joint-high-mobility \
  --measurement-std-db 0,1,2,3,4 --staleness-steps 0,1,2

python -m experiments.lcb_calibration \
  --checkpoint checkpoints/novanet_paper.pt \
  --samples 720 --bins 10 \
  --kappas 0,0.5,1,1.5,2,2.5,3 \
  --output-dir results/lcb/calibration

python -m experiments.aerial_doppler \
  --speeds-kmh 60,300 --headings-deg 0,90,180,270 \
  --aerial-speed-kmh 300 --aerial-altitude-m 1000

python -m experiments.blockage \
  --conditions 8:0.10,12:0.20

python -m experiments.freeze_sensitivity \
  --freeze-steps 0,1,2,3
```

The LCB runner writes Rate-Dwell, NovaNet without LCB, and NovaNet with the
fixed configured κ for each variance/staleness cell. Its two protocol modes
keep the nominal ground table separate from the 300-km/h joint sensitivity
grid. The calibration runner regenerates the canonical 3,600-sequence stream
and uses its final 20% validation partition. It excludes `h=0` (which consumes
the current measurement), and writes `raw.csv`, `binned.csv`, `coverage.csv`,
and `protocol.json`. Its separately named independent split is diagnostic
only. It refuses a stale TLE by default and does not copy plotted values from
Liu9. The channel simulator
uses `RealizedChannelTrace`: fading, measurement noise, Doppler error, and
blockage are keyed by UE/satellite/time rather than query order. CHO and HOF
replay uses the same realized source/target trace at 10-ms resolution.

`freeze_sensitivity` reports event-derived ping-pong, CHO, rate, HOF, and
outage metrics. It intentionally does not reconstruct the manuscript's
normalized reward-equivalent score without the original validation-objective
artifact and normalization rule.

## Multi-UE extension

The runner shares the preceding load snapshot, applies synchronous two-phase
association, enforces each UE's freeze counter, evaluates CHO on the common
10-ms event trace, and reruns the common capacity/admission/PF allocator at
each execution-start and completion boundary. A late target consumes no
data-plane capacity before completion, successful execution changes the
serving association only at completion, and cross-epoch execution remains
pending. Blocked and newly failed-transition epochs contribute zero service;
capacity released by a sub-threshold admission is redistributed. The
manuscript defines the dummy blocking cost
`c_blk` but does not report its value, so a manuscript-grade run must receive
the validation-selected value explicitly:

```bash
python -m experiments.multi_ue \
  --users 50,100,200 --blocking-cost VALUE_FROM_VALIDATION
```

`--diagnostic-no-blocking-cost` is available only to exercise the software;
it lets the dummy option enter after all real candidates are exhausted and
must not be used to regenerate the reported table.

The density run trains and evaluates all 27 density/cap/seed cells by default.
It saves full learning curves, best checkpoints, held-out performance, the
untruncated horizon-union size, effective candidate count, cap-activation rate,
and the persistent five-epoch convergence statistic:

```bash
python -m experiments.density_convergence \
  --densities 60,120,240 --candidate-caps 8,16,32 \
  --seeds 2025,2026,2027 --epochs 100 --test-users 60
```

## Ephemeris aging

The aging run needs matched, real TLE snapshots.  Supply every requested age;
the script verifies the selected satellite names/order, time grid, and median
epoch separation before running.  It uses the age-0 ephemeris as physical
truth and the specified snapshot only for candidate/TTL planning.

```bash
python -m experiments.ephemeris_aging \
  --ages-hours 0,24,72 \
  --planning-tle 0=/data/starlink_t0.tle \
  --planning-tle 24=/data/starlink_tminus24h.tle \
  --planning-tle 72=/data/starlink_tminus72h.tle
```

Timestamp-shifting one TLE file is not equivalent to aging and is rejected as
an evidence source.  The exact snapshots and paper checkpoint are external
artifacts that still need to be released.

## Trace-based latency and offline references

```bash
python -m experiments.latency_from_traces \
  --trace-dir traces/paper --output results/latency/summary.csv
```

Each `METHOD.csv` requires `time_s,rate_bps`; an optional
`METHOD.blackouts.csv` supplies `start_s,end_s`. The finite FIFO, Poisson
arrivals, serialization, fixed network delay, and protocol processing are then
replayed from those inputs. A rate trace must start at zero; its final value is
held through the configured observation endpoint. The runner writes a sibling
`summary_protocol.json` containing the SHA-256 and row count of every rate and
blackout input. The author-side method traces are not checked in, so the
manuscript latency quantiles cannot yet be regenerated exactly.

The non-causal offline oracle is generated by the nominal evaluator and its
gap is computed after globally aggregating target-selection cost. The only
remaining score-normalization artifact is the separate normalized
reward-equivalent quantity reported in the freeze sensitivity paragraph; that
quantity is not inferred from printed values.
