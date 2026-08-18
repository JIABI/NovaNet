# NovaNet

This repository is the synchronized implementation target for **NovaNet:
Orbit-Aware Structured Decision Learning for Reliable Handover in LEO
Constellations**. The current code uses one configuration, one link/CHO
simulator, and one finite-horizon planner across training and evaluation.
It does not treat the numbers typeset in the manuscript as generated results.

## Reproducibility status

The method and experiment entry points are present, but exact regeneration of
the manuscript tables still needs author-side scientific artifacts and one
validation-selected constant:

1. the historical Starlink TLE snapshot used at
   `2023-05-01T09:30:00Z`; and
2. the trained `checkpoints/novanet_paper.pt`, the separately trained GNN-only
   and DQN+GNN checkpoints, and a source-verified implementation/checkpoint for
   the cited DHO protocol;
3. the per-method rate/packet/blackout traces behind the originally reported
   latency distributions (new runs generate their own trace artifacts);
4. the validation-selected multi-UE dummy cost `c_blk`; and
5. the exact normalization artifact used for the reported freeze-sensitivity
   reward-equivalent score.

The `starlink.tle` file inherited from the earlier repository has a median epoch
of 2025-08-31, 852.7 days after the paper start time. The code rejects that
combination by default instead of silently propagating a stale TLE and
presenting the output as a reproduction. `--allow-stale-tle` exists only for
software diagnostics.

No random or smoke-test model is labeled as a paper checkpoint, and missing
learned-baseline rows are not replaced by NovaNet ablations. The offline
oracle is generated internally by non-causal event replay. See
`checkpoints/README.md`. The issue-by-issue implementation and verification
record is in [`CODE_PAPER_ALIGNMENT.md`](CODE_PAPER_ALIGNMENT.md).

## Canonical paper configuration

Every entry point reads [`configs/paper.yaml`](configs/paper.yaml). It currently
uses schema 5 and has fingerprint `d53396a525bf68cc`; the runtime value is also
printed by `python scripts/check_consistency.py`. Important values are:

| Item | Value |
|---|---:|
| elevation mask | 10 deg |
| candidate set | top 8 |
| TLE subset | shell-stratified; deterministic nested balancing in RAAN/mean anomaly |
| evaluation horizon | 2400 s |
| decision / geometry interval | 30 s / 5 s |
| CHO TTT / execution | 100 ms / 150 ms |
| freeze window | 1 decision step (30 s) |
| traffic / packet | Poisson 500 packets/s / 1500 bytes |
| FIFO / fixed delays | 4096 packets / 33.5 ms network + 1 ms protocol |
| bandwidth | 20 / 100 MHz |
| carrier | 12 GHz |
| planning horizon | 6 decisions (180 s) |
| current node features | 6 (elevation/rate, range/rate, direct TTL, causal SINR) |
| future recurrent fields | 5 (geometry/rates + direct TTL; no future measured SINR) |
| energy references | R_ref = 50 Mbps; T_ref = 600 s |
| energy coefficients | alpha = 1, beta = 0.35, c0 = 1, c1 = 0.5, c2 = 1.5 |
| pairwise CHO context | 5 (TTT, execution, hysteresis, threshold, failure fraction) |
| trainable parameters | 249,603 |
| multi-UE load coefficient | 0.5 (fixed prior-load penalty) |

`config.py` exposes the same values under legacy constant names. It is not a
second configuration.

## What changed from the earlier public code

- `novanet/soft_dp.py` implements the entropy-regularized finite-horizon
  recursion over the joint `(incumbent, freeze counter)` state. A still-visible
  incumbent is locked while its counter is positive; an unavailable incumbent
  releases the lock for recovery. The first action is conditioned on both the
  actual incumbent and actual counter. There is no `horizon=1` fallback in the
  paper model.
- `novanet/forecast.py` specifies deployment-time future inputs. SGP4 supplies
  future geometry and direct dwell time. The six-dimensional current node is
  used only at `h=0`; future recurrent inputs contain its first five
  geometry/dwell fields, with the same input-layer weights tied to the current
  projection. Deterministic future SINR is supplied separately to the residual
  and uncertainty heads. Future realized fading is never inserted into the
  deployed model input. A current no-coverage epoch is retained as zero
  service and outage rather than silently dropped.
- `novanet/model.py` contains a geometry-prior-gated learned adjacency, a
  satellite-ID-aligned recurrent state, log-SINR-residual mean/positive-scale
  heads, and a pairwise HOF head. The learned graph is evaluated from the
  current six-dimensional snapshot; the future five-dimensional rollout is
  recurrent. TTL is propagated directly by TLE/SGP4 and passed to the planner;
  there is no learned TTL head. The transition cost is exactly
  `1[i != j] (c0 + c1 Tbar_i + c2 pHOF_ij)`.
- `novanet/losses.py` implements the manuscript objective: residual Gaussian
  NLL, pairwise HOF BCE, sequence-policy KL, and explicit L2 regularization.
- Rate and TTL use the fixed configuration references `R_ref` and `T_ref`.
  Energy coefficients are fixed validation-selected configuration values,
  rather than trainable softplus weights or checkpoint-fitted z-scores.
- `novanet/channel.py` implements the 12 GHz link budget, SINR with optional
  exogenous interference and receive-array gain, the documented
  elevation-dependent gas/rain slant-path approximation, controlled excess
  attenuation, Rician/shadowing variation, Doppler, residual tracking error,
  and coherent-integration loss. `RealizedChannelTrace` keys fading,
  measurement noise, Doppler error, and blockage by UE, satellite, time, and
  random stream so replay does not depend on function-call order.
  `receive_array_gain_linear` evaluates the paper's local-frame
  orientation/array expression, and `Scenario.receive_gain_model` injects such
  a gain into every forecast, CHO, and service-link query. The reported
  experiments keep the configured single isotropic element, so this interface
  does not constitute a directional-array or pointing-error experiment.
- `novanet/handover.py` monitors the stored target over the decision interval,
  starts execution after the first continuously satisfied 100 ms TTT, and
  evaluates the following 150 ms execution window at 10 ms resolution.
  Training labels and evaluation attempts replay the same realized
  source/target channel trace; HOF-head labels are defined only for attempted
  executions.
- `novanet/latency.py` uses a documented Poisson source, 1500-byte packets,
  a finite drop-tail FIFO, work-conserving service, method-independent network
  delay, a shared 1 ms data-plane protocol component, and CHO service
  blackouts.
- `novanet/multi_ue.py` uses a fixed per-satellite capacity, admission cap,
  proportional-fair allocation, and synchronous two-phase association. The
  PF scheduler recycles capacity after a sub-threshold admission is removed.
  The experiment reruns that scheduler at each within-epoch CHO event boundary:
  the incumbent serves before execution, neither endpoint consumes service
  capacity during the blackout, and the target serves only after successful
  completion. Executions crossing a decision boundary remain pending.
- `novanet/baselines.py` and `train_learned_baseline.py` contain distinct,
  source-conditioned GNN-only and DQN+GNN models. DQN+GNN uses offline
  sequential fitted Q iteration with replay, TD targets, and a target network.
  The included DHO-shaped MLP is labeled as a surrogate because it does not
  reproduce the cited interaction-trained DHO protocol. It is rejected for
  manuscript-table evaluation by default. Paper checkpoints are not included.

Historical evaluation filenames remain as small aliases, so they can no longer
drift into separate parameter sets.

## Install and test

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest
python scripts/check_consistency.py
```

Built wheels include `configs/paper.yaml` and `starlink.tle`; the configuration
loader falls back to the wheel-installed `share/novanet` directory when no
source checkout is present. This packaging fallback does not change the TLE
date check described below.

The last command intentionally exits nonzero while the bundled 2025 TLE is
paired with the paper's 2023 start date.

## Train

First replace `starlink.tle` with the exact snapshot used by the paper, then:

```bash
python train_oaest.py \
  --config configs/paper.yaml \
  --output checkpoints/novanet_paper.pt
```

For a small software-only run with the currently bundled, date-mismatched TLE:

```bash
python train_oaest.py --samples 32 --epochs 1 --allow-stale-tle \
  --output checkpoints/diagnostic_only.pt
```

Do not compare diagnostic output with manuscript results.

## Evaluate

```bash
python -m experiments.evaluate_paper \
  --config configs/paper.yaml \
  --checkpoint checkpoints/novanet_paper.pt \
  --users 60
```

The evaluator uses a held-out layout/channel seed by default and emits
per-user and aggregate CSV files. Link rate, effective
throughput, HO, HOF, outage, transmission-only latency, end-to-end quantiles,
CHO occupancy, realized target-selection cost, and the globally aggregated
oracle gap are obtained in one run from the same events.

Deterministic baselines can be evaluated without learned checkpoints:

```bash
python -m experiments.evaluate_paper --baselines-only --allow-stale-tle
```

Learned comparison rows require their own checkpoints, for example
`--gnn-checkpoint checkpoints/gnn_only_paper.pt`. The evaluator emits the
offline oracle by non-causally replaying the same realized future link and
pairwise CHO events with the same energy and freeze constraints. It is labeled
as a reference and is never used as a causal deployment policy.

Every accepted checkpoint records the complete configuration fingerprint,
the exact TLE SHA-256/epoch/subset metadata, and validation metrics.
Evaluation rejects a checkpoint trained
on a different configuration or TLE.

The source-conditioned GNN-only and offline DQN+GNN baselines can be trained
with:

```bash
python train_learned_baseline.py --kind gnn_only
python train_learned_baseline.py --kind dqn_gnn
```

The local DHO-shaped network is available only as an explicitly acknowledged
diagnostic surrogate:

```bash
python train_learned_baseline.py --kind dho --allow-surrogate-dho \
  --output checkpoints/dho_surrogate_diagnostic.pt
```

Its checkpoint has `paper_table_eligible=false`; the default evaluator rejects
it. `--allow-surrogate-baselines` permits it only in a diagnostic run. A cited
DHO table row requires a separately verified implementation/checkpoint of that
protocol. Every accepted table-grade baseline checkpoint must include the
training protocol, source-fidelity qualification, sample/epoch/optimizer
metadata, validation metric, complete configuration, and TLE provenance.
DQN+GNN qualification additionally fixes the undiscounted Bellman target,
updates per epoch, target-network update interval, and validation-replay size;
overriding those settings automatically makes the checkpoint diagnostic.

## Reviewer-requested experiments

### Validation-only heuristic selection

```bash
python -m experiments.tune_heuristics --users-per-seed 20
```

Periodic-HO, Dwell-Aware, and Rate-Dwell candidates share the same held-out
validation layouts and channel seeds. Selection minimizes the aggregate
realized `mean_target_cost`; the reserved test seed is rejected during tuning.
The raw rows, summaries, selected settings, and complete protocol are written
under `results/heuristic_tuning/`.

### Density and convergence

The full factorial experiment uses deterministic nested, orbit-balanced TLE
subsets that preserve the source snapshot's inclination/mean-motion shell
mixture for 60/120/240 satellites and does not cap every density at top 8. It
records the full validation curve for K=8/16/32 and writes a separate
five-epoch-moving-average convergence summary:

```bash
python -m experiments.density_convergence \
  --densities 60,120,240 --candidate-caps 8,16,32 \
  --seeds 2025,2026,2027 --epochs 100
```

### Ablations

All tags in the manuscript are executable switches:

```bash
python -m experiments.ablation \
  --variants Full,OrbitPrior,DynAdj,Temporal,Planner,UncLCB,TransTTL,TransHOF
```

### Estimation-variance sensitivity

This varies measurement standard deviation and staleness while keeping the
fixed LCB coefficient unchanged. The first command reproduces the separate
ground/report-corruption protocol. The second is the reviewer-requested joint
high-mobility variance x LCB on/off sweep:

```bash
python -m experiments.lcb_variance_sensitivity \
  --protocol-mode nominal-ground \
  --measurement-std-db 0,1,2,3,4 --staleness-steps 0,1,2
python -m experiments.lcb_variance_sensitivity \
  --protocol-mode joint-high-mobility \
  --measurement-std-db 0,1,2,3,4 --staleness-steps 0,1,2
```

### Uncertainty calibration

The Appendix-A/Liu9 calibration regenerates the canonical 3,600-sequence
stream and uses sequences 2,880--3,599, the same final 20% partition used by
`train_oaest.py`:

```bash
python -m experiments.lcb_calibration \
  --checkpoint checkpoints/novanet_paper.pt \
  --samples 720 --bins 10 \
  --kappas 0,0.5,1,1.5,2,2.5,3 \
  --output-dir results/lcb/calibration
```

It writes row-level predictions, equal-count uncertainty bins, one-sided LCB
coverage, and a provenance-rich protocol file. Stale TLEs are rejected by
default. Exact paper calibration still requires the missing paper checkpoint
and historical TLE. A fast, deliberately non-paper smoke run is available as
`--split-source independent-diagnostic --samples 2`; its protocol is marked
diagnostic and its rows must not be used as manuscript results.

### 300 km/h aerial vehicle and Doppler

The defaults are 1 km altitude, 300 km/h, headings
0/90/180/270 deg, 12 GHz carrier, 99.5% Doppler compensation,
25 Hz estimation-error standard deviation, and 100 microsecond coherent
integration:

```bash
python -m experiments.aerial_doppler
```

All of those parameters are written into the output rows.

### Rain attenuation

The controlled excess-attenuation sweep uses the same channel implementation
and seeds for every policy:

```bash
python -m experiments.rain_fade --attenuation-db 0,5,10
```

### Multi-UE

The default region is a 500 km-radius disk centered at
51.5 deg N, 0 deg E. Each satellite has 1200 Mbps capacity and at most 32
admitted UEs; the network performs synchronous two-phase association and
proportional-fair sharing:

```bash
python -m experiments.multi_ue --users 50,100,200 \
  --blocking-cost VALUE_FROM_VALIDATION
```

The canonical config deliberately leaves `c_blk` unset because the manuscript
does not report its selected numeric value. `--diagnostic-no-blocking-cost`
only exercises the software path and is not a paper-reproduction setting.

### Latency

Packet latency is recomputed from explicit method traces:

```bash
python -m experiments.latency_from_traces --trace-dir traces/paper
```

Each `METHOD.csv` contains `time_s,rate_bps`; optional
`METHOD.blackouts.csv` contains `start_s,end_s`. The arrival process and service
rule are never inferred from the method name.

### Blockage, freeze, and ephemeris aging

```bash
python -m experiments.blockage --conditions 8:0.10,12:0.20
python -m experiments.freeze_sensitivity --freeze-steps 0,1,2,3
python -m experiments.ephemeris_aging --help
```

The aging runner requires independent, timestamp-matched TLE snapshots; it
rejects timestamp-shifted copies as evidence. Freeze sensitivity reports the
event-derived metrics but does not invent the unavailable normalization for
the manuscript's normalized reward-equivalent score.

## Parameter inference and Fig. 7

`results/reference/liu7.csv` exposes the five bar values as:

`queueing + serialization + shared 1.000 ms protocol processing`.

This gives 3.982, 9.201, 8.072, 6.943, and 4.521 ms in the plotted order.
The shared component is not method-specific tuning.

Run:

```bash
python -m experiments.calibrate_shared_parameters
```

The diagnostic shows that the five aggregate delays and five aggregate rates
do **not** uniquely support a common simple M/M/1 queue (current cross-method
RMSE is about 1.87 ms). Exact queue parameters therefore must come from packet
traces or the original scheduler; the code refuses to invent five separate
arrival rates merely to hit five bars.

## Inference benchmark

```bash
python scripts/benchmark_inference.py
```

The runner measures each of K=8/16/32 in a separate process and writes
`results/benchmark/inference_current.json`. It records the current 6D model's
calculated parameter count, latency distribution, exact parameter/input bytes,
worker-process peak RSS, and host/software metadata. Peak RSS is explicitly a
whole-worker high-water mark, not a model-only allocator estimate. The timed
scope contains the encoder, learned adjacency, heads, energy construction, and
Soft-DP; ephemeris propagation and feature construction are excluded. Random
weights are used only because inference cost depends on tensor shapes rather
than trained values. The earlier `results/benchmark/inference.json` is retained
and marked `legacy_incompatible` instead of being overwritten.

## Repository map

```text
configs/paper.yaml                 canonical parameters
novanet/config.py                  typed validation and fingerprint
novanet/ephemeris.py               TLE/SGP4 ECEF position + velocity
novanet/geometry.py                WGS-84 UE motion, visibility, robust TTL
novanet/channel.py                 link budget, fading, Doppler, reports
novanet/forecast.py                causal H-step candidate/energy inputs
novanet/model.py                   encoder, HOF head, energy, Soft-DP
novanet/soft_dp.py                 finite-horizon differentiable recursion
novanet/losses.py                  implemented four-term training objective
novanet/baselines.py               separate learned comparison models
train_learned_baseline.py          learned-baseline training entry point
novanet/handover.py                event-level CHO and HOF labels
novanet/latency.py                 Poisson/FIFO packet simulation
novanet/multi_ue.py                capacity, admission, PF allocation
experiments/                       paper and reviewer-requested runs
tests/                             numerical and consistency tests
```

## Citation

Please use the final IEEE TWC bibliographic record after acceptance. Do not
cite an “under review” placeholder as a published article.
