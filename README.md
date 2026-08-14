# NovaNet

This repository is the synchronized reference implementation of **NovaNet: Orbit-Aware
Structured Decision Learning for Reliable Handover in LEO Constellations**.
The current code uses one configuration, one link/CHO simulator, and one
finite-horizon planner across training and evaluation.

## Reproducibility status

The source implementation is complete, but two scientific artifacts must be
supplied before the exact manuscript numbers can be regenerated:

1. the historical Starlink TLE snapshot used at
   `2023-05-01T09:30:00Z`; and
2. the trained `checkpoints/novanet_paper.pt` checkpoint (or its immutable
   download URL and SHA-256).

The `starlink.tle` file inherited from the earlier repository has a median epoch
of 2025-08-31, 852.7 days after the paper start time. The code rejects that
combination by default instead of silently propagating a stale TLE and
presenting the output as a reproduction. `--allow-stale-tle` exists only for
software diagnostics.

No random or smoke-test model is labeled as the paper checkpoint. See
`checkpoints/README.md`. The issue-by-issue implementation and verification
record is in [`CODE_PAPER_ALIGNMENT.md`](CODE_PAPER_ALIGNMENT.md).

## Canonical paper configuration

Every entry point reads [`configs/paper.yaml`](configs/paper.yaml). Important
values are:

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
| node features | 6 (elevation/rate, range/rate, direct TTL, causal SINR) |
| energy references | R_ref = 50 Mbps; T_ref = 600 s |
| energy coefficients | alpha = 1, beta = 0.35, c0 = 1, c1 = 0.5, c2 = 1.5 |
| pairwise CHO context | 5 (TTT, execution, hysteresis, threshold, failure fraction) |
| trainable parameters | 249,603 |
| multi-UE load coefficient | 0.5 (frozen, shared after score standardization) |

`config.py` exposes the same values under legacy constant names. It is not a
second configuration.

## What changed from the earlier public code

- `novanet/soft_dp.py` implements a real entropy-regularized finite-horizon
  recursion over `[batch, horizon, candidates]` state costs and
  `[batch, horizon, source, target]` transition costs. The first action is
  conditioned on the actual incumbent. There is no `horizon=1` fallback in the
  paper model.
- `novanet/forecast.py` specifies deployment-time future inputs. SGP4 supplies
  future geometry; future channel features use the deterministic link budget
  plus a decaying residual from the latest measurement report. Predictive
  variance grows with report age and horizon. Future realized fading is never
  read by the deployed policy. If a sparse-density forecast contains a
  no-coverage future epoch, the controller uses a documented current-epoch
  rate/dwell fail-safe; a current no-coverage epoch is recorded as zero
  service and outage rather than silently dropped.
- `novanet/model.py` contains a geometry-prior-gated learned adjacency, a
  satellite-ID-aligned recurrent state, log-SINR-residual mean/variance heads,
  and a pairwise HOF head. TTL is propagated directly by TLE/SGP4 and passed to
  the planner; there is no learned TTL head. The transition cost is exactly
  `1[i != j] (c0 + c1 Tbar_i + c2 pHOF_ij)`.
- `novanet/losses.py` implements the manuscript objective: residual Gaussian
  NLL, pairwise HOF BCE, sequence-policy KL, and explicit L2 regularization.
- Rate and TTL use the fixed configuration references `R_ref` and `T_ref`.
  Energy coefficients are fixed validation-selected configuration values,
  rather than trainable softplus weights or checkpoint-fitted z-scores.
- `novanet/channel.py` implements the 12 GHz link budget, the documented
  elevation-dependent gas/rain slant-path approximation, controlled excess
  attenuation, Rician/shadowing variation, Doppler, residual tracking error,
  and coherent-integration loss.
- `novanet/handover.py` evaluates TTT and the 150 ms execution window at 10 ms
  resolution and produces the realized labels used by the HOF head.
- `novanet/latency.py` uses a documented Poisson source, 1500-byte packets,
  a finite drop-tail FIFO, work-conserving service, method-independent network
  delay, a shared 1 ms data-plane protocol component, and CHO service
  blackouts.
- `novanet/multi_ue.py` uses a fixed per-satellite capacity, admission cap,
  proportional-fair allocation, and synchronous two-phase association.

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

The evaluator emits per-user and aggregate CSV files. Link rate, effective
throughput, HO, HOF, outage, transmission-only latency, end-to-end quantiles,
and CHO occupancy are obtained in one run from the same events.

Every accepted checkpoint records the complete configuration fingerprint,
the exact TLE SHA-256/epoch/subset metadata, and validation metrics.
Evaluation rejects a checkpoint trained
on a different configuration or TLE.

## Reviewer-requested experiments

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
validation-selected LCB coefficient fixed:

```bash
python -m experiments.lcb_variance_sensitivity \
  --measurement-std-db 0,1,2,3,4 --staleness-steps 0,1,2
```

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
python -m experiments.multi_ue --users 50,100,200
```

### Latency

Packet latency is recomputed from explicit method traces:

```bash
python -m experiments.latency_from_traces --trace-dir traces/paper
```

Each `METHOD.csv` contains `time_s,rate_bps`; optional
`METHOD.blackouts.csv` contains `start_s,end_s`. The arrival process and service
rule are never inferred from the method name.

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

The checked-in JSON states the scope, host/software metadata, warm-up,
repetitions, parameter count, and K=8/16/32 latency. It measures the neural
encoder, learned adjacency, heads, and Soft-DP; ephemeris propagation and
feature construction are explicitly excluded.

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
novanet/losses.py                  complete training objective
novanet/handover.py                event-level CHO and HOF labels
novanet/latency.py                 Poisson/FIFO packet simulation
novanet/multi_ue.py                capacity, admission, PF allocation
experiments/                       paper and reviewer-requested runs
tests/                             numerical and consistency tests
```

## Citation

Please use the final IEEE TWC bibliographic record after acceptance. Do not
cite an “under review” placeholder as a published article.
