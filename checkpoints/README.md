# Checkpoints

`novanet_paper.pt` is the expected NovaNet paper checkpoint name. The learned
comparison checkpoints are expected as `gnn_only_paper.pt`,
`dqn_gnn_paper.pt`, and `dho_paper.pt`. None is represented by a randomly
initialized or smoke-test model in this repository.

Run:

```bash
python train_oaest.py --config configs/paper.yaml \
  --output checkpoints/novanet_paper.pt
```

Each current checkpoint stores the model state, canonical configuration,
configuration fingerprint, exact TLE
SHA-256 and epoch metadata, epoch, and validation metrics. Configuration or TLE
mismatches are rejected at evaluation.

The source-conditioned GNN-only model and offline sequential DQN+GNN training
path are implemented in `novanet/baselines.py` and
`train_learned_baseline.py`:

```bash
python train_learned_baseline.py --kind gnn_only \
  --output checkpoints/gnn_only_paper.pt
python train_learned_baseline.py --kind dqn_gnn \
  --output checkpoints/dqn_gnn_paper.pt
```

The repository also contains a DHO-shaped diagnostic surrogate. It is not the
interaction-trained cited DHO protocol and cannot populate the manuscript row:

```bash
python train_learned_baseline.py --kind dho --allow-surrogate-dho \
  --output checkpoints/dho_surrogate_diagnostic.pt
```

These commands create new checkpoints from the supplied TLE/data. They do not
recover the author-trained paper checkpoints and must not be used to label a
diagnostic run as an exact table reproduction. Table-grade loading requires
explicit `training_protocol`, `source_fidelity`, `paper_table_eligible`, full
training/validation metadata, configuration, fingerprint, and TLE provenance.
The local DHO trainer writes `paper_table_eligible=false`, and the evaluator
rejects it unless diagnostic surrogate loading is explicitly enabled.

The current configuration is schema 5 with fingerprint
`d53396a525bf68cc`. `diagnostic_only.pt` predates the current 6D-current,
5D-future, freeze-aware architecture and uses the bundled date-mismatched TLE.
Its sidecar metadata carries
`artifact_status=legacy_diagnostic_incompatible` and
`usable_for_current_manuscript=false`; it is retained only to document an
earlier software smoke test and must not be loaded or cited as paper evidence.

The public release must upload all actual trained paper checkpoints (or publish
immutable download URLs and SHA-256 values here) before it claims that the
trained NovaNet and learned-baseline models are available. Exact evaluation
also needs the historical 2023 TLE and the matching held-out traces; checkpoint
files alone do not close those evidence gaps.
