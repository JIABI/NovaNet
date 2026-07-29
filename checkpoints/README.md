# Checkpoints

`novanet_paper.pt` is the expected paper checkpoint name. It is intentionally
not represented by a randomly initialized or smoke-test model.

Run:

```bash
python train_oaest.py --config configs/paper.yaml \
  --output checkpoints/novanet_paper.pt
```

Each checkpoint stores the model state, frozen training-set normalization
statistics, canonical configuration, configuration fingerprint, exact TLE
SHA-256 and epoch metadata, epoch, and validation metrics. Configuration or TLE
mismatches are rejected at evaluation.

The public release must upload the actual trained paper checkpoint (or publish
its immutable download URL and SHA-256 here) before the manuscript claims that
trained models are available.
