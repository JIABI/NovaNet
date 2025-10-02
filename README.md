# NovaNet: Orbit-Aware Energy-based Spatio-Temporal PCGNN for LEO Satellite Handover

This repository provides the official PyTorch implementation of **NovaNet (OA-EST-PCGNN)**, a learning-based handover (HO) strategy for Low Earth Orbit (LEO) satellite networks.  
NovaNet integrates **Point-Cloud-inspired dynamic graph construction** with **Spatio-Temporal Graph Neural Networks (GNNs)**, enhanced by **orbit-aware priors** and an **energy-based objective** to balance throughput and HO frequency.

## Background

- Large-scale LEO constellations (e.g., Starlink) operate at ~400–600 km altitude.  
- Each satellite is visible to a ground User Equipment (UE) for only a few minutes.  
- Frequent handovers (HOs) are required to maintain connectivity.  

**Challenges:**
- Frequent HOs → high signaling overhead & service interruption.  
- Traditional heuristics (Max-Elevation, Max-ServeTime) are greedy and cause ping-pong.  
- Learning-based methods (e.g., DQN+GNN) improve stability but often sacrifice throughput.  

## Key Idea

**NovaNet** = **Dynamic Point-Cloud Graph Construction + Orbit-Aware Spatio-Temporal GNN + Energy-based Objective**

- **Point-Cloud Graph Builder** (`PCGraphBuilder`):  
  Treats candidate satellites as a small point cloud. Dynamically learns adjacency (UE–Sat and Sat–Sat) instead of fixed KNN/threshold rules.

- **Spatio-Temporal GNN**:  
  Models interactions across satellites and UE over time, predicting:  
  - best satellite (classification)  
  - link SNR and TTL (regression)  
  - HO probability and stay decision (binary)

- **Energy-based Objective**:  
  Unified loss combines throughput maximization and HO penalty.

- **Orbit-aware priors**:  
  Uses orbital dynamics (ephemeris, elevation trends) to regularize predictions.

## Repository Structure

```
├── config.py              # Experiment configs
├── train_oaest.py         # Training script
├── evaluate_oaest.py            # Evaluation
├── model_oaest.py         # Model definition
├── sim_dataset.py         # Satellite dataset generation
├── tle_ephem.py           # Ephemeris building from TLE
├── utils_geo.py           # Geometry utilities
├── viz_ephem.py           # Visualization of orbits
├── viz_snr_time.py        # SNR/time plotting
└── checkpoints/           # Pretrained models
```

## Dependencies

- Python 3.9+  
- PyTorch >= 1.12  
- NumPy, Matplotlib  
- sgp4 (for orbital propagation)  
- STK (optional)

Install via:
```bash
pip install -r requirements.txt
```

## Training

```bash
python train_oaest.py --ckpt checkpoints/nova_best.pt
```

Main parameters in `config.py`:
- `DT_S` : control interval  
- `TOP_K` : number of candidate satellites  
- `FREEZE_S` : freeze window  
- `BANDWIDTH_HZ` : bandwidth  
- `CARRIER_HZ` : carrier frequency  

## Evaluation

```bash
python evaluate.py --ckpt checkpoints/nova_best.pt
```

Metrics:
- HO Frequency  
- Throughput  
- HOF rate  
- Outage probability  
- Latency CDF  

## Citation

```bibtex
@article{NovaNet2025,
  title={NovaNet: Orbit-Aware Energy-based Spatio-Temporal PCGNN for LEO Satellite Handover},
  author={Jia Bi, Haochen Liu, Ting Liu},
  journal={IEEE TWC (Under Review)},
  year={2025}
}
```

