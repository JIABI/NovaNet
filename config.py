"""Backward-compatible access to the single canonical paper configuration.

New code should import :func:`novanet.config.load_config`.  The constants below
exist only so the historical top-level entry points keep working while reading
exactly the same values as ``configs/paper.yaml``.
"""

from novanet.config import load_config


CFG = load_config()

# Geometry and simulation.
RNG_SEED = CFG.experiment.seed
TLE_PATH = CFG.experiment.tle_path
EARTH_RADIUS_M = 6_371_000.0
ELEV_MIN_DEG = CFG.experiment.minimum_elevation_deg
SIM_DURATION_S = CFG.experiment.duration_s
DT_S = CFG.experiment.decision_interval_s
SUBSAMPLE_S = CFG.experiment.geometry_subsample_s
TOP_K = CFG.experiment.candidate_cap
LIMIT_SATS = CFG.experiment.num_satellites
DELTA = CFG.experiment.decision_interval_s
NUM_SAMPLES = CFG.training.num_samples

# Link budget.
CARRIER_HZ = CFG.channel.carrier_hz
BANDWIDTH_HZ = CFG.channel.bandwidth_hz
BANDWIDTH_OPTIONS_HZ = CFG.channel.bandwidth_options_hz
EIRP_DENSITY_DBW_MHZ = CFG.channel.eirp_density_dbw_mhz
SAT_TX_POWER_DBM = CFG.channel.total_eirp_dbm
SAT_ANT_GAIN_DBI = 0.0  # EIRP already includes the transmit antenna pattern.
UE_ANT_GAIN_DBI = CFG.channel.ue_antenna_gain_dbi
NOISE_PSD_DBM_HZ = CFG.channel.noise_psd_dbm_hz
SYSTEM_NOISE_TEMPERATURE_K = CFG.channel.system_noise_temperature_k
PHY_EFFICIENCY = CFG.channel.implementation_efficiency
EFFICIENCY = PHY_EFFICIENCY
SMALL_SCALE_FADING_DB = 0.0  # Random fading is generated, not added as loss.
ATTEN_DB_PER_KM = 0.0  # Replaced by elevation-dependent gas/rain functions.
MIN_DATA_RATE_BPS = CFG.channel.minimum_rate_bps
MIN_SNR_DB = CFG.channel.outage_threshold_db

# CHO and decision controls.
TTT_SEC = CFG.handover.ttt_s
TTT_MS = 1e3 * CFG.handover.ttt_s
HO_DELAY_MS = 1e3 * CFG.handover.execution_s
CHO_HYS_DB = CFG.handover.hysteresis_db
HOM_DB = CFG.handover.hysteresis_db
EXTRA_MARGIN_DB = 0.0
FREEZE_S = CFG.handover.freeze_steps * CFG.experiment.decision_interval_s
W_STAT_S = CFG.handover.statistics_window_s
CHO_MIN_SNR_DB = CFG.channel.outage_threshold_db
HYS_MARGIN = CFG.handover.hysteresis_db

# Traffic and latency.
PKT_SIZE_BITS = 8 * CFG.traffic.packet_size_bytes
PROC_DELAY_MS = CFG.traffic.protocol_processing_ms
QUEUE_DELAY_MS = 0.0  # Produced by the configured FCFS queue, never hard-coded.

# Model and training.
F_UE = CFG.model.ue_feature_dim
F_SAT = CFG.model.node_feature_dim
F_EDGE = CFG.model.transition_feature_dim
HIDDEN = CFG.model.hidden_dim
GNN_LAYERS = CFG.model.gnn_layers
GRAPH_TOPK = CFG.model.graph_neighbors
ADJ_TAU = CFG.model.adjacency_temperature
DP_HORIZON_STEPS = CFG.planner.horizon_steps
DP_TEMPERATURE = CFG.planner.temperature
DP_SWITCH_COST = CFG.planner.base_switch_cost
DP_KAPPA = DP_SWITCH_COST
E_KAPPA_UNCERT = CFG.planner.lcb_kappa
ENERGY_USE_LCB = True
E_ALPHA_SNR = CFG.planner.rate_weight
E_BETA_TTL = CFG.planner.dwell_weight
E_BETA2_TTL = 0.0
E_GAMMA_SWITCH = CFG.planner.base_switch_cost
E_BETA_VELOCITY = CFG.planner.angular_speed_weight
E_BETA_HOF = CFG.planner.hof_weight
TTL_SCALE = 1.0  # TTL is z-scored with frozen training statistics.

EPOCHS = CFG.training.epochs
BATCH_SIZE = CFG.training.batch_size
NUM_WORKERS = CFG.training.num_workers
LR = CFG.training.learning_rate
WEIGHT_DECAY = CFG.training.weight_decay
GRAD_CLIP = CFG.training.gradient_clip
USE_AMP = CFG.training.use_amp
L_UNCERT_W = CFG.training.snr_nll_weight
L_TTL_W = CFG.training.ttl_weight
L_HOF_W = CFG.training.hof_weight
L_PATH_W = CFG.training.path_weight
L_SEL_W = CFG.training.selection_weight
L_ENT_W = CFG.training.entropy_weight
L_ENERGY_SPARSE_W = 0.0
STAY_W = CFG.training.handover_weight_init
TARGET_SWITCH_RATE = CFG.training.target_switch_rate
LAMBDA_INIT = CFG.training.handover_weight_init
LAMBDA_RHO = CFG.training.dual_step
LAMBDA_MAX = CFG.training.handover_weight_max

MODEL_CKPT = CFG.training.checkpoint_path

