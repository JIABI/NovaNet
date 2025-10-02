# ===== 链路/几何相关 =====
SAT_TX_POWER_DBM = 40  # 卫星发射功率 [dBm]

CARRIER_HZ = 11.9e9  # 载波频率 [Hz]

SAT_ANT_GAIN_DBI = 35  # 卫星天线增益 [dBi]

ELEV_MIN_DEG = 25.0  # 最小仰角 [deg]

NOISE_PSD_DBM_HZ = -173  # 噪声功率谱密度 [dBm/Hz]

SMALL_SCALE_FADING_DB = 20  # 小尺度衰落 [dB]

ATTEN_DB_PER_KM = 0.05  # 大气衰减 [dB/km]

EARTH_RADIUS_M = 6371e3  # 地球半径 [m]

# ===== 切换判决相关 =====



# ===== 采样/仿真 =====

DT_S = 1  # 采样间隔 [s]

SIM_DURATION_S = 1800  # 仿真总时长 [s]

TOP_K = 5  # 候选卫星数

RNG_SEED = 42  # 随机种子

# ===== 数据规模/星数/前瞻秒数 =====

NUM_SAMPLES = 3600  # 数据集样本数（可调 9600）

LIMIT_SATS = 1000  # 限制卫星数量（调试用，200/500/1000/2000）

DELTA = 30  # 前瞻预测秒数 (t -> t+DELTA)

# ===== 训练参数 =====

SWITCH_PENALTY_W = 1.5

EPOCHS = 100

BATCH_SIZE = 64

NUM_WORKERS = 0

LR = 1e-3

WEIGHT_DECAY = 1e-4

MSE_W = 0.02

CE_W = 1.2

BCE_W = 0.5

GRAD_CLIP = 1.0

# —— 模型结构 ——
F_UE = 4
F_SAT = 6  # 若加入 d_elev_dt, d_range_dt 则设为 6
F_EDGE = 9  # 若 F_SAT=6 则 9
HIDDEN = 128
GNN_LAYERS = 2
TAU_TTL_STEPS = 2
TTL_W = 0.3
STAY_W = 0.7

# 图构建控制
GRAPH_TOPK = None  # 或者 5/8 做稀疏化
ADJ_TAU = 1.0  # >1 更平滑，<1 更尖锐

# ===== 文件路径 =====

TLE_PATH = "starlink.tle"  # 或 STK 导出的 CSV

LIGHT_SPEED_M_S = 299792458.0

# Handover delay per event (ms)
HO_SIGNALING_MS = 60.0  # 信令/切换执行
HO_SYNC_MS = 10.0  # 重同步
HO_BACKOFF_MS = 5.0  # 随机退避/控制开销
HO_DELAY_MS = HO_SIGNALING_MS + HO_SYNC_MS + HO_BACKOFF_MS

# Transmission delay model
PKT_SIZE_BITS = 12 * 1024 * 8  # 12 KB 包
PHY_EFFICIENCY = 0.75  # 编解码/MAC/调度效率
MIN_DATA_RATE_BPS = 1e5  # 极低SNR兜底
PROC_DELAY_MS = 0.3  # 编解码/协议栈处理
QUEUE_DELAY_MS = 0.5  # 队列平均

# ========== Anti-ping-pong / Training Stabilizers ==========
CONS_W = 0.10  # 时间一致性正则权重（0.05–0.20 常见）

# ========== Anti-ping-pong / Inference Stabilizers ==========
TTT_SEC = 5.0  # 条件需持续多少秒才触发切换
DELTA0_DB = 3.0  # 动态迟滞基础门槛 (dB)
ALPHA_DB_PER_DEG_S = 0.5  # 动态迟滞：仰角变化率差系数
BETA_DB = 2.0  # 动态迟滞：随 TTL 的系数 (1/TTL)
DP_KAPPA = 2.5  # DP 切换成本 (dB 等效)

# ==== Spatio-Temporal Graph (OA-PC) ====
DELTA_T_SEC = 30  # t - Δt 的时间间隔（与数据步长一致）
USE_TEMPORAL_EDGE = True
ORBIT_PRIOR_W = 1.0  # 轨道先验（轨道元素差）权重
GEOM_PRIOR_W = 1.0  # 几何核权重（Δx, Δv, elev等）
TIME_EDGE_GATE_INIT = 0.7  # 同ID跨帧时间边的初始 gate

# ==== Energy Head (E_i^t) ====
E_KAPPA_UNCERT = 1.0  # κ_u * σ（不确定性惩罚），若做uncertainty
ENERGY_USE_LCB = True  # SNR 采用 LCB = μ - κ_u*σ

# ==== Soft-DP (Differentiable Planning) ====
DP_HORIZON_STEPS = 6  # 窗口长度 H（训练时展开）
DP_SWITCH_COST = 2.5  # κ（边切换成本）
TEACHER_EVAL_DELTA = 30  # 教师路径生成时的步长（与数据一致）

# ==== Loss Weights (新增) ====

# —— 时空图 / 能量头 ——

E_ALPHA_SNR = 1.0  # SNR收益权重
E_BETA_TTL = 1.5  # “留存惩罚”权重（只对 i!=cur 生效）
E_BETA2_TTL = 0.5  # 候选TTL偏好权重（对所有候选）
TTL_SCALE = 8.0  # 留存惩罚时间尺度（单位=步）
E_GAMMA_SWITCH = 5.0  # 切换代价权重（从 2.5 提高到 4~6 抑制切换）

# —— Soft-DP（用于KL里的log_softmax温度）——

DP_TEMPERATURE = 1.0

# —— 训练与约束 ——

L_UNCERT_W = 0.2
L_TTL_W = 1.0
L_PATH_W = 0.5
L_ENERGY_SPARSE_W = 0.0
TARGET_SWITCH_RATE = 0.12  # 目标换星率（≈ 3~5 分钟/次）
LAMBDA_INIT = 3.0
LAMBDA_RHO = 0.05
LAMBDA_MAX = 30.0
HOM_DB = 1.0
EXTRA_MARGIN_DB = 0.0

# —— 评估期稳定策略 ——

HYS_MARGIN = 0.5  # 能量滞回（胜者需明显更好才切）
FREEZE_S = 90.0  # 切换后冻结时间（秒），确保 > step_s

# 模型权重

MODEL_CKPT = "checkpoints/oaest_best.pt"

# 吞吐计算的带宽（Hz）

BANDWIDTH_HZ = 400e6  # 20 MHz

# sp[..., F] 特征索引（请按你的数据构造实际填写）

ELEVATION_IDX = 0  # 假设仰角在第0维，如不同请改

TTL_IDX = 5  # 假设 TTL 在第5维，如不同请改

SNR_IDX = 6  # 若要在启发式里计算吞吐，建议把候选SNR(dB)也放入特征并给出索引

