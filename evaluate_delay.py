import numpy as np
import matplotlib.pyplot as plt

# =========================
# 可配置参数（按需修改）
# =========================
SEED = 2025
WINDOW_S = 2400  # 评估窗口
N_PACKETS = 20000  # CDF采样包数（越大越平滑）
# CHO / HO 执行窗口
TTT_MS = 100.0
EXEC_MS = 150.0
HO_EXTRA_JITTER_MS = 10.0  # 执行窗口内附加轻尾抖动
X_MAX_MS = 250.0  # CDF x轴上限
# 基础E2E时延（不在切换窗口时）
BASE_LATENCY_MEAN_MS = 35.0
BASE_LATENCY_STD_MS = 6.0

# 20 MHz, 12 GHz 下你当前实验/估计的吞吐（用于Transmission Delay模拟）
# 注意：Transmission Delay不计HO执行开销；这里的吞吐仅用来拟合SNR/效率均值
THROUGHPUT_20MHZ = {
    "Max-Elevation": 68.39,
    "Max-ServeTime": 62.07,
    "GNN-only": 55.00,
    "DQN+GNN": 60.00,
    "NovaNet (Ours)": 62.19,
}

# 2400 s 内的HO次数（来自你的实验/估计）
HO_PER_2400S = {
    "Max-Elevation": 16.5,
    "Max-ServeTime": 10.5,
    "GNN-only": 12.0,
    "DQN+GNN": 6.8,
    "NovaNet (Ours)": 3.1,
}

# 物理层/系统参数（Transmission Delay用）
BANDWIDTH_HZ = 20e6  # 20 MHz
EFF_IMPL = 0.8  # 实现效率η
PKT_BITS = 12000  # 1500B
TTI_MS = 1.0  # 调度周期(可用于排队/离散时间约束)
SNR_STD_DB = 3.0  # 为了生成SNR序列的标准差（可调）

# =========================
# 工具函数
# =========================
rng = np.random.default_rng(SEED)


def truncated_normal(mean, std, size):
    x = rng.normal(mean, std, size)
    return np.clip(x, 0.0, None)


def draw_ho_windows(n_ho, win_ms, window_s):
    starts_s = rng.uniform(0.0, window_s, int(round(n_ho)))
    starts_ms = np.sort(starts_s * 1000.0)
    ends_ms = starts_ms + win_ms
    return starts_ms, ends_ms


def in_any_window(t_ms, starts_ms, ends_ms):
    if len(starts_ms) == 0:
        return np.zeros_like(t_ms, dtype=bool)
    mask = np.zeros_like(t_ms, dtype=bool)
    for s, e in zip(starts_ms, ends_ms):
        mask |= ((t_ms >= s) & (t_ms < e))
    return mask


def simulate_latency_ecdf(n_packets, window_s, n_ho, base_mu, base_std,
                          ho_exec_ms, ho_extra_ms, x_max_ms=250.0):
    pkt_t_ms = rng.uniform(0.0, window_s, n_packets) * 1000.0
    base_lat = truncated_normal(base_mu, base_std, n_packets)

    starts_ms, ends_ms = draw_ho_windows(n_ho, ho_exec_ms, window_s)
    hit = in_any_window(pkt_t_ms, starts_ms, ends_ms)
    extra = np.zeros(n_packets, dtype=np.float64)
    n_hit = hit.sum()
    if n_hit > 0:
        # 执行期额外时延：执行时间 + 轻尾指数抖动
        extra[hit] = EXEC_MS + rng.exponential(ho_extra_ms, n_hit)

    total = base_lat + extra
    xs = np.sort(total)
    ys = np.linspace(0, 1, len(xs), endpoint=True)
    # 为便于比较，裁剪x范围
    if x_max_ms is not None:
        # 仅用于显示；返回全量以便统计
        pass
    return xs, ys, n_hit / n_packets, total


# SNR(dB) -> bps/Hz（阶跃近似；或用香农）
def snr_to_eff_bpshz(snr_db):
    # 一个简单的MCS台阶，近似常见范围
    edges = np.array([-5, 0, 5, 10, 15, 20, 25, 30], dtype=float)
    effs = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 5.5], dtype=float)
    idx = np.searchsorted(edges, snr_db, side='right') - 1
    idx = np.clip(idx, 0, len(effs) - 1)
    return effs[idx]


# 由"目标吞吐(20MHz)"反推名义SNR均值（用香农近似），用于生成SNR序列
def mean_snr_db_from_throughput(thr_mbps, bw_hz=BANDWIDTH_HZ, eta=EFF_IMPL):
    se = (thr_mbps * 1e6) / (eta * bw_hz)  # bps/Hz
    # 香农：se = log2(1+SNR) => SNR = 2^se - 1
    snr_lin = np.maximum(2.0 ** se - 1.0, 1e-6)
    snr_db = 10.0 * np.log10(snr_lin)
    return float(snr_db)


def generate_snr_series(mean_db, std_db=SNR_STD_DB, n=N_PACKETS):
    return rng.normal(mean_db, std_db, n)


def transmission_delay_ms_from_snr(snr_db_series, bw_hz=BANDWIDTH_HZ,
                                   pkt_bits=PKT_BITS, tti_ms=TTI_MS):
    # 速率 = eff(bps/Hz)*BW；序列化时延 = L/R
    eff = snr_to_eff_bpshz(np.array(snr_db_series))
    phy_rate = np.maximum(eff * bw_hz, 1e3)  # bps, 下限防nan
    ser_ms = (pkt_bits / phy_rate) * 1e3
    # 简单排队（截断正态），不考虑HO执行
    q_ms = np.clip(rng.normal(0.5, 0.3, size=len(ser_ms)), 0.0, None)
    return ser_ms + q_ms


# =========================
# 1) 画 E2E Latency CDF
# =========================
def plot_latency_cdf():
    ho_exec_ms = TTT_MS + EXEC_MS
    methods = list(HO_PER_2400S.keys())
    colors = ["C2", "C1", "C0", "C3", "C4"]  # 仅为区分；可改

    plt.figure(figsize=(7.6, 5.2), dpi=300)
    hit_ratio_map = {}
    stats_map = {}

    for m, c in zip(methods, colors):
        xs, ys, hit_ratio, total = simulate_latency_ecdf(
            N_PACKETS, WINDOW_S, HO_PER_2400S[m],
            BASE_LATENCY_MEAN_MS, BASE_LATENCY_STD_MS,
            ho_exec_ms, HO_EXTRA_JITTER_MS, x_max_ms=X_MAX_MS
        )
        hit_ratio_map[m] = hit_ratio
        # 统计P50/P95/P99
        p50 = np.percentile(total, 50)
        p95 = np.percentile(total, 95)
        p99 = np.percentile(total, 99)
        stats_map[m] = (p50, p95, p99)

        plt.plot(xs, ys, label=m, linewidth=2.0)

    plt.xlim(0, X_MAX_MS)
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.55)
    plt.xlabel("End-to-End Latency (ms)")
    plt.ylabel("CDF")
    plt.title("Latency CDF with CHO Execution Windows (TTT + Exec)")
    plt.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig("latency_cdf.pdf")
    print("[Saved] latency_cdf.pdf")

    # 控制台打印摘要
    print("\n[Latency CDF Summary @ 20 MHz / clear-sky]")
    print("Method                P50(ms)   P95(ms)   P99(ms)   CHO-hit%")
    for m in methods:
        p50, p95, p99 = stats_map[m]
        print(f"{m:18s}  {p50:8.2f}  {p95:8.2f}  {p99:8.2f}   {100 * hit_ratio_map[m]:7.3f}%")


# =========================
# 2) 画 Transmission Delay 柱形图（对标）
# =========================
def plot_tx_delay_bars():
    methods = list(THROUGHPUT_20MHZ.keys())
    means = []
    stds = []

    for m in methods:
        # 由吞吐反推SNR均值 -> 生成SNR序列 -> 计算TxDelay（不计HO执行）
        mu_snr_db = mean_snr_db_from_throughput(THROUGHPUT_20MHZ[m])
        snr_series = generate_snr_series(mu_snr_db, SNR_STD_DB, N_PACKETS)
        tx_ms = transmission_delay_ms_from_snr(snr_series)
        means.append(np.mean(tx_ms))
        stds.append(np.std(tx_ms))

    x = np.arange(len(methods))
    w = 0.6

    plt.figure(figsize=(7.2, 4.8), dpi=300)
    bars = plt.bar(x, means, yerr=stds, width=w, capsize=4)
    for bx, h in zip(bars, means):
        plt.text(bx.get_x() + bx.get_width() / 2, h + 0.05, f"{h:.2f}",
                 ha="center", va="bottom", fontsize=9)

    plt.xticks(x, methods, rotation=15)
    plt.ylabel("Transmission Delay (ms)")
    plt.title("Transmission Delay (Queueing + Serialization only)")
    plt.grid(axis="y", linestyle="--", alpha=0.45)
    plt.tight_layout()
    plt.savefig("tx_delay_bars.pdf")
    print("[Saved] tx_delay_bars.pdf")

    # 控制台打印摘要
    print("\n[Transmission Delay Summary (20 MHz, no HO exec)]")
    print("Method                mean(ms)   std(ms)")
    for m, mu, sd in zip(methods, means, stds):
        print(f"{m:18s}  {mu:9.3f}  {sd:8.3f}")


# =========================
# main
# =========================
if __name__ == "__main__":
    plot_latency_cdf()
    plot_tx_delay_bars()

    # 额外输出：按“HO窗口占比”的背靠背核对
    print("\n[Packets affected by CHO execution window (sanity check)]")
    for m, nho in HO_PER_2400S.items():
        ratio = (nho * (TTT_MS + EXEC_MS)) / (WINDOW_S * 1000.0)
        print(f"{m:18s}: {100 * ratio:6.3f}%  (approx)")

