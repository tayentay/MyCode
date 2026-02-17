import math
import numpy as np
from enum import Enum
from dataclasses import dataclass

@dataclass
class CongestionControl:
    """拥塞控制配置类"""
    name: str
    enum_value: int
    ns3_class: str

# 拥塞控制配置
CC_CONFIGS = {
    'QuicNewReno': CongestionControl('QuicNewReno', 0, "ns3::TcpNewReno"),
    'OLIA': CongestionControl('OLIA', 1, "ns3::MpQuicCongestionControl"), 
    'RateB': CongestionControl('RateB', 2, "ns3::RateBasedCongestionOps")
}

# ==================== 配置参数 ====================
@dataclass
class Config:
    myscheme = 'PPO'
    cpscheme = 'DAS'  # DAS, RR, Random
    cc = 'RateB'  # QuicNewReno, OLIA, RateB

    # RateB Hyperparameters
    rate_b_gamma: float = 0.5
    rate_b_rho: float = 50000.0
    rate_b_a: float = 10.0
    rate_b_b: float = 0.5
    rate_b_m_max: float = 2.0
    
    # Congestion Avoidance Hyperparameters
    ca_delta_r: float = 0.05    # RTT increase threshold (seconds)
    ca_d_th_1: float = 1000.0      # Distance threshold for distinguishing congestion vs wireless
    ca_d_th_2: float = 1100e3      # Distance threshold for distinguishing congestion vs wireless
    ca_w_prime: int = 10        # History window size for recovering cwnd
    ca_gamma: float = 0.8       # Reduction factor for wireless loss (Con3)
    ca_sigma: float = 0.5       # Weight factor for lambda calculation
    ca_snr_max: float = 30.0    # Max expected SNR (dB) for normalization

    # Preference Estimation Algorithm
    enable_preference_algo: bool = True
    preference_warmup_steps: int = 500  # Number of steps to train estimator before using it for decisions
    preference_lambda: float = 0.99
    preference_epsilon: float = 1e-2
    preference_buffer_capacity: int = 20000
    preference_min_buffer_size: int = 2000
    preference_fit_interval_steps: int = 100
    preference_explore_eps: float = 0.05
    preference_target_clip: float = 20.0
    preference_min_rtt: float = 1e-4

    seed = 1
    
    # G2A
    area_size: float = 1000.0
    num_ues: int = 9
    num_uavs: int = 4
    uav_height: float = 100.0
    uav_speed: float = 50.0
    time_slot_duration: float = 0.1
    simulation_time: float = 6.4
    frequency: float = 2.4e9
    tx_power: float = 0.1
    noise_figure: float = 7.0
    n_rbs: int = 2
    packet_size: int = 1024
    error_rate: float = 0.000001  # 0.000001
    RBs: bool = True  # Whether to Consider Resource Blocks in scheduling
    G2A_bandwidth: float = 1e6
    N = 1.9952623149688832e-18  # 高斯白噪声功率谱密度 W/Hz
    sub_bw = G2A_bandwidth / n_rbs
    scene: str = 'suburban'
    port = 9  # port for communication
    uav_radius: float = area_size * 0.4
    ue_positions = np.array([[0, 0, 0], [0, 100, 0], [100, 0, 0],
                       [100, 100, 0], [100, 200, 0], [200, 100, 0],
                       [200, 200, 0], [200, 300, 0], [300, 200, 0]])
    # ue_positions = np.array([[350, 350, 0], [350, 450, 0], [450, 350, 0],
    #                 [450, 450, 0], [450, 550, 0], [550, 450, 0],
    #                 [550, 550, 0], [550, 650, 0], [650, 550, 0]])
    def __post_init__(self):
        self.uav_angles = [2 * math.pi * uav_idx / self.num_uavs for uav_idx in range(self.num_uavs)]

    # A2S
    R_earth = 6371e3  # 地球半径m
    num_sats = 22  # 卫星数量
    sat_speed = 7.59e3  # 7.59 km/s
    min_elevation_deg = 30  # degree

    A2S_bandwidth: float = 1e7  # 载波带宽 10MHz -> Hz
    satellite_altitude: float = 550e3  # 卫星轨道高度 550km -> m
    G_T_db: float = 34  # 发射天线增益与噪声温度比 34dB/K
    lambda_rician_db: float = 8  # 小尺度衰落因子 dB
    d_s: float = 2.0  # 卫星天线阵列直径 m
    f_c_s: float = 20e9  # 卫星载波中心频率 Hz
    G_m_l_db: float = 15  # UAV发射天线增益 dB -> 线性值
    T_N: float = 290  # 噪声温度 K
    mu_r: float = -2.6  # 雨衰均值 dB
    sigma_r: float = 1.63  # 雨衰方差 dB
    h_LOS_power_min: float = 0.6  # LOS分量功率最小值
    h_LOS_power_max: float = 0.7  # LOS分量功率最大值
    v_NLOS_real: float = 0.2  # NLOS分量实部方差
    v_NLOS_imag: float = 0.25  # NLOS分量虚部方差

    use_self_predictive: bool = True
    sp_latent_dim: int = 128 # 128
    sp_hidden_dim: int = 128 # 256
    sp_model_hidden_dim: int = 128 # 256
    sp_model_layers: int = 2
    sp_tau: float = 0.005
    sp_coef: float = 1.0
    sp_lr: float = 1e-5
    noise_dim: int = 160
    d_loss_factor: float = 1e3
    b_scale: float = 3e2
    alpha: float = 0.8

    # Self-Predictive Attention 参数
    sp_attn_num_heads = 4        # 注意力头数（需整除 sp_latent_dim）
    sp_attn_num_layers = 1       # Transformer 层数
    sp_attn_dropout = 0.0        # Dropout
    # GradNorm asymmetry parameter (α)
    # α=0: enforce equal gradient norms regardless of training speed
    # α=1.0: standard GradNorm
    # α=1.5: stronger rebalancing towards slower tasks (recommended)
    gradnorm_alpha = 1
    # Learning rate for GradNorm weight parameters
    gradnorm_lr = 0.025
    
# class Config:
#     def __init__(self):
#         # 系统参数
#         self.T = 600  # 每轮时长
#         self.t = 1
#         self.clusters = 9  # clusters个数
#         self.areas = 6  # areas个数
#         self.groups = 9  # 每个地区groups个数
#         self.Wsu = 1000000  # 汇聚点到无人机的带宽
#         self.N = 1.9952623149688832e-17  # 高斯白噪声功率谱密度 W/Hz
#         self.G0 = [500, 500]  # 充电点所在位置
#         self.sdata = [000000000, 2400000000]  # 每轮汇聚点从感知点接收到的数据总量范围

#         # 汇聚点相关参数
#         self.nnum = self.groups  # 每个cluster的汇聚点总个数
#         self.ndots = np.array([[166, 165], [498, 165], [830, 165],
#                                [166, 498], [498, 498], [830, 498],
#                                [166, 830], [498, 830], [830, 830]])  # 九个汇聚点的初始位置
#         self.ndist = [-60, 60]  # 汇聚点的随机位置波动范围[-60,60]
#         self.ncapacity = 30  # 汇聚点最大电池容量，J
#         self.nharvest = [0, 24]  # 汇聚点时隙t内可获得电量，J
#         self.nPmax = 1  # 汇聚点最大发射功率，w
#         self.nplevel = 5  # 汇聚点电量等级划分，用于action，不包括0

#         # 无人机相关参数
#         self.unum = self.clusters  # 无人机总个数
#         self.uvelocity = 10  # 无人机飞行速度，m/s
#         self.ulgains = [1e-14, 1e-9]  # 无人机-卫星信道增益波动范围，[1e-14, 1e-9]
#         self.uvf = 10  # 无人机飞行速度，m/s
#         self.uph = 4  # 无人机悬停功率，w
#         self.upf = 2  # 无人机飞行功率，w
#         self.uPmax = 5  # 汇聚点最大发射功率，w
#         self.uBmax = 2000  # 无人机最大电池容量
#         self.uPmax = 1  # 无人机最大发射功率，w
#         self.uplevel = 5  # 汇聚点电量等级划分，用于action，不包括0

#         self.slgW = 1000000  # 单个子信道的带宽

#         # env2所需参数
#         self.uBDs = [000000000, 30000000]  # 每轮无人机从汇聚点接收到的数据总量范围，7000000000/600*2
#         self.ipy = 0.00  # imperfect SIC系数
#         self.uharvest = [0, 6.4]
#         self.ucapacity = 8
#         self.ufixp = 0.5  # 固定功率

#         self.area_sub_bands = [3,4,4,7,8,17]  # 按地区子信道数划分
#         self.UAV_sub_bands = [9,10,10,10,10,10,10,10,10]   # 按无人机子信道数划分
