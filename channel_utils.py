import numpy as np
import math
import cmath
from typing import List, Dict, Tuple
from config import Config
from scipy.special import j1, jn
from typing import Tuple

class G2AChannel(object):
    """Groud-to-Air (ATG) channel model"""

    chan_params = {
        'suburban': (4.88, 0.43, 0.1, 21),
        'urban': (9.61, 0.16, 1, 20),
        'dense-urban': (12.08, 0.11, 1.6, 23),
        'high-rise-urban': (27.23, 0.08, 2.3, 34)
    }

    def __init__(self, config):
        # Determine the scene-specific parameters.
        params = self.chan_params[config.scene]
        self.a, self.b = params[0], params[1]  # Constants for computing p_los
        self.eta_los, self.eta_nlos = params[2], params[3]  # Path loss exponents (LoS/NLoS)
        self.fc = config.frequency  # Central carrier frequency (Hz)

    def estimate_chan_gain(self, ue_positions, uav_positions):
        """Estimates the channel gain.
        returns: numpy array of shape (num_uavs, num_ues)
        """
        num_uavs = uav_positions.shape[0]
        num_ues = ue_positions.shape[0]
        
        # 扩展维度以便广播计算
        uav_expanded = uav_positions[:, np.newaxis, :]  # shape: (num_uavs, 1, 3)
        ue_expanded = ue_positions[np.newaxis, :, :]    # shape: (1, num_ues, 3)
        
        diff = uav_expanded - ue_expanded  # shape: (num_uavs, num_ues, 3)
        dist = np.sqrt(np.sum(diff**2, axis=2))  # shape: (num_uavs, num_ues)
        dist = np.maximum(dist, 1)  # 确保距离不小于1
        
        horizontal_diff = diff[:, :, :2]  # 只取x,y坐标
        horizontal_distances = np.sqrt(np.sum(horizontal_diff**2, axis=2))  # shape: (num_uavs, num_ues)
        
        height_diff = np.abs(diff[:, :, 2])  # shape: (num_uavs, num_ues)
        
        angles = np.degrees(np.arctan(height_diff / np.maximum(horizontal_distances, 1e-10)))
        
        # 处理水平距离为0的情况
        angles = np.where(horizontal_distances < 1e-10, 90.0, angles)

        # Estimate probability of LoS link emergence.
        p_los = 1 / (1 + self.a * np.exp(-self.b * (angles - self.a)))
        # Get direct link distance.
        # d = np.sqrt(np.square(d_level) + np.square(h_ubs))
        # Compute free space path loss (FSPL).
        fspl = (4 * math.pi * self.fc * dist / 3e8) ** 2
        # Path loss is the weighted average of LoS and NLoS cases.
        pl = p_los * fspl * 10 ** (self.eta_los / 20) + (1 - p_los) * fspl * 10 ** (self.eta_nlos / 20)
        return 1 / pl

class G2AScheduler:

    def __init__(self, config: Config, channel: G2AChannel):
        self.config = config
        self.channel = channel
        self.history:  List[Dict] = []
        self.RBs = self.config.RBs
        self.n_rbs = self.config.n_rbs
        if config.cpscheme == "DAS":
            self.scheduler = self.Distance_Aware_Scheduling
        elif config.cpscheme == "RR":
            self.scheduler = self.Round_Robin_Scheduling
        elif config.cpscheme == "Random":
            self.scheduler = self.Random_Scheduling
        else:
            self.scheduler = self.Distance_Aware_Scheduling


    def get_default_assignment(self, ue_positions, uav_positions):
        """
        分配最近的UAV给每个UE，考虑RBs限制。
        returns: assignment matrix of shape (num_uavs, num_ues) or (num_uavs, num_ues, n_rbs)
        """
        num_ues = ue_positions.shape[0]
        num_uavs = uav_positions.shape[0]
        
        # 计算所有UAV-UE对的欧几里得距离
        uav_expanded = uav_positions[:, np.newaxis, :]
        ue_expanded = ue_positions[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum((uav_expanded - ue_expanded) ** 2, axis=2))
        
        # 生成随机UE顺序
        ue_order = np.random.permutation(num_ues)
        assignment = np.zeros((num_uavs, num_ues), dtype=bool)
        uav_loads = np.zeros(num_uavs, dtype=int)  # 每个UAV当前服务的UE数
        
        for ue_id in ue_order:
            ue_distances = dist_matrix[:, ue_id]
            sorted_uavs = np.argsort(ue_distances)
            
            # 找到第一个有容量的UAV
            for uav_id in sorted_uavs:
                if uav_loads[uav_id] < self.n_rbs:
                    assignment[uav_id, ue_id] = True
                    uav_loads[uav_id] += 1
                    break
    
        return assignment
    
    def Distance_Aware_Scheduling(self, ue_positions, uav_positions):
        num_ues = ue_positions.shape[0]
        num_uavs = uav_positions.shape[0]
        
        # 计算所有UAV-UE对的欧几里得距离
        uav_expanded = uav_positions[:, np.newaxis, :]
        ue_expanded = ue_positions[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum((uav_expanded - ue_expanded) ** 2, axis=2))
        assignment = np.zeros((num_uavs, num_ues), dtype=bool)
        
        for ue_id in range(num_ues):
            ue_distances = dist_matrix[:, ue_id]
            nearest_uav = np.argmin(ue_distances)
            assignment[nearest_uav, ue_id] = True
        
        return assignment

    def Round_Robin_Scheduling(self, ue_positions, uav_positions):
        if self.history is not None:
            start_idx = len(self.history)
        else: start_idx = 0
        num_ues = ue_positions.shape[0]
        num_uavs = uav_positions.shape[0]
        print('------------------------------------')
        assignment = np.zeros((num_uavs, num_ues), dtype=bool)
        for ue_id in range(num_ues):
            uav_id = (start_idx + ue_id) % num_uavs
            assignment[uav_id, ue_id] = True
            print(f'uav_id: {uav_id}, ue_id: {ue_id}')
        return assignment 


    def Random_Scheduling(self, ue_positions, uav_positions):
        num_ues = ue_positions.shape[0]
        num_uavs = uav_positions.shape[0]
        assignment = np.zeros((num_uavs, num_ues), dtype=bool)
        for ue_id in range(num_ues):
            uav_id = np.random.randint(0, num_uavs)
            assignment[uav_id, ue_id] = True
        return assignment

    def get_rates(self, ue_positions, uav_positions, assignment, gain, time_slot: int, tx_power=None):
        num_ues = ue_positions.shape[0]
        num_uavs = uav_positions.shape[0]
        g = gain  # Channel gain
        rates = np.zeros(num_ues)
        if tx_power is None:
            tx_power = np.array(self.config.tx_power,dtype=float)
        
        if self.config.RBs:
            sche = np.zeros((num_uavs, num_ues, self.n_rbs), dtype=bool)
            uav_idxs, ue_idxs = np.where(assignment)
            rb_idxs = np.random.randint(0, self.n_rbs, size=len(uav_idxs))
            sche[uav_idxs, ue_idxs, rb_idxs] = True

            if np.ndim(tx_power) != 0:
                tx_power = tx_power[np.newaxis, :]  # (1, num_ues)
            rx_power_matrix = tx_power * g
            rx_power_matrix = rx_power_matrix[:, :, np.newaxis]  # (num_uavs, num_ues, 1)
            rx_power_matrix = np.broadcast_to(rx_power_matrix, (num_uavs, num_ues, self.n_rbs))  # (num_uavs, num_ues, n_rbs)
            ue_assigned_rb_mask = sche.sum(axis=0, dtype=bool)  # (num_ues, n_rbs)，某UE在某RB上是否被分配到
            interference_mask = np.broadcast_to(ue_assigned_rb_mask, (num_uavs, num_ues, self.n_rbs))  # (num_uavs, num_ues, n_rbs)
            p_rx = rx_power_matrix*sche
            all_rb = (interference_mask * rx_power_matrix).sum(axis=1)  #  (num_uavs, n_rbs)，每个UAV在某RB上接收到的所有信号
            snr = p_rx/(all_rb[:,np.newaxis,:]-p_rx + self.config.sub_bw * self.config.N)
            rates = self.config.sub_bw*np.log2(1+snr).sum(axis=(0,2))
        else:
            if np.ndim(tx_power) != 0:
                tx_power = tx_power[np.newaxis, :]  # (1, num_ues)
            rx_power_matrix = tx_power * g
            # 确定每个UE是否被分配
            ue_assigned = assignment.sum(axis=0, dtype=bool)
            # 每个UE接入的UAV索引
            ue_to_uav = assignment.T.argmax(axis=1)  # 包括了未被分配的UE
            # 计算每个UAV的干扰（来自其他UAV的UE）
            # 创建UE到UAV的信号掩码
            signal_mask = np.broadcast_to(ue_assigned, (num_uavs, num_ues))
            # 创建UE干扰掩码矩阵
            interference_mask = ue_to_uav != np.arange(num_uavs).reshape(-1, 1)
            # 计算接收功率矩阵（考虑发射功率）
            # 如果tx_power是标量，则所有UE使用相同发射功率
            # 如果tx_power是数组，则每个UE使用其对应的发射功率
            # 计算每个UAV受到的总干扰功率
            interference_per_uav = (rx_power_matrix * interference_mask * signal_mask).sum(axis=1)
            # 计算每个被分配UE的信号功率和干扰功率
            # 获取被分配UE的索引
            # assigned_indices = np.flatnonzero(ue_assigned)
            # 获取每个被分配UE接入的UAV索引
            # assigned_uavs = ue_to_uav[assigned_indices]
            # 计算信号功率：UE的发射功率 * 到其接入UAV的信道增益
            signal_power = (rx_power_matrix*assignment).sum(axis=0)
            # 计算干扰功率：UE所在UAV受到的总干扰
            interference_power = interference_per_uav[ue_to_uav]
            # 计算SINR和速率
            snr = signal_power / (interference_power + self.config.sub_bw * self.config.N)
            rates = self.config.sub_bw*np.log2(1 + snr)

        self.history.append({
            'time_slot': time_slot,
            'assignment': assignment.copy(),
            'rates': rates
        })
        return rates

class LEOSatelliteMobility:

    def __init__(self, config):
        """初始化卫星移动模型并预先计算常数项"""
        self.R_earth = config.R_earth/1000
        self.h = config.satellite_altitude/1000
        self.r_sat = self.R_earth + self.h 
        self.speed = config.sat_speed/1000
        self.min_elevation = math.radians(config.min_elevation_deg)
        
        # 预先计算常数项
        self._precompute_constants()
        
    def _precompute_constants(self):
        """预先计算卫星移动相关的常数项"""
        # 1. 计算覆盖边缘时的地心角 theta_max
        cos_el = math.cos(self.min_elevation)
        beta = math.asin((self.R_earth / self.r_sat) * cos_el)
        self.theta_max = (math.pi / 2.0) - self.min_elevation - beta
        
        # 2. 计算角速度 (弧度/秒)
        self.angular_velocity = self.speed / self.r_sat
        
        # 3. 计算服务总时长
        self.total_angle = 2 * self.theta_max
        self.service_duration = self.total_angle / self.angular_velocity
        
        # 4. 预先计算余弦定理中的常数项
        self.r_sat_sq = self.r_sat**2
        self.R_earth_sq = self.R_earth**2
        self.two_R_r = 2 * self.R_earth * self.r_sat
        
    def get_distance_and_elevation(self, t):
        """获取t时刻卫星的位置和距离（使用预计算常数）"""
        # if t < 0 or t > self.service_duration:
        #     return float('inf'), -90.0, False
        
        # 计算当前地心角
        angle_traveled = self.angular_velocity * t
        theta_current = -self.theta_max + angle_traveled
        
        # 计算直线距离（使用预计算常数）
        cos_theta = math.cos(theta_current)
        dist_sq = self.R_earth_sq + self.r_sat_sq - self.two_R_r * cos_theta
        distance = math.sqrt(dist_sq)
        
        # 计算仰角
        sin_el = (self.r_sat * cos_theta - self.R_earth) / distance
        sin_el = max(-1.0, min(1.0, sin_el))
        elevation = math.degrees(math.asin(sin_el))
        
        return distance, elevation
    
class A2SChannel:
    def __init__(self, config=None):
        """初始化参数并预先计算常数项"""
        # 基础常数
        self.c = 3e8  # 光速 m/s
        self.kappa = 1.38e-23  # 玻尔兹曼常数 J/K
        self.cfg = config
        self._precompute_constants()
    
    def _precompute_constants(self):
        """预先计算所有常数项"""
        cfg = self.cfg
        
        # 1. 转换为线性值
        self.G_T = 10**(cfg.G_T_db/10)  # 发射天线增益与噪声温度比
        self.lambda_rician = 10**(cfg.lambda_rician_db/10)  # Rician因子
        self.G_m_l = 10**(cfg.G_m_l_db/10)  # UAV发射天线增益
        
        # 2. 预先计算公式(14)中的常数项
        self.const_part_g = (self.c / (4 * np.pi * cfg.f_c_s * cfg.satellite_altitude))**2
        self.const_part_g *= self.G_T / (self.kappa * cfg.A2S_bandwidth)
        
        # 3. 噪声功率常数
        self.N0 = self.kappa * cfg.T_N  # 噪声功率谱密度 (W/Hz)
        self.noise_power_const = self.N0 * cfg.A2S_bandwidth  # 总噪声功率常数
        
        # 4. 雨衰参数
        self.mu_r = cfg.mu_r
        self.sigma_r = np.sqrt(cfg.sigma_r)
        
        # 5. LOS/NLOS参数
        self.h_LOS_power_min = cfg.h_LOS_power_min
        self.h_LOS_power_max = cfg.h_LOS_power_max
        self.v_NLOS_real = cfg.v_NLOS_real
        self.v_NLOS_imag = cfg.v_NLOS_imag
        
        # 6. 其他配置参数
        self.W_AS = cfg.A2S_bandwidth
        self.d_s = cfg.d_s
        self.f_c_s = cfg.f_c_s
        self.T_N = cfg.T_N
        
        # 7. 贝塞尔函数相关的预计算系数
        self.bessel_coeff = np.pi * self.d_s * self.f_c_s / self.c
    
    def calculate_omega(self, theta: float) -> complex:
        """计算卫星接收天线增益ω (公式12和13)"""
        phi = self.bessel_coeff * np.sin(theta)
        
        # 避免除零错误
        if np.abs(phi) < 1e-12:
            phi = 1e-12
        
        # 计算ω 公式(12)
        term1 = j1(phi) / (2 * phi)
        term2 = 36 * jn(3, phi) / (phi**3)
        
        return complex(term1 + term2)
    
    def calculate_channel_gain(self, distance: float, theta: float, generate_new_fading: bool = False):
        """计算完整的信道增益 公式(11)
        
        Args:
            distance: UAV到卫星距离 (m)
            theta: 离轴角 (rad)
            time_slot: 时隙索引
            generate_new_fading: 是否生成新的随机衰落
            return_intermediate: 是否返回中间结果
            
        Returns:
            如果return_intermediate=True，返回(channel_gain, intermediate_results)
            否则返回channel_gain
        """
        # 1. 计算卫星接收天线增益ω
        omega = self.calculate_omega(theta)
        # 2. 生成雨衰系数r_m_l
        if generate_new_fading:
            r_dB = np.random.normal(self.mu_r, self.sigma_r)
            r_m_l = 10**(r_dB / 20)
        else:
            # 使用固定值（例如均值）
            r_m_l = 10**(self.mu_r / 20)
        # 3. 计算大尺度衰落因子g_m_l 公式(14)
        g_m_l = self.const_part_g / (r_m_l * distance)
        
        # 4. 生成LOS分量h_LOS
        if generate_new_fading:
            power = np.random.uniform(self.h_LOS_power_min, self.h_LOS_power_max)
            magnitude = np.sqrt(power)
            phase = np.random.uniform(0, 2 * np.pi)
            h_LOS = magnitude * cmath.exp(1j * phase)
        else:
            # 使用固定值（例如中间值）
            power = (self.h_LOS_power_min + self.h_LOS_power_max) / 2
            magnitude = np.sqrt(power)
            h_LOS = magnitude  # 相位为0
        
        # 5. 生成NLOS分量h_NLOS
        if generate_new_fading:
            real_part = np.random.normal(0, np.sqrt(self.v_NLOS_real/2))
            imag_part = np.random.normal(0, np.sqrt(self.v_NLOS_imag/2))
            h_NLOS = complex(real_part, imag_part)
        else:
            # 使用固定值（均值为0）
            h_NLOS = complex(0, 0)
        
        # 6. 计算权重系数
        lambda_plus_one = self.lambda_rician + 1
        weight_LOS = np.sqrt(self.lambda_rician * g_m_l / lambda_plus_one)
        weight_NLOS = np.sqrt(g_m_l / lambda_plus_one)
        # 7. 计算信道增益 公式(11)
        channel_gain = omega * (weight_LOS * h_LOS + weight_NLOS * h_NLOS)
        
        return channel_gain
    
    def calculate_channel_gain_mean(self, distance: float, theta: float) -> float:
        """计算信道增益的均值（确定性分析）"""
        # 计算卫星接收天线增益ω
        omega = self.calculate_omega(theta)
        # 使用雨衰均值
        r_m_l = 10**(self.mu_r / 20)
        # 计算大尺度衰落因子g_m_l
        g_m_l = self.const_part_g / (r_m_l * distance)
        # 使用LOS分量的均值
        power_mean = (self.h_LOS_power_min + self.h_LOS_power_max) / 2
        h_LOS_magnitude = np.sqrt(power_mean)
        # 计算权重系数
        lambda_plus_one = self.lambda_rician + 1
        weight_LOS = np.sqrt(self.lambda_rician * g_m_l / lambda_plus_one)
        # 计算信道增益（均值）
        channel_gain_mean = abs(omega * weight_LOS * h_LOS_magnitude)
        return channel_gain_mean
    
    def calculate_transmission_rate(self, P_t, bw, distance, elevation_deg, use_deterministic=False):
        """
        计算发送性能的简化函数
        Args:
            P_t: 发送功率 (W)
            distance: UAV到卫星距离 (m)
            elevation_deg: 仰角 (度)
            channel_model: SatelliteChannel实例
            use_deterministic: 是否使用确定性分析
            
        Returns:
            包含性能指标的字典
        """
        # 1. 将仰角转换为离轴角θ
        theta_rad = math.radians(90.0 - elevation_deg)
        
        if use_deterministic:
            # 使用确定性分析计算信道增益均值
            channel_gain_mag = self.calculate_channel_gain_mean(distance, theta_rad)
            channel_gain = complex(channel_gain_mag, 0)
        else:
            # 使用随机衰落计算信道增益
            # 注意：这里调用时指定return_intermediate=False，只返回信道增益
            channel_gain = self.calculate_channel_gain(
                distance=distance,
                theta=theta_rad,
                generate_new_fading=True,
            )
        
        # 计算接收功率
        P_r = P_t * (abs(channel_gain)**2)
        
        # 计算噪声功率（已预计算为常数）
        noise_power = self.noise_power_const
        
        # 计算信噪比
        SNR = P_r / noise_power
        
        # 计算发送速率（香农容量）
        rates = bw * np.log2(1 + SNR)
        
        # 返回结果
        # performance = {
        #     'distance_km': distance / 1000,
        #     'elevation_deg': elevation_deg,
        #     'theta_deg': math.degrees(theta_rad),
        #     'channel_gain': channel_gain,
        #     'channel_gain_mag': abs(channel_gain),
        #     'channel_gain_db': 20 * np.log10(abs(channel_gain)),
        #     'transmit_power_w': P_t,
        #     'transmit_power_dbm': 10 * np.log10(P_t * 1000),
        #     'receive_power_w': P_r,
        #     'receive_power_dbm': 10 * np.log10(P_r * 1000),
        #     'SNR': SNR,
        #     'SNR_db': 10 * np.log10(SNR),
        #     'rate_bps': rate,
        #     'rate_mbps': rate / 1e6,
        #     'analysis_type': 'deterministic' if use_deterministic else 'stochastic'
        # }
        
        return rates