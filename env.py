#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV-Ground Network Simulation for ns-3.46.1
"""
from ns import ns
import numpy as np
from config import Config
from commnet import UAVNetworkSimulation


class EnvForDRL:
    """
    封装UAV网络仿真为 RL-friendly 环境，用于PPO训练
    """
    def __init__(self, config: Config):
        self.config = config
        self.seed = self.config.seed
        self.step_count = 0
        # self.max_steps = int(config.simulation_time / config.time_slot_duration)
        self.single_observation_space = config.num_ues * config.num_uavs + config.num_uavs*3 + config.num_ues + config.num_uavs + getattr(config, 'noise_dim', 0)
        self.single_action_space = config.num_ues
        self.sum_action = config.num_ues*config.num_uavs
        self.nvec = np.array([config.num_uavs]*config.num_ues)
        # self.snr_matrix = []
        # self.ue_positions = []
        # self.uav_positions = []
        self.sim = UAVNetworkSimulation(self.config)
        self.reset()
    
    def reset(self):
        print("Resetting environment...")
        # 重新初始化场景，节点，重置计数
        # try:
        #     from ns.internet import Ipv4AddressGenerator
        #     Ipv4AddressGenerator.Reset()
        # except Exception:
        #     pass  # 如果没有该模块则跳过
        self.sim.setup_all()
        self.step_count = 0
        return self.sim.get_observation()


    def step(self, action):
        print("action:",action)
        """
        action: shape=[num_ues], 每个元素为0~num_uavs-1, 代表UE被分配到的UAV
        """
        self.sim.do_scheduling_action(action)
        # reward = self.sim.G2A_rates.sum() / 1e6  # Mbps
        # reward = self.sim.trans_per_step / 1e4  # 
        reward = (self.sim.G2A_rates.sum()/1e6+0.5*self.sim.trans_per_step/1e4) * \
                        (1-self.sim.ooo_monitor.m_oooCount/1e4)
        print('reward1:',self.sim.G2A_rates.sum() / 1e6,'reward2：',0.1*self.sim.trans_per_step / 1e4)
        obs = self.sim.get_observation()
        # 计算reward: 当前时隙所有分配的UE到UAV的容量和
        # Optionally, info dict记录一些统计量，供 wandb 等用
        # info = dict(
        #     step=self.step_count,
        #     total_throughput_Mbps=reward,
        #     snr_matrix=np.array(self.snr_matrix),
        #     assignment=assignment,
        #     uav_throughput_Mbps=[v/1e6 for v in record["uav_throughput"]],
        # )

        # done = (self.step_count >= self.max_steps)
        done = True
        self.step_count += 1
        if (self.step_count+1) == self.config.simulation_time / self.config.time_slot_duration:
            done = True
        return obs, reward, done
    
    def get_metrics(self):
        # 返回一些统计指标
        metrics = self.sim.collect_results()
        return metrics