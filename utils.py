import numpy as np
import torch
import math
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from distutils.util import strtobool


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-name", type=str, default="4", help="the project name")
    parser.add_argument("--out-subdir", type=str, default="runs", help="the name of output subdirectory")
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="MicrortsMining-v1",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
        help="the learning rate of the optimizer")# 2.5e-5,
    # parser.add_argument("--total-timesteps", type=int, default=12800,
    #     help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to save the model")
    parser.add_argument("--save-interval", type=int, default=100, help="interval for saving models (in updates)")
    parser.add_argument("--load-model-path", type=str, default='./4/runs/PPO_RateB_True_2026-02-16_10-03-40/model_900.pth', help="path to load a pretrained model from")
    parser.add_argument("--eval-mode", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, run in evaluation mode (no training)")
    parser.add_argument("--log-interval", type=int, default=10, help="log interval for updates")

    # Algorithm specific arguments
    parser.add_argument("--num-updates", type=int, default=1000,
        help="total updates of the models")
    parser.add_argument("--num-steps", type=int, default=64,
        help="the number of steps to run in each environment per policy rollout") #128
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--simulation_time", type=float, default=None,
        help="Duration of the simulation in seconds (overrides Config if set)")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    # Noise options for robustness testing
    parser.add_argument("--noise-dim", type=int, default=0, help="dimension of additive noise to append to observations")
    parser.add_argument("--noise-scale", type=float, default=0.1, help="standard deviation of additive Gaussian noise")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args




# Fourth new ######################################################################
def smooth_boost_congestion_control(cwnd=1, ssthresh=128, a=7, b=0.5, gamma=1.0, scale=None, M_max=4):
    """
    指数衰减的 Boost (平滑且受控)
    
    参数:
        cwnd: 当前拥塞窗口大小
        ssthresh: 慢启动阈值
        a: S型基准参数，默认7
        b: S型基准参数，默认0.5
        gamma: boost增幅系数，默认1.0 (早期boost增加100%，初期multiplier≈3)
        scale: boost衰减速度，默认为 ssthresh / 4
        M_max: 最大RTT乘子上界，默认4 (单RTT不超过4×)
    
    返回:
        cwnd_next: 下一个RTT的拥塞窗口大小
    """

    if scale is None:
        scale = ssthresh / 4

    # 计算 x = cwnd / ssthresh
    x = cwnd / ssthresh
    
    # 计算 base_gf (S型基准)
    # base_gf = 1 / (1 + exp(a(x - b)))
    base_gf = 1 / (1 + math.exp(a * (x - b)))
    print("ssss",math.exp(a * (x - b)))
    # 计算 boost (早期额外增幅，随cwnd指数衰减)
    # boost = 1 + γ · exp(-cwnd/scale)
    boost = 1 + gamma * math.exp(-cwnd / scale)
    
    # 计算 multiplier (对单RTT的最大乘子上界)
    # multiplier = min(1 + base_gf × boost, M_max)
    multiplier = min(1 + base_gf * boost, M_max)
    
    # 计算下一个RTT的cwnd
    # cwnd_next = cwnd × multiplier
    cwnd_next = cwnd * multiplier
    print(f"Debug: cwnd={cwnd}, x={x:.4f}, base_gf={base_gf:.4f}, boost={boost:.4f}, multiplier={multiplier:.4f}")

    return cwnd_next







###################################################################################
def compute_jain_fairness(x):
        """
        Computes the Jain's fairness index of entries in the given ndarray.

        Jain's fairness index is calculated as:
        (sum(x)^2) / (n * sum(x^2))
        where n is the number of elements in x. The index ranges from 0 to 1,
        with 1 indicating perfect fairness (all elements are equal).

        Parameters:
        x (ndarray): Input array of non-negative values.

        Returns:
        float: Jain's fairness index.
        """
        if x.size > 0:
            x = np.clip(x, 1e-6, np.inf)
            return np.square(x.sum()) / (x.size * np.square(x).sum())
        else:
            raise ValueError("Jain's fairness index is undefined for an empty set.")

# 要当具体文件里设置，调用无效
# def set_rand_seed(seed):
#     """Sets random seed for reproducibility."""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)

class Writer:
    def __init__(self, args, path):
        self.writer = SummaryWriter(path)
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % (
                "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    def write_scalar(self, inf, steps , epoch):
        for key, value in inf.items():
            self.writer.add_scalar(key, value / steps, epoch)

class GroundToAirChannel(object):
    """Groud-to-Air (ATG) channel model"""

    chan_params = {
        'suburban': (4.88, 0.43, 0.1, 21),
        'urban': (9.61, 0.16, 1, 20),
        'dense-urban': (12.08, 0.11, 1.6, 23),
        'high-rise-urban': (27.23, 0.08, 2.3, 34)
    }

    def __init__(self, scene, fc):
        # Determine the scene-specific parameters.
        params = self.chan_params[scene]
        self.a, self.b = params[0], params[1]  # Constants for computing p_los
        self.eta_los, self.eta_nlos = params[2], params[3]  # Path loss exponents (LoS/NLoS)

        self.fc = fc  # Central carrier frequency (Hz)

    def estimate_chan_gain(self, d, angle):
        """Estimates the channel gain from horizontal distance.
        d: Three-dimensional distance
        angle: The Angle between height and horizontal distance
        """
        # Estimate probability of LoS link emergence.
        p_los = 1 / (1 + self.a * np.exp(-self.b * (angle - self.a)))
        # Get direct link distance.
        # d = np.sqrt(np.square(d_level) + np.square(h_ubs))
        # Compute free space path loss (FSPL).
        fspl = (4 * np.pi * self.fc * d / 3e8) ** 2
        # Path loss is the weighted average of LoS and NLoS cases.
        pl = p_los * fspl * 10 ** (self.eta_los / 20) + (1 - p_los) * fspl * 10 ** (self.eta_nlos / 20)
        return 1 / pl


class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, center=True, scale=True, clip=None):
        assert shape is not None
        if clip is not None:
            assert clip > 0

        self.center = center
        self.scale = scale
        self.clip = clip
        self.rs = RunningStat(shape)

    def __call__(self, x, **kwargs):
        self.update_statistics(x)

        if self.center:
            x = x - self.rs.mean
        if self.scale:
            if self.center:
                x = x / (self.rs.std + 1e-8)
            else:
                # diff = x - self.rs.mean
                # diff = diff / (self.rs.std + 1e-8)
                # x = diff + self.rs.mean
                x = (x - self.rs.mean) / (self.rs.std + 1e-8) + self.rs.mean

        if self.clip:
            x = torch.clip(x, -self.clip, self.clip)

        return x

    def update_statistics(self, x):
        self.rs.push(x)


class RunningStat(object):


    def __init__(self, shape):
        self._n = 0
        self._M = torch.zeros(*shape, dtype=torch.float32)
        self._S = torch.zeros(*shape, dtype=torch.float32)

    def push(self, x):
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.clone()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else torch.square(self._M)

    @property
    def std(self):
        return torch.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape

## 新增

# def cat(data_list):
#     """Concatenates list of inputs"""
#     if isinstance(data_list[0], torch.Tensor):
#         return torch.cat(data_list)
#     elif isinstance(data_list[0], HeteroData):
#         return Batch.from_data_list(data_list)
#     else:
#         raise TypeError("Unrecognised observation type.")
    
# def check_args_sanity(args):
#     """Checks sanity and avoids conflicts of arguments."""

#     # Ensure specified cuda is used when it is available.
#     if args.device == 'cuda' and torch.cuda.is_available():
#         args.device = f'cuda:{args.cuda_index}'
#     else:
#         args.device = 'cpu'
#     print(f"Choose to use {args.device}.")

#     # When QMix is used, ensure a scalar reward is used.
#     if hasattr(args, 'mixer'):
#         if args.mixer and not args.share_reward:
#             args.share_reward = True
#             print("Since QMix is used, all agents are forced to share a scalar reward.")

#     return args