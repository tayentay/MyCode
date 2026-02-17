"""
Self-Predictive Representation Learning Module with Multi-Head Self-Attention
Based on: "Bridging State and History Representations: Understanding Self-Predictive RL" (ICLR 2024)
Repository: https://github.com/twni2016/self-predictive-rl
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ========================= Utility Functions =========================

def sp_weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target: nn.Module, source: nn.Module):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


# ========================= Network Definitions =========================

class LatentEncoder(nn.Module):
    """
    Encoder: obs -> z (latent state).
    单步输入不变，保持与 PPO agent 的兼容性。
    """
    def __init__(self, obs_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.apply(sp_weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ====================== Multi-Head Self-Attention ======================

class TemporalSelfAttention(nn.Module):
    """
    对潜在状态序列 z_1, z_2, ..., z_T 做多头自注意力，
    捕获时序上下文依赖，输出上下文增强后的 z 序列。

    用于 ZP loss 计算前，增强 encoder 输出的 z 的时��表征能力。
    """
    def __init__(
        self,
        latent_dim: int,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.0,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads

        # 可学习的位置编码
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, latent_dim) * 0.02
        )

        # Transformer Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Layer norm on output
        self.output_norm = nn.LayerNorm(latent_dim)

        self.apply(sp_weight_init)

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_seq: (batch, seq_len, latent_dim) 潜在状态序列
        Returns:
            z_enhanced: (batch, seq_len, latent_dim) 注意力增强后的序列
        """
        B, T, D = z_seq.shape

        # 加位置编码
        z_seq = z_seq + self.pos_embedding[:, :T, :]

        # 因果掩码
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=z_seq.device)

        # 不再需要强制 MATH 后端，GradNorm-Lite 无二阶梯度
        z_enhanced = self.transformer_encoder(z_seq, mask=causal_mask)

        # 残差连接 + LayerNorm
        z_enhanced = self.output_norm(z_enhanced + z_seq)

        return z_enhanced


# ========================= Transition Model =========================

class LatentTransitionModel(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        layers = [nn.Linear(latent_dim + action_dim, hidden_dim), nn.ELU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
        layers += [nn.Linear(hidden_dim, latent_dim)]
        self.model = nn.Sequential(*layers)
        self.apply(sp_weight_init)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, action], dim=-1)
        return self.model(x)


# ========================= Self-Predictive Module =========================

class SelfPredictiveModule(nn.Module):
    """
    Self-Predictive Representation Learning Module + Multi-Head Self-Attention.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        model_hidden_dim: int = 256,
        model_num_layers: int = 2,
        tau: float = 0.005,
        attn_num_heads: int = 4,
        attn_num_layers: int = 1,
        attn_dropout: float = 0.0,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__()
        self.tau = tau
        self.device = device
        self.latent_dim = latent_dim

        # Online encoder
        self.encoder = LatentEncoder(obs_dim, hidden_dim, latent_dim).to(device)

        # Target encoder (EMA)
        self.encoder_target = LatentEncoder(obs_dim, hidden_dim, latent_dim).to(device)
        hard_update(self.encoder_target, self.encoder)
        for param in self.encoder_target.parameters():
            param.requires_grad = False

        # Temporal Self-Attention
        self.temporal_attention = TemporalSelfAttention(
            latent_dim=latent_dim,
            num_heads=attn_num_heads,
            num_layers=attn_num_layers,
            dropout=attn_dropout,
        ).to(device)

        # Latent transition model
        self.transition_model = LatentTransitionModel(
            latent_dim, action_dim, model_hidden_dim, model_num_layers
        ).to(device)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def encode_target(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.encoder_target(obs)

    def compute_zp_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> torch.Tensor:
        T = obs.shape[0]

        if self.temporal_attention is not None and T > 1:
            z_seq = self.encoder(obs)
            z_seq_batched = z_seq.unsqueeze(0)
            z_enhanced_batched = self.temporal_attention(z_seq_batched)
            z_enhanced = z_enhanced_batched.squeeze(0)
            z_next_pred = self.transition_model(z_enhanced, action)

            with torch.no_grad():
                z_next_target = self.encoder_target(next_obs)

            zp_loss = ((z_next_pred - z_next_target) ** 2).sum(dim=-1).mean()
        else:
            z = self.encoder(obs)
            z_next_pred = self.transition_model(z, action)

            with torch.no_grad():
                z_next_target = self.encoder_target(next_obs)

            zp_loss = ((z_next_pred - z_next_target) ** 2).sum(dim=-1).mean()

        return zp_loss

    def update_target_encoder(self):
        soft_update(self.encoder_target, self.encoder, self.tau)

    def get_all_parameters(self):
        params = list(self.encoder.parameters()) + list(self.transition_model.parameters())
        params += list(self.temporal_attention.parameters())
        return params

    def get_transition_model_parameters(self):
        params = list(self.transition_model.parameters())
        params += list(self.temporal_attention.parameters())
        return params