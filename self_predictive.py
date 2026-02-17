import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfPredictiveEncoder(nn.Module):
    """Encoder network for self-predictive representation learning"""
    def __init__(self, obs_dim, latent_dim, hidden_dim):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
    
    def forward(self, obs):
        return self.net(obs)


class TransitionModel(nn.Module):
    """Transition model predicting next latent state from current state and action"""
    def __init__(self, latent_dim, action_dim, hidden_dim, num_layers=2):
        super().__init__()
        
        layers = []
        input_dim = latent_dim + action_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, z, action):
        x = torch.cat([z, action], dim=-1)
        return self.net(x)


class SelfPredictiveModule(nn.Module):
    """
    Self-Predictive Representation Learning Module
    Learns representations by predicting future states
    """
    def __init__(self, obs_dim, action_dim, latent_dim=128, hidden_dim=256,
                 model_hidden_dim=256, model_num_layers=2, tau=0.005,
                 attn_num_heads=4, attn_num_layers=1, attn_dropout=0.0,
                 device='cpu'):
        super().__init__()
        self.device = device
        self.tau = tau
        self.latent_dim = latent_dim
        
        # Online encoder (trainable)
        self.encoder = SelfPredictiveEncoder(obs_dim, latent_dim, hidden_dim).to(device)
        
        # Target encoder (EMA of online encoder)
        self.target_encoder = SelfPredictiveEncoder(obs_dim, latent_dim, hidden_dim).to(device)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        
        # Transition model
        self.transition_model = TransitionModel(
            latent_dim, action_dim, model_hidden_dim, model_num_layers
        ).to(device)
    
    def forward(self, obs):
        """Encode observation to latent representation"""
        return self.encoder(obs)
    
    def compute_zp_loss(self, obs_t, action_t, obs_tp1):
        """
        Compute self-predictive loss
        Args:
            obs_t: current observations [batch, obs_dim]
            action_t: actions taken [batch, action_dim]
            obs_tp1: next observations [batch, obs_dim]
        """
        # Encode current state
        z_t = self.encoder(obs_t)
        
        # Predict next latent state
        z_tp1_pred = self.transition_model(z_t, action_t)
        
        # Encode next state with target encoder
        with torch.no_grad():
            z_tp1_target = self.target_encoder(obs_tp1)
        
        # Compute MSE loss
        loss = F.mse_loss(z_tp1_pred, z_tp1_target)
        
        return loss
    
    def update_target_encoder(self):
        """Update target encoder with EMA"""
        with torch.no_grad():
            for param, target_param in zip(self.encoder.parameters(), 
                                          self.target_encoder.parameters()):
                target_param.data.mul_(1 - self.tau)
                target_param.data.add_(self.tau * param.data)
    
    def get_transition_model_parameters(self):
        """Return transition model parameters for separate optimizer"""
        return self.transition_model.parameters()
