# MyCode

## UAV-Ground Communication Network with Deep Reinforcement Learning

This repository implements a UAV-based communication network simulation using ns-3 with Deep Reinforcement Learning (DRL) for intelligent link selection.

## Key Features

- **PPO-based Link Selection**: Uses Proximal Policy Optimization for UAV-UE assignment
- **Preference Estimation Algorithm**: Implements preference-based link selection from "Enhancing_Preference_bas" paper
- **Self-Predictive Representation Learning**: Optional module for learning robust state representations
- **Multi-task Learning with GradNorm**: Balances multiple learning objectives
- **Channel Models**: Realistic G2A (Ground-to-Air) and A2S (Air-to-Satellite) channel models

## Components

- `ppo_1.py`: Main PPO training loop with preference estimator integration
- `reward_m.py`: Preference estimation algorithm implementation
- `env.py`: RL environment wrapper for ns-3 simulation
- `commnet.py`: UAV network simulation using ns-3
- `config.py`: Configuration parameters
- `channel_utils.py`: Channel models and schedulers
- `self_predictive.py`: Self-predictive representation learning
- `gradnorm.py`: Gradient normalization for multi-task learning
- `utils.py`: Utility functions

## Getting Started

### Configuration

Edit `config.py` to configure:
- Number of UEs and UAVs
- Preference estimation parameters
- Self-predictive learning options
- Channel and mobility parameters

### Running

```bash
python ppo_1.py [options]
```

Key options:
- `--num-updates`: Total training episodes (default: 1000)
- `--learning-rate`: Learning rate (default: 3e-4)
- `--eval-mode`: Run in evaluation mode (no training)
- `--load-model-path`: Load pretrained model

## Preference Estimation Algorithm

The preference estimator learns user equipment preferences for UAV selection based on:
- Binary feedback (ACK/loss)
- Effective decision times (RTT/timeout)
- Link features (SNR, distance, rate, positions)

See [PREFERENCE_ESTIMATOR_README.md](PREFERENCE_ESTIMATOR_README.md) for detailed documentation.

## References

- Paper: "Enhancing_Preference_bas" (see Enhancing_Preference_bas.pdf)
- ns-3 Network Simulator: https://www.nsnam.org/
