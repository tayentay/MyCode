# Preference Estimator Implementation

## Overview

This implementation adds a preference estimation algorithm based on the paper "Enhancing_Preference_bas" for communication link selection in UAV networks. The algorithm learns user equipment (UE) preferences for selecting UAV communication links based on binary feedback (ACK/loss) and effective decision times (RTT/timeout).

## Algorithm Description

### Core Concept

The preference estimator learns a latent preference vector θ_n for each UE that linearly quantifies link quality:

```
Quality(link) = s_j^T * θ_n
```

where s_j is a d-dimensional feature vector characterizing the link.

### Key Components

1. **Binary Feedback Signal**:
   - c_j = +1 if ACK is successfully received
   - c_j = -1 if packet is lost (timeout)

2. **Effective Decision Time**:
   - t̃_j = RTT_j if ACK is received
   - t̃_j = T_max if packet loss (timeout)

3. **Preference Estimation**:
   ```
   θ̂_n = (Σ n_j * s_j * s_j^T + λI)^(-1) * Σ n_j * s_j * (C_j / (T_j + ε))
   ```
   where:
   - n_j: number of observations for feature vector s_j
   - C_j: sum of binary feedback signals
   - T_j: sum of effective decision times
   - λ: ridge regularization parameter
   - ε: small constant for numerical stability

4. **Feature Vector Components**:
   - Normalized SNR (signal-to-noise ratio)
   - Inverse normalized distance
   - Normalized transmission rate
   - Normalized UE position (x, y)
   - Normalized UAV position (x, y)
   - Bias term

## Files Added

1. **reward_m.py**: PreferenceEstimator class implementing the algorithm
2. **utils.py**: Argument parsing utilities
3. **self_predictive.py**: Self-predictive representation learning module
4. **gradnorm.py**: Gradient normalization for multi-task learning
5. **channel_utils.py**: Channel models and schedulers for G2A and A2S links

## Configuration

Key parameters in `config.py`:

```python
enable_preference_algo: bool = True          # Enable/disable the algorithm
preference_warmup_steps: int = 200           # Steps before using estimator
preference_lambda: float = 0.99              # Regularization parameter
preference_epsilon: float = 1e-2             # Numerical stability constant
preference_buffer_capacity: int = 20000      # Replay buffer size
preference_min_buffer_size: int = 500        # Minimum buffer before fitting
preference_fit_interval_steps: int = 50      # Fit frequency
preference_explore_eps: float = 0.1          # Exploration rate
preference_target_clip: float = 20.0         # Clipping for effective rates
```

## How It Works

### Training Phase (Warmup)

During the first `preference_warmup_steps` steps:
1. PPO agent selects actions
2. Preference estimator observes:
   - Selected UE-UAV pairs
   - Link features (SNR, distance, rate, etc.)
   - Binary feedback (success/failure)
   - Effective decision times (RTT or timeout)
3. Experiences are stored in replay buffer
4. Estimator is fitted periodically every `preference_fit_interval_steps` steps

### Inference Phase

After warmup:
1. Preference estimator computes scores for all candidate UAVs for each UE
2. Selects UAV with highest preference score: `q_i = s_i^T * θ̂_n`
3. Uses ε-greedy exploration (10% random actions)
4. Falls back to closest UAV if preference vector is unavailable

## Key Improvements

### Bug Fixes
1. **SNR Matrix Indexing**: Fixed reversed indexing `[ue_idx, uav_idx]` → `[uav_idx, ue_idx]`
2. **Rate Calculation**: Now properly evaluates each UE-UAV pair independently
3. **Vectorized SNR Computation**: Improved performance by removing nested loops

### Algorithmic Improvements
1. **Scaled Regularization**: Regularization scaled by matrix trace for numerical stability
2. **Robust Fitting**: Added fallback to pseudo-inverse for singular matrices
3. **Extensive Debugging**: Added logging to track fitting progress and action selection
4. **Fallback Mechanisms**: Defaults to closest UAV when preference unavailable

### Hyperparameter Tuning
1. Reduced warmup from 500 → 200 steps for faster testing
2. Reduced min buffer from 2000 → 500 for faster initialization
3. Increased exploration from 5% → 10% for more diverse actions
4. Fit more frequently (every 50 vs 100 steps)

## Debugging

The implementation includes extensive debugging output:

```
[PreferenceEstimator] Initialized with λ=0.99, ε=0.01
[PreferenceEstimator] Step 50: Added 9 experiences, buffer size=450
[PreferenceEstimator] Step 500: Buffer size=4500
[PreferenceEstimator] UE 0: Fitted θ with 500 experiences
  θ norm: 2.3456, features: 15
[PreferenceEstimator] Fitted estimator for 9/9 UEs with 4500 experiences
[PreferenceEstimator] Switched to estimator at step=200, buffer=1800
[PreferenceEstimator] Step 200: Actions=[1 2 0 3 2 1 3 0 1], fitted=[True, True, True, ...]
```

## Troubleshooting

### Issue: Actions all zeros `[[0 0 0 0 0 0 0 0 0]]`

**Possible Causes**:
1. Preference estimator not fitted yet (still in warmup)
2. All preference vectors are None (not enough data)
3. All preference scores similar (always selecting UAV 0)

**Solutions**:
1. Check if warmup period has passed
2. Verify replay buffer has sufficient experiences
3. Check debug logs for fitting status
4. Ensure features are being extracted correctly
5. Verify SNR and rate calculations are working

### Issue: Poor Performance

**Possible Causes**:
1. Regularization too strong/weak
2. Features not discriminative enough
3. Feedback signal too noisy

**Solutions**:
1. Tune `preference_lambda` (regularization)
2. Adjust `preference_epsilon` (stability constant)
3. Increase `preference_explore_eps` (exploration)
4. Check feature extraction logic

## Integration with DRL

The preference estimator integrates with PPO as follows:

1. **During Warmup**: PPO trains normally while estimator collects data
2. **After Warmup**: Estimator takes over action selection, PPO training stops
3. **Epsilon-Greedy**: Maintains exploration even during inference

Note: Once preference estimator is active, PPO training is disabled. This is by design to allow pure preference-based policy.

## Future Enhancements

Potential improvements:
1. Combine PPO and preference scores (weighted mixture)
2. Online adaptation of preference vectors
3. Multi-objective optimization (latency, reliability, throughput)
4. Hierarchical preference learning
5. Transfer learning across similar scenarios
