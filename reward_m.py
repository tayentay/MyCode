import numpy as np
from collections import deque, defaultdict

class PreferenceEstimator:
    """
    Preference Estimator for Communication Link Selection
    
    Implements preference estimation based on effective choice rate:
    θ_hat = (Σ n_j * s_j * s_j^T)^-1 * Σ n_j * s_j * (C_j / (T_j + ε))
    
    where:
    - s_j: d-dimensional feature vector for link j
    - n_j: number of observations for feature j
    - C_j: sum of binary feedback (+1 for ACK, -1 for loss)
    - T_j: sum of effective decision times (RTT for ACK, T_max for loss)
    - ε: small constant for numerical stability
    """
    
    def __init__(self, config, num_ues, num_uavs):
        self.config = config
        self.num_ues = num_ues
        self.num_uavs = num_uavs
        
        # Hyperparameters
        self.lambda_reg = getattr(config, 'preference_lambda', 0.99)
        self.epsilon = getattr(config, 'preference_epsilon', 1e-2)
        self.min_buffer_size = getattr(config, 'preference_min_buffer_size', 2000)
        self.buffer_capacity = getattr(config, 'preference_buffer_capacity', 20000)
        self.fit_interval = getattr(config, 'preference_fit_interval_steps', 100)
        self.explore_eps = getattr(config, 'preference_explore_eps', 0.05)
        self.target_clip = getattr(config, 'preference_target_clip', 20.0)
        self.min_rtt = getattr(config, 'preference_min_rtt', 1e-4)
        
        # Replay buffer for experiences
        self.replay_buffer = deque(maxlen=self.buffer_capacity)
        
        # Statistics for each (UE, UAV) pair feature
        # Key: tuple of feature vector (rounded for hashing)
        # Value: dict with 'n_j', 'C_j', 'T_j', 's_j'
        self.feature_stats = defaultdict(lambda: {
            'n_j': 0,
            'C_j': 0.0,
            'T_j': 0.0,
            's_j': None
        })
        
        # Estimated preference vectors for each UE
        self.theta_hat = {}  # theta_hat[ue_idx] = preference vector
        for ue_idx in range(num_ues):
            self.theta_hat[ue_idx] = None
        
        self.is_fitted = False
        self.step_count = 0
        
        print(f"[PreferenceEstimator] Initialized with λ={self.lambda_reg}, ε={self.epsilon}")
        print(f"[PreferenceEstimator] Buffer capacity={self.buffer_capacity}, min_size={self.min_buffer_size}")
    
    def _get_feature_key(self, feature_vec):
        """Convert feature vector to hashable key (rounded to 4 decimals)"""
        return tuple(np.round(feature_vec, 4))
    
    def update(self, stats, sim):
        """
        Update statistics with new observations from simulation
        
        Args:
            stats: Statistics dictionary from simulation
            sim: Simulation object to extract features
        """
        self.step_count += 1
        
        # Extract information from stats
        # Assuming stats contains information about each UE's transmission
        if not hasattr(sim, 'last_actions') or sim.last_actions is None:
            # No actions recorded yet, skip update
            return
        
        actions = sim.last_actions  # Shape: [num_ues] with UAV indices
        
        # For each UE, record the experience
        experiences_added = 0
        for ue_idx in range(self.num_ues):
            if ue_idx >= len(actions):
                continue
                
            uav_idx = int(actions[ue_idx])
            if uav_idx < 0 or uav_idx >= self.num_uavs:
                continue
            
            # Extract features for this (UE, UAV) pair
            features = self._extract_features(ue_idx, uav_idx, sim)
            if features is None:
                continue
            
            # Get feedback and effective decision time
            feedback, eff_time = self._extract_feedback(ue_idx, uav_idx, stats, sim)
            
            # Store experience
            experience = {
                'ue_idx': ue_idx,
                'uav_idx': uav_idx,
                'features': features,
                'feedback': feedback,
                'eff_time': eff_time
            }
            self.replay_buffer.append(experience)
            experiences_added += 1
            
            # Update feature statistics
            feature_key = self._get_feature_key(features)
            stat = self.feature_stats[feature_key]
            
            if stat['s_j'] is None:
                stat['s_j'] = features.copy()
            
            stat['n_j'] += 1
            stat['C_j'] += feedback
            stat['T_j'] += eff_time
        
        if self.step_count % 50 == 0:
            print(f"[PreferenceEstimator] Step {self.step_count}: Added {experiences_added} experiences, buffer size={len(self.replay_buffer)}")
        
        # Periodically fit the estimator
        if len(self.replay_buffer) >= self.min_buffer_size:
            if self.step_count % self.fit_interval == 0:
                self._fit_estimator()
    
    def _extract_features(self, ue_idx, uav_idx, sim):
        """
        Extract d-dimensional feature vector for (UE, UAV) link
        
        Features include:
        - Received signal strength (or SNR)
        - Distance between UE and UAV
        - Queue length
        - Historical reliability (success rate)
        - Normalized coordinates
        """
        try:
            # Get positions
            if not hasattr(sim, 'ue_positions') or not hasattr(sim, 'uav_positions'):
                return None
            
            ue_pos = sim.ue_positions[ue_idx]
            uav_pos = sim.uav_positions[uav_idx]
            
            # Compute distance
            distance = np.linalg.norm(ue_pos - uav_pos)
            
            # Get SNR or signal strength from G2A_rates or SNR matrix
            if hasattr(sim, 'G2A_SNR_dB') and sim.G2A_SNR_dB is not None:
                # Note: G2A_SNR_dB has shape (num_uavs, num_ues)
                snr_db = sim.G2A_SNR_dB[uav_idx, ue_idx] if len(sim.G2A_SNR_dB.shape) > 1 else 0.0
            else:
                snr_db = 20.0  # Default reasonable SNR
            
            # Compute rate for this specific UE-UAV pair
            if hasattr(sim, 'G2A_gain') and sim.G2A_gain is not None:
                # Calculate rate based on channel gain for this specific link
                noise_power = self.config.N * self.config.G2A_bandwidth
                gain = sim.G2A_gain[uav_idx, ue_idx]
                received_power = self.config.tx_power * gain
                snr_linear = received_power / noise_power if noise_power > 0 else 0
                efficiency = 0.8  # Practical efficiency
                rate = efficiency * self.config.G2A_bandwidth * np.log2(1 + snr_linear) / 1e6  # Mbps
            elif hasattr(sim, 'G2A_rates') and sim.G2A_rates is not None and ue_idx < len(sim.G2A_rates):
                # Fallback: use current assigned rate (less accurate for non-assigned pairs)
                rate = sim.G2A_rates[ue_idx] / 1e6  # Mbps
            else:
                rate = 1.0  # Default
            
            # Normalized distance (assume max area size)
            max_distance = self.config.area_size * 1.5
            norm_distance = distance / max_distance
            
            # Construct feature vector
            features = np.array([
                snr_db / 30.0,  # Normalized SNR (assuming max ~30 dB)
                1.0 / (1.0 + norm_distance),  # Inverse normalized distance
                rate / 10.0,  # Normalized rate (assuming max ~10 Mbps)
                ue_pos[0] / self.config.area_size,  # Normalized UE x
                ue_pos[1] / self.config.area_size,  # Normalized UE y
                uav_pos[0] / self.config.area_size,  # Normalized UAV x
                uav_pos[1] / self.config.area_size,  # Normalized UAV y
                1.0  # Bias term
            ], dtype=np.float32)
            
            return features
            
        except Exception as e:
            print(f"[PreferenceEstimator] Error extracting features: {e}")
            return None
    
    def _extract_feedback(self, ue_idx, uav_idx, stats, sim):
        """
        Extract binary feedback and effective decision time
        
        Returns:
            feedback: +1 for ACK, -1 for packet loss
            eff_time: RTT for ACK, T_max for loss
        """
        T_max = self.config.time_slot_duration  # Use time slot as max timeout
        
        # Check if packet was successfully transmitted
        # This depends on simulation statistics
        success = True  # Default to success
        rtt = self.config.time_slot_duration / 2.0  # Default RTT
        
        # Try to get actual transmission results
        if hasattr(sim, 'packet_success_rate'):
            success_rate = sim.packet_success_rate.get((ue_idx, uav_idx), 0.5)
            success = np.random.random() < success_rate
        elif stats and 'packet_loss_rate' in stats:
            loss_rate = stats.get('packet_loss_rate', 0.1)
            success = np.random.random() > loss_rate
        
        # Simulate RTT variability
        if success:
            # RTT varies based on distance and congestion
            base_rtt = max(self.min_rtt, self.config.time_slot_duration * 0.3)
            rtt = base_rtt * (0.8 + 0.4 * np.random.random())
            feedback = +1
            eff_time = rtt
        else:
            feedback = -1
            eff_time = T_max
        
        return feedback, eff_time
    
    def _fit_estimator(self):
        """
        Fit the preference estimator using aggregated statistics
        
        Computes: θ_hat = (Σ n_j * s_j * s_j^T + λI)^-1 * Σ n_j * s_j * (C_j / (T_j + ε))
        """
        if len(self.feature_stats) == 0:
            print("[PreferenceEstimator] No feature stats to fit")
            return
        
        # Build data for each UE
        fitted_count = 0
        for ue_idx in range(self.num_ues):
            # Collect relevant experiences for this UE
            ue_experiences = [exp for exp in self.replay_buffer if exp['ue_idx'] == ue_idx]
            
            if len(ue_experiences) < 10:  # Need minimum samples
                print(f"[PreferenceEstimator] UE {ue_idx}: Only {len(ue_experiences)} experiences, need 10+")
                continue
            
            # Get feature dimension from first experience
            feature_dim = len(ue_experiences[0]['features'])
            
            # Build matrices
            A = np.zeros((feature_dim, feature_dim))  # Σ n_j * s_j * s_j^T
            b = np.zeros(feature_dim)  # Σ n_j * s_j * (C_j / T_j)
            
            # Aggregate by feature
            feature_data = defaultdict(lambda: {'n': 0, 'C': 0.0, 'T': 0.0, 's': None})
            
            for exp in ue_experiences:
                key = self._get_feature_key(exp['features'])
                if feature_data[key]['s'] is None:
                    feature_data[key]['s'] = exp['features']
                feature_data[key]['n'] += 1
                feature_data[key]['C'] += exp['feedback']
                feature_data[key]['T'] += exp['eff_time']
            
            # Build A and b
            for key, data in feature_data.items():
                s_j = data['s']
                n_j = data['n']
                C_j = data['C']
                T_j = data['T']
                
                # Effective choice rate with clipping
                # This represents reliability/latency: high positive = good, negative = bad
                eff_rate = C_j / (T_j + self.epsilon)
                eff_rate = np.clip(eff_rate, -self.target_clip, self.target_clip)
                
                # Accumulate
                A += n_j * np.outer(s_j, s_j)
                b += n_j * s_j * eff_rate
            
            # Add regularization (scaled for numerical stability)
            reg_scale = np.trace(A) / feature_dim if np.trace(A) > 0 else 1.0
            A += self.lambda_reg * reg_scale * np.eye(feature_dim)
            
            # Solve for theta_hat
            try:
                theta = np.linalg.solve(A, b)
                self.theta_hat[ue_idx] = theta
                fitted_count += 1
                print(f"[PreferenceEstimator] UE {ue_idx}: Fitted θ with {len(ue_experiences)} experiences")
                print(f"  θ norm: {np.linalg.norm(theta):.4f}, features: {len(feature_data)}")
            except np.linalg.LinAlgError:
                print(f"[PreferenceEstimator] UE {ue_idx}: Singular matrix, using pseudo-inverse")
                try:
                    theta = np.linalg.lstsq(A, b, rcond=None)[0]
                    self.theta_hat[ue_idx] = theta
                    fitted_count += 1
                except Exception as e:
                    print(f"[PreferenceEstimator] UE {ue_idx}: Failed to fit - {e}")
        
        if fitted_count > 0:
            self.is_fitted = True
            print(f"[PreferenceEstimator] Fitted estimator for {fitted_count}/{self.num_ues} UEs with {len(self.replay_buffer)} experiences")
    
    def get_action(self, sim):
        """
        Get action (UAV selection) for each UE based on estimated preferences
        
        Returns:
            actions: numpy array of shape [num_ues] with UAV indices
        """
        actions = np.zeros(self.num_ues, dtype=np.int32)
        
        for ue_idx in range(self.num_ues):
            # Epsilon-greedy exploration
            if np.random.random() < self.explore_eps:
                actions[ue_idx] = np.random.randint(0, self.num_uavs)
                continue
            
            # Use preference scores if available
            if self.theta_hat[ue_idx] is None:
                # Fallback: select closest UAV
                if hasattr(sim, 'ue_positions') and hasattr(sim, 'uav_positions'):
                    ue_pos = sim.ue_positions[ue_idx]
                    distances = [np.linalg.norm(sim.uav_positions[i] - ue_pos) for i in range(self.num_uavs)]
                    actions[ue_idx] = np.argmin(distances)
                else:
                    actions[ue_idx] = np.random.randint(0, self.num_uavs)
                continue
            
            # Compute preference scores for all UAVs
            scores = []
            valid_features = []
            for uav_idx in range(self.num_uavs):
                features = self._extract_features(ue_idx, uav_idx, sim)
                if features is None:
                    scores.append(-1e9)
                    valid_features.append(False)
                else:
                    score = np.dot(features, self.theta_hat[ue_idx])
                    scores.append(score)
                    valid_features.append(True)
            
            # Select UAV with highest score
            if any(valid_features):
                actions[ue_idx] = np.argmax(scores)
            else:
                # All features invalid, fallback to random
                actions[ue_idx] = np.random.randint(0, self.num_uavs)
        
        # Debug output (print occasionally)
        if self.step_count % 100 == 0:
            print(f"[PreferenceEstimator] Step {self.step_count}: Actions={actions}, fitted={[self.theta_hat[i] is not None for i in range(self.num_ues)]}")
        
        return actions
