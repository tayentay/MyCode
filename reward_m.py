import numpy as np
import collections

class PreferenceEstimator:
    def __init__(self, config, num_ues, num_uavs):
        self.config = config
        self.num_ues = num_ues
        self.num_uavs = num_uavs
        
        # Dimensions of feature vector s_j
        # Features: Global observation from commnet.py (excluding noise)
        # Size = num_ues*num_uavs + num_uavs*3 + num_ues + num_uavs
        self.feature_dim = config.num_ues * config.num_uavs + config.num_uavs*3 + config.num_ues + config.num_uavs
        
        # Hyperparameters
        self.lambda_reg = getattr(config, 'preference_lambda', 0.1)
        self.epsilon = getattr(config, 'preference_epsilon', 1e-4)
        self.explore_eps = getattr(config, 'preference_explore_eps', 0.05)
        self.timeout_threshold = 1.0 # T_max (seconds)
        self.target_clip = float(getattr(config, 'preference_target_clip', 20.0))
        self.min_rtt = float(getattr(config, 'preference_min_rtt', 1e-4))

        # Replay buffer and fit schedule
        self.buffer_capacity = int(getattr(config, 'preference_buffer_capacity', 20000))
        self.min_buffer_size = int(getattr(config, 'preference_min_buffer_size', 2000))
        self.fit_interval_steps = int(getattr(config, 'preference_fit_interval_steps', 100))
        self.step_counter = 0
        self.last_fit_step = -1
        self.is_fitted = False
        self.replay_buffer = collections.deque(maxlen=self.buffer_capacity)
        self.state_mean = np.zeros(self.feature_dim, dtype=np.float32)
        self.state_std = np.ones(self.feature_dim, dtype=np.float32)
        self.theta_cache = np.zeros((num_ues, num_uavs, self.feature_dim), dtype=np.float32)
        
        # A and b per UE per UAV (Contextual Bandit per Arm)
        # A[ue_id][uav_id] -> (D, D)
        self.A = np.zeros((num_ues, num_uavs, self.feature_dim, self.feature_dim), dtype=np.float32)
        # b[ue_id][uav_id] -> (D,)
        self.b = np.zeros((num_ues, num_uavs, self.feature_dim), dtype=np.float32)
        
        for n in range(num_ues):
            for m in range(num_uavs):
                self.A[n, m] = np.eye(self.feature_dim) * self.lambda_reg

        # Historical reliability: [accum_acks, accum_losses]
        self.reliability_stats = np.zeros((num_ues, num_uavs, 2), dtype=np.float32)

    def _reset_linear_system(self):
        self.A.fill(0.0)
        self.b.fill(0.0)
        for n in range(self.num_ues):
            for m in range(self.num_uavs):
                self.A[n, m] = np.eye(self.feature_dim, dtype=np.float32) * self.lambda_reg

    def calculate_features(self, sim):
        """
        Return global observation vector (excluding noise)
        """
        if not hasattr(sim, 'G2A_gain'):
            return np.zeros(self.feature_dim, dtype=np.float32)
            
        # Match commnet.py get_observation() logic exactly (minus noise)
        obs = sim.G2A_gain.flatten()*1e9  # shape=(num_uavs*num_ues,)
        obs = np.concatenate((obs, sim.uav_positions.flatten()/1000))  # shape=(num_uavs*num_ues+num_uavs*3,)
        
        # Append per-UE normalized send-buffer-remaining
        try:
            buf = sim.ue_send_buf_remain.flatten()
            obs = np.concatenate((obs, buf))
        except Exception:
            obs = np.concatenate((obs, np.ones(self.num_ues, dtype=np.float32)))
            
        # Append per-UAV A2S queue occupancy
        try:
            obs = np.concatenate((obs, sim.A2S_queue_occupancy.flatten()))
        except Exception:
            obs = np.concatenate((obs, np.zeros(self.num_uavs, dtype=np.float32)))
            
        return obs

    def _normalize_state(self, state):
        return (state - self.state_mean) / self.state_std

    def _build_target(self, acks, losses, rtt_sum):
        c_j = float(acks - losses)
        t_j = float(rtt_sum) + float(losses) * self.timeout_threshold
        t_j = max(t_j, self.min_rtt)
        raw = c_j / (t_j + self.epsilon)
        shaped = np.sign(raw) * np.log1p(abs(raw))
        return float(np.clip(shaped, -self.target_clip, self.target_clip))

    def add_sample(self, sim, stats):
        state = self.calculate_features(sim).astype(np.float32, copy=True)
        clean_stats = {}
        for ue_id, uav_stats in stats.items():
            ue_key = int(ue_id)
            clean_stats[ue_key] = {}
            for uav_id, stat in uav_stats.items():
                clean_stats[ue_key][int(uav_id)] = {
                    'acks': int(stat.get('acks', 0)),
                    'losses': int(stat.get('losses', 0)),
                    'rtt_sum': float(stat.get('rtt_sum', 0.0)),
                }
        self.replay_buffer.append({'state': state, 'stats': clean_stats})
        self.step_counter += 1

    def can_fit(self):
        return len(self.replay_buffer) >= self.min_buffer_size

    def fit_from_buffer(self):
        if not self.can_fit():
            return False

        states = np.stack([sample['state'] for sample in self.replay_buffer], axis=0)
        self.state_mean = states.mean(axis=0).astype(np.float32)
        state_std = states.std(axis=0).astype(np.float32)
        self.state_std = np.clip(state_std, 1e-3, None)

        self._reset_linear_system()
        self.reliability_stats.fill(0.0)

        for sample in self.replay_buffer:
            state = self._normalize_state(sample['state'])
            stats = sample['stats']
            for ue_id, uav_stats in stats.items():
                for uav_id, stat in uav_stats.items():
                    acks = stat['acks']
                    losses = stat['losses']
                    rtt_sum = stat['rtt_sum']

                    if acks == 0 and losses == 0:
                        continue

                    self.reliability_stats[ue_id, uav_id, 0] += acks
                    self.reliability_stats[ue_id, uav_id, 1] += losses

                    eff_rate = self._build_target(acks, losses, rtt_sum)

                    n_samples = acks + losses
                    self.A[ue_id, uav_id] += n_samples * np.outer(state, state)
                    self.b[ue_id, uav_id] += n_samples * state * eff_rate

        for u in range(self.num_ues):
            for v in range(self.num_uavs):
                try:
                    self.theta_cache[u, v] = np.linalg.solve(self.A[u, v], self.b[u, v])
                except np.linalg.LinAlgError:
                    self.theta_cache[u, v] = np.linalg.lstsq(self.A[u, v], self.b[u, v], rcond=1e-5)[0]

        self.is_fitted = True
        self.last_fit_step = self.step_counter
        return True

    def maybe_fit(self):
        if not self.can_fit():
            return False
        if self.last_fit_step >= 0 and (self.step_counter - self.last_fit_step) < self.fit_interval_steps:
            return False
        return self.fit_from_buffer()

    def update(self, stats, sim):
        self.add_sample(sim, stats)
        self.maybe_fit()

    def get_action(self, sim):
        """
        Return actions for all UEs
        """
        actions = np.zeros(self.num_ues, dtype=np.int32)
        current_features = self._normalize_state(self.calculate_features(sim))
        
        for u in range(self.num_ues):
            scores = np.zeros(self.num_uavs)
            
            for v in range(self.num_uavs):
                theta_v = self.theta_cache[u, v]
                scores[v] = np.dot(current_features, theta_v)
            
            if np.random.rand() < self.explore_eps:
                actions[u] = np.random.randint(0, self.num_uavs)
            else:
                actions[u] = np.argmax(scores)
                
        return actions
