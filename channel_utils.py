"""
Channel utilities for UAV-Ground and Air-to-Satellite communication
"""
import numpy as np
import math
from config import Config


class G2AChannel:
    """Ground-to-Air Channel Model"""
    
    def __init__(self, config: Config):
        self.config = config
        self.frequency = config.frequency
        self.tx_power = config.tx_power
        self.noise_figure = config.noise_figure
        self.N = config.N
        self.scene = config.scene
        
    def estimate_chan_gain(self, ue_positions, uav_positions):
        """
        Estimate channel gain between UEs and UAVs
        
        Returns:
            gain_matrix: shape (num_uavs, num_ues) - channel gain in linear scale
        """
        num_uavs = len(uav_positions)
        num_ues = len(ue_positions)
        gain_matrix = np.zeros((num_uavs, num_ues), dtype=np.float32)
        
        for i in range(num_uavs):
            for j in range(num_ues):
                # Calculate 3D distance
                distance = np.linalg.norm(uav_positions[i] - ue_positions[j])
                
                # Free space path loss (simplified)
                wavelength = 3e8 / self.frequency
                if distance < 1.0:
                    distance = 1.0
                
                # Free space path loss in dB
                fspl_db = 20 * np.log10(distance) + 20 * np.log10(self.frequency) - 147.55
                
                # Additional losses based on scene
                if self.scene == 'suburban':
                    additional_loss = 3.0  # dB
                elif self.scene == 'urban':
                    additional_loss = 5.0  # dB
                else:
                    additional_loss = 1.0  # dB
                
                # Total path loss
                path_loss_db = fspl_db + additional_loss
                
                # Channel gain in dB
                gain_db = -path_loss_db
                
                # Convert to linear scale
                gain_linear = 10 ** (gain_db / 10.0)
                gain_matrix[i, j] = gain_linear
        
        return gain_matrix


class G2AScheduler:
    """Ground-to-Air Scheduler"""
    
    def __init__(self, config: Config, channel: G2AChannel):
        self.config = config
        self.channel = channel
        self.num_ues = config.num_ues
        self.num_uavs = config.num_uavs
        self.bandwidth = config.G2A_bandwidth
        self.n_rbs = config.n_rbs
        self.rb_bandwidth = self.bandwidth / self.n_rbs if self.n_rbs > 0 else self.bandwidth
        self.history = []
        
    def get_default_assignment(self, ue_positions, uav_positions):
        """
        Get default assignment (e.g., round-robin or closest UAV)
        
        Returns:
            assignment: shape (num_uavs, num_ues) - boolean assignment matrix
        """
        num_uavs = len(uav_positions)
        num_ues = len(ue_positions)
        assignment = np.zeros((num_uavs, num_ues), dtype=bool)
        
        # Assign each UE to the closest UAV
        for ue_idx in range(num_ues):
            distances = []
            for uav_idx in range(num_uavs):
                dist = np.linalg.norm(uav_positions[uav_idx] - ue_positions[ue_idx])
                distances.append(dist)
            
            closest_uav = np.argmin(distances)
            assignment[closest_uav, ue_idx] = True
        
        return assignment
    
    def get_rates(self, ue_positions, uav_positions, assignment, 
                  gain_matrix=None, time_slot=0, tx_power=None):
        """
        Calculate transmission rates for UEs based on assignment
        
        Args:
            ue_positions: UE positions
            uav_positions: UAV positions
            assignment: shape (num_uavs, num_ues) - boolean assignment matrix
            gain_matrix: precomputed channel gain matrix (optional)
            time_slot: current time slot (optional)
            tx_power: transmit power (optional)
            
        Returns:
            rates: shape (num_ues,) - transmission rates in bps
        """
        num_ues = len(ue_positions)
        rates = np.zeros(num_ues, dtype=np.float32)
        
        if tx_power is None:
            tx_power = self.config.tx_power
        
        noise_power = self.config.N * self.bandwidth
        
        # Get channel gains
        if gain_matrix is None:
            gain_matrix = self.channel.estimate_chan_gain(ue_positions, uav_positions)
        
        # Calculate rates for each UE
        for ue_idx in range(num_ues):
            # Find which UAV this UE is assigned to
            assigned_uavs = np.where(assignment[:, ue_idx])[0]
            
            if len(assigned_uavs) == 0:
                continue
            
            uav_idx = assigned_uavs[0]
            
            # Calculate received power
            received_power = tx_power * gain_matrix[uav_idx, ue_idx]
            
            # Calculate SNR
            snr = received_power / noise_power
            
            # Shannon capacity (with some practical efficiency factor)
            efficiency = 0.8  # Practical efficiency
            rate = efficiency * self.bandwidth * np.log2(1 + snr)
            
            rates[ue_idx] = max(0, rate)
        
        return rates
    
    def scheduler(self, ue_positions, uav_positions):
        """
        Run scheduler algorithm (e.g., DAS, RR, Random)
        
        Returns:
            assignment: shape (num_uavs, num_ues) - boolean assignment matrix
        """
        # Simple default: assign to closest UAV
        return self.get_default_assignment(ue_positions, uav_positions)


class A2SChannel:
    """Air-to-Satellite Channel Model"""
    
    def __init__(self, config: Config):
        self.config = config
        self.bandwidth = config.A2S_bandwidth
        self.satellite_altitude = config.satellite_altitude
        self.G_T_db = config.G_T_db
        self.lambda_rician_db = config.lambda_rician_db
        self.f_c = config.f_c_s
        self.G_m_l_db = config.G_m_l_db
        self.T_N = config.T_N
        
    def calculate_transmission_rate(self, tx_power, bandwidth, distance, elevation):
        """
        Calculate A2S transmission rate
        
        Args:
            tx_power: transmit power (linear scale)
            bandwidth: bandwidth in Hz
            distance: distance to satellite in meters
            elevation: elevation angle in radians
            
        Returns:
            rates: transmission rates for each UAV
        """
        # Free space path loss
        wavelength = 3e8 / self.f_c
        fspl_db = 20 * np.log10(distance) + 20 * np.log10(self.f_c) - 147.55
        
        # Antenna gain
        G_r_db = self.G_T_db  # Receiver antenna gain
        
        # Total gain in dB
        total_gain_db = self.G_m_l_db + G_r_db - fspl_db
        
        # Convert to linear
        total_gain = 10 ** (total_gain_db / 10.0)
        
        # Received power
        rx_power = tx_power * total_gain
        
        # Noise power
        k_boltzmann = 1.38e-23  # Boltzmann constant
        noise_power = k_boltzmann * self.T_N * bandwidth
        
        # SNR
        snr = rx_power / noise_power
        
        # Shannon capacity
        rate = bandwidth * np.log2(1 + snr)
        
        # If input is scalar, return scalar; if array, return array
        if np.isscalar(tx_power):
            return max(0, rate)
        else:
            return np.maximum(0, rate)


class LEOSatelliteMobility:
    """LEO Satellite Mobility Model"""
    
    def __init__(self, config: Config):
        self.config = config
        self.R_earth = config.R_earth
        self.satellite_altitude = config.satellite_altitude
        self.sat_speed = config.sat_speed
        self.num_sats = config.num_sats
        self.min_elevation_rad = np.deg2rad(config.min_elevation_deg)
        
        # Initial satellite position (simplified circular orbit)
        self.orbit_radius = self.R_earth + self.satellite_altitude
        self.initial_angle = 0.0
        
    def get_distance_and_elevation(self, current_time):
        """
        Get distance and elevation angle to satellite at current time
        
        Args:
            current_time: current simulation time in seconds
            
        Returns:
            distance: distance to satellite in meters
            elevation: elevation angle in radians
        """
        # Angular velocity (radians per second)
        angular_velocity = self.sat_speed / self.orbit_radius
        
        # Current angle
        current_angle = self.initial_angle + angular_velocity * current_time
        
        # Satellite position in orbital plane (simplified 2D model)
        sat_x = self.orbit_radius * np.cos(current_angle)
        sat_y = 0.0
        sat_z = self.orbit_radius * np.sin(current_angle)
        
        # Ground station position (assume at origin on Earth surface)
        gs_x, gs_y, gs_z = 0.0, 0.0, self.R_earth
        
        # Distance
        distance = np.sqrt((sat_x - gs_x)**2 + (sat_y - gs_y)**2 + (sat_z - gs_z)**2)
        
        # Elevation angle (simplified calculation)
        # Angle from ground station to satellite
        dx = sat_x - gs_x
        dy = sat_y - gs_y
        dz = sat_z - gs_z
        
        horizontal_dist = np.sqrt(dx**2 + dy**2)
        elevation = np.arctan2(dz, horizontal_dist)
        
        # Ensure minimum elevation
        if elevation < self.min_elevation_rad:
            elevation = self.min_elevation_rad
        
        return distance, elevation
