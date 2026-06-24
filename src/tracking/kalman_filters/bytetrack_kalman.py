from src.tracking.kalman_filters.base_kalman import BaseKalman
import numpy as np


class ByteKalman(BaseKalman):
    def __init__(self):
        state_dim = 8
        observation_dim = 4
        F = np.eye(state_dim)
        for i in range(state_dim // 2):
            F[i, i + state_dim // 2] = 1
        H = np.eye(state_dim // 2, state_dim)
        super().__init__(state_dim=state_dim, observation_dim=observation_dim, F=F, H=H)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initialize(self, observation):
        self.kf.x = np.r_[observation, np.zeros_like(observation)]
        std = [
            2 * self._std_weight_position * observation[3],
            2 * self._std_weight_position * observation[3],
            1e-2,
            2 * self._std_weight_position * observation[3],
            10 * self._std_weight_velocity * observation[3],
            10 * self._std_weight_velocity * observation[3],
            1e-5,
            10 * self._std_weight_velocity * observation[3],
        ]
        self.kf.P = np.diag(np.square(std))

    def predict(self):
        std_pos = [
            self._std_weight_position * self.kf.x[3],
            self._std_weight_position * self.kf.x[3],
            1e-2,
            self._std_weight_position * self.kf.x[3],
        ]
        std_vel = [
            self._std_weight_velocity * self.kf.x[3],
            self._std_weight_velocity * self.kf.x[3],
            1e-5,
            self._std_weight_velocity * self.kf.x[3],
        ]
        self.kf.predict(Q=np.diag(np.square(np.r_[std_pos, std_vel])))

    def update(self, z):
        std = [
            self._std_weight_position * self.kf.x[3],
            self._std_weight_position * self.kf.x[3],
            1e-1,
            self._std_weight_position * self.kf.x[3],
        ]
        self.kf.update(z=z, R=np.diag(np.square(std)))
