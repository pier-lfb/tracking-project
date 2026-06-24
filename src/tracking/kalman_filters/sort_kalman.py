from src.tracking.kalman_filters.base_kalman import BaseKalman
import numpy as np


class SORTKalman(BaseKalman):
    def __init__(self):
        state_dim = 7
        observation_dim = 4
        F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])
        H = np.eye(state_dim // 2 + 1, state_dim)
        super().__init__(state_dim=state_dim, observation_dim=observation_dim, F=F, H=H)
        self.kf.R[2:, 2:] *= 10
        self.kf.P[4:, 4:] *= 1000
        self.kf.P *= 10
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

    def initialize(self, observation):
        self.kf.x = self.kf.x.flatten()
        self.kf.x[:4] = observation

    def predict(self):
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()

    def update(self, z):
        self.kf.update(z)
