from filterpy.kalman import KalmanFilter
import numpy as np
import scipy


class BaseKalman:
    def __init__(self, state_dim=8, observation_dim=4,
                 F=np.zeros((0,)), P=np.zeros((0,)), Q=np.zeros((0,)),
                 H=np.zeros((0,)), R=np.zeros((0,))) -> None:
        self.kf = KalmanFilter(dim_x=state_dim, dim_z=observation_dim, dim_u=0)
        if F.shape[0] > 0: self.kf.F = F
        if P.shape[0] > 0: self.kf.P = P
        if Q.shape[0] > 0: self.kf.Q = Q
        if H.shape[0] > 0: self.kf.H = H
        if R.shape[0] > 0: self.kf.R = R

    def initialize(self, observation):
        raise NotImplementedError

    def predict(self):
        self.kf.predict()

    def update(self, observation, **kwargs):
        self.kf.update(observation)

    def get_state(self):
        return self.kf.x

    def gating_distance(self, measurements, only_position=False):
        mean = np.dot(self.kf.H, self.kf.x.copy())
        covariance = np.linalg.multi_dot((self.kf.H, self.kf.P, self.kf.H.T))
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
        return np.sum(z * z, axis=0)
