from src.tracking.kalman_filters.base_kalman import BaseKalman
import numpy as np
from copy import deepcopy


class OCSORTKalman(BaseKalman):
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
        self.history_obs = []
        self.attr_saved = None
        self.observed = False

    def initialize(self, observation):
        self.kf.x = self.kf.x.flatten()
        self.kf.x[:4] = observation

    def predict(self):
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()

    def _freeze(self):
        """Sauvegarde l'état du filtre au moment où le track devient non observé."""
        self.attr_saved = deepcopy(self.kf.__dict__)

    def _unfreeze(self):
        """ORU : restaure l'état gelé puis rejoue une trajectoire virtuelle
        linéaire entre les deux dernières observations réelles."""
        if self.attr_saved is None:
            return

        new_history = deepcopy(self.history_obs)
        self.kf.__dict__ = self.attr_saved
        self.history_obs = self.history_obs[:-1]

        occur = [int(d is None) for d in new_history]
        indices = np.where(np.array(occur) == 0)[0]

        # Garde-fou : il faut au moins 2 observations réelles pour interpoler
        if len(indices) < 2:
            return

        index1, index2 = indices[-2], indices[-1]
        x1, y1, s1, r1 = new_history[index1]
        w1, h1 = np.sqrt(s1 * r1), np.sqrt(s1 / r1)
        x2, y2, s2, r2 = new_history[index2]
        w2, h2 = np.sqrt(s2 * r2), np.sqrt(s2 / r2)

        time_gap = index2 - index1
        dx, dy = (x2 - x1) / time_gap, (y2 - y1) / time_gap
        dw, dh = (w2 - w1) / time_gap, (h2 - h1) / time_gap

        for i in range(index2 - index1):
            x = x1 + (i + 1) * dx
            y = y1 + (i + 1) * dy
            w = w1 + (i + 1) * dw
            h = h1 + (i + 1) * dh
            s, r = w * h, w / float(h)
            self.kf.update(np.array([x, y, s, r]).reshape((4, 1)))
            if i != index2 - index1 - 1:
                self.kf.predict()

    def update(self, z):
        self.history_obs.append(z)
        if z is None:
            # Track non observé cette frame : on gèle l'état à la transition
            if self.observed:
                self._freeze()
            self.observed = False
            self.kf.update(z)
        else:
            # Réacquisition apres une période non observée : ORU
            if not self.observed:
                self._unfreeze()
            self.observed = True
            self.kf.update(z)
