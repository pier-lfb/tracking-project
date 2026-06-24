# src/trafic/speed_estimator.py
import numpy as np
from collections import defaultdict, deque
from scipy.signal import savgol_filter


class SpeedEstimator:
    """Estime la vitesse d'un track après projection par homographie.

    Les positions image sont converties en coordonnées monde, puis la vitesse
    est calculée sur une trajectoire lissée. Les mesures trop loin de la caméra
    peuvent être ignorées car le bruit de détection y devient trop important.
    """

    def __init__(
        self,
        homography,
        fps=25,
        window=11,
        polyorder=2,
        ema_alpha=0.2,
        max_history=60,
        min_points=8,
        speed_span=5,
        min_speed_kmh=1.0,
        max_speed_kmh=180.0,
        min_valid_y_px=0,
        stale_frames=90,
    ):
        self.H = np.asarray(homography, dtype=np.float64)
        self.fps = float(fps) if fps and fps > 0 else 25.0

        # Savitzky-Golay nécessite une fenêtre impaire avec polyorder < window.
        self.window = int(window) | 1
        self.polyorder = min(int(polyorder), self.window - 1)

        self.ema_alpha = float(ema_alpha)
        self.max_history = int(max_history)
        self.min_points = max(2, int(min_points))
        self.speed_span = max(1, int(speed_span))
        self.min_speed_kmh = float(min_speed_kmh)
        self.max_speed_kmh = float(max_speed_kmh)
        self.min_valid_y_px = float(min_valid_y_px)
        self.stale_frames = int(stale_frames)

        # Nombre de points utiles pour lisser puis mesurer le déplacement.
        self._tail_len = self.window + self.speed_span

        self.positions = defaultdict(lambda: deque(maxlen=self.max_history))
        self.ema_speeds = {}
        self.last_seen = {}

    # ---------- interne ----------

    def _to_world(self, cx, cy):
        pt = self.H @ np.array([float(cx), float(cy), 1.0], dtype=np.float64)
        if abs(pt[2]) < 1e-9:
            return None
        return pt[0] / pt[2], pt[1] / pt[2]

    def _smooth(self, pts):
        arr = np.asarray(pts, dtype=np.float64)
        n = len(arr)
        if n < 3:
            return arr
        window = min(self.window, n) | 1
        if window > n:
            window -= 2
        if window <= self.polyorder:
            return arr
        return savgol_filter(arr, window_length=window,
                             polyorder=self.polyorder, axis=0, mode="interp")

    # ---------- API ----------

    def update(self, track_id, cx, cy, frame_id=0):
        """Retourne (wx, wy, speed_kmh).

        La vitesse vaut None pendant le warm-up ou hors de la zone de mesure.
        """
        self.last_seen[track_id] = frame_id

        if cy < self.min_valid_y_px:
            self.positions.pop(track_id, None)
            self.ema_speeds.pop(track_id, None)
            world = self._to_world(cx, cy)
            if world is None:
                return None, None, None
            return world[0], world[1], None

        world = self._to_world(cx, cy)

        if world is None:
            return None, None, self.ema_speeds.get(track_id)

        wx, wy = world

        self.positions[track_id].append((wx, wy))

        pts = self.positions[track_id]

        if len(pts) < self.min_points:
            return wx, wy, None

        tail = list(pts)[-self._tail_len:]
        smoothed = self._smooth(tail)

        k = min(self.speed_span, len(smoothed) - 1)
        delta = smoothed[-1] - smoothed[-1 - k]
        speed_kmh = (np.linalg.norm(delta) / k) * self.fps * 3.6

        if speed_kmh < self.min_speed_kmh:
            speed_kmh = 0.0
        elif speed_kmh > self.max_speed_kmh:
            speed_kmh = self.ema_speeds.get(track_id, self.max_speed_kmh)

        prev = self.ema_speeds.get(track_id)
        self.ema_speeds[track_id] = speed_kmh if prev is None else \
            self.ema_alpha * speed_kmh + (1.0 - self.ema_alpha) * prev

        return wx, wy, self.ema_speeds[track_id]

    def get_speed(self, track_id):
        return self.ema_speeds.get(track_id)

    def get_smoothed_trajectory(self, track_id):
        pts = self.positions.get(track_id)
        if not pts:
            return np.empty((0, 2), dtype=np.float64)
        return self._smooth(pts)

    def prune(self, frame_id):
        stale = [tid for tid, f in self.last_seen.items()
                 if frame_id - f > self.stale_frames]
        for tid in stale:
            self.positions.pop(tid, None)
            self.ema_speeds.pop(tid, None)
            self.last_seen.pop(tid, None)
