# src/trafic/vehicule_counter.py
import numpy as np


class VehicleCounter:
    """Compte les véhicules lorsqu'ils franchissent une ligne.

    Un offset évite les doubles comptages quand un point oscille près de la
    ligne. Un cooldown limite aussi les recomptages trop rapprochés pour un
    même track.
    """

    def __init__(self, p1, p2, offset=15, cooldown=20, stale_frames=90):
        self.p1 = np.array(p1, dtype=np.float32)
        self.p2 = np.array(p2, dtype=np.float32)
        self.offset = float(offset)
        self.cooldown = int(cooldown)
        self.stale_frames = int(stale_frames)

        self.last_side = {}
        self.last_count_frame = {}
        self.last_seen = {}

        self.count_left = 0
        self.count_right = 0

    def _signed_distance(self, point):
        point = np.asarray(point, dtype=np.float32)
        line = self.p2 - self.p1
        rel = point - self.p1
        line_len = np.linalg.norm(line)
        if line_len < 1e-6:
            return 0.0
        return (line[0] * rel[1] - line[1] * rel[0]) / line_len

    def _side(self, point):
        d = self._signed_distance(point)
        if d > self.offset:
            return 1
        if d < -self.offset:
            return -1
        return 0

    def update(self, track_id, point, frame_id=0):
        """Retourne le sens du franchissement, ou None si rien n'est compté."""
        self.last_seen[track_id] = frame_id

        side = self._side(point)
        if side == 0:
            return None

        old_side = self.last_side.get(track_id)
        self.last_side[track_id] = side

        if old_side is None or old_side == side:
            return None

        last_frame = self.last_count_frame.get(track_id, -10**9)
        if frame_id - last_frame < self.cooldown:
            return None
        self.last_count_frame[track_id] = frame_id

        if old_side == -1 and side == 1:
            self.count_left += 1
            return "left"
        if old_side == 1 and side == -1:
            self.count_right += 1
            return "right"
        return None

    def get_counts(self):
        return {
            "left": self.count_left,
            "right": self.count_right,
            "total": self.count_left + self.count_right,
        }

    def prune(self, frame_id):
        stale = [tid for tid, f in self.last_seen.items()
                 if frame_id - f > self.stale_frames]
        for tid in stale:
            self.last_side.pop(tid, None)
            self.last_count_frame.pop(tid, None)
            self.last_seen.pop(tid, None)
