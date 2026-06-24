# src/retail/zone_monitor.py
import json
import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class PersonStatus:
    track_id: int
    in_zone: bool = False
    zone_frames: int = 0
    current_visit_frames: int = 0
    last_seen: int = 0

    def zone_seconds(self, fps: float) -> float:
        return self.zone_frames / fps


class ZoneMonitor:
    """Mesure le temps de présence dans une zone polygonale.

    La zone est définie en pixels avec `tools/zone_drawer.py`.
    Le test d'appartenance se fait sur le point au sol de chaque personne,
    c'est-à-dire le bas-centre de sa boîte englobante.

    Le temps affiché correspond au cumul de toutes les visites d'une même
    identité. Les visites trop courtes sont ignorées pour éviter les faux
    positifs au bord de la zone.
    """

    def __init__(self, zone_file, fps,
                 min_zone_seconds=0.3,
                 track_buffer_frames=90):
        with open(zone_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.polygon = np.array(data["points"], dtype=np.int32)
        if len(self.polygon) < 3:
            raise ValueError(f"Zone invalide dans {zone_file} (>= 3 points requis)")

        self.fps = float(fps)
        self.min_zone_frames = max(1, int(min_zone_seconds * self.fps))
        self.track_buffer = int(track_buffer_frames)

        self.persons: dict[int, PersonStatus] = {}
        self.zone_records: dict[int, int] = {}   # meilleur temps cumulé par ID
        self.frame_idx = 0

    def _inside(self, point) -> bool:
        return cv2.pointPolygonTest(
            self.polygon, (float(point[0]), float(point[1])), False) >= 0

    def update(self, persons) -> dict[int, PersonStatus]:
        """
        persons : liste de dicts {"tid", "point": (cx, cy)}.
        Retourne {tid: PersonStatus} pour les tracks visibles sur la frame.
        """
        self.frame_idx += 1
        out = {}

        for p in persons:
            tid = p["tid"]
            st = self.persons.get(tid)
            if st is None:
                st = PersonStatus(track_id=tid)
                self.persons[tid] = st
            st.last_seen = self.frame_idx

            if self._inside(p["point"]):
                st.current_visit_frames += 1
                # On ne valide une visite qu'après quelques frames consécutives.
                if st.current_visit_frames == self.min_zone_frames:
                    st.zone_frames += self.min_zone_frames
                elif st.current_visit_frames > self.min_zone_frames:
                    st.zone_frames += 1
                st.in_zone = st.current_visit_frames >= self.min_zone_frames
            else:
                st.current_visit_frames = 0
                st.in_zone = False

            out[tid] = st

        # Suppression des tracks qui ne sont plus visibles.
        stale = [t for t, s in self.persons.items()
                 if self.frame_idx - s.last_seen > self.track_buffer]
        for t in stale:
            del self.persons[t]

        # Historique persistant pour garder le classement même après disparition.
        for tid, st in out.items():
            if st.zone_frames > self.zone_records.get(tid, 0):
                self.zone_records[tid] = st.zone_frames

        return out

    def count_in_zone(self, statuses) -> int:
        return sum(1 for s in statuses.values() if s.in_zone)

    def top_dwell(self, n=5):
        """Retourne les n plus longs temps de présence cumulés."""
        ranked = sorted(self.zone_records.items(), key=lambda kv: -kv[1])[:n]
        return [{"id": tid, "seconds": round(frames / self.fps, 1)}
                for tid, frames in ranked]
