import cv2
import numpy as np

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class BagState(Enum):
    MOVING = "moving"
    STOPPED = "stopped"
    UNATTENDED = "unattended"
    ABANDONED = "abandoned"


@dataclass
class BagStatus:
    track_id: int
    state: BagState = BagState.MOVING
    anchor_world: Optional[np.ndarray] = None
    stop_frames: int = 0
    unattended_frames: int = 0
    move_candidate_frames: int = 0
    occluded: bool = False
    carried: bool = False
    owner_id: Optional[int] = None
    nearby_ids: tuple = ()
    last_zone_frame: int = -10**9
    last_seen: int = 0

    def timer_seconds(self, fps: float) -> float:
        return self.unattended_frames / fps


class LuggageMonitor:
    """Machine à états par track pour détecter les bagages abandonnés.

    La logique métier se base sur la position au sol du bagage, projetée par
    homographie, et sur la présence ou non de personnes à proximité.

    Un cas particulier est traité séparément : le bagage porté. Il est détecté
    à partir de son élévation dans l'image, par exemple lorsqu'un sac est porté
    à la main ou sur le dos.

    La proximité latérale avec une personne n'est utilisée que pour l'affichage,
    notamment pour tracer un lien visuel avec une valise tractée. Elle ne doit
    pas modifier l'état métier du bagage.
    """

    def __init__(self, homography, fps,
                 radius_m=1.0,
                 stop_seconds=2.0,
                 abandoned_seconds=10.0,
                 move_threshold_m=0.15,
                 move_confirm_seconds=0.4,
                 zone_grace_seconds=1.0,
                 carried_elevation=0.15,
                 track_buffer_frames=90):
        self.H = np.asarray(homography, dtype=np.float64)
        self.H_inv = np.linalg.inv(self.H)
        self.fps = float(fps)
        self.radius_m = float(radius_m)
        self.move_threshold_m = float(move_threshold_m)
        self.carried_elevation = float(carried_elevation)

        self.stop_frames_req = max(1, int(stop_seconds * self.fps))
        self.abandoned_frames_req = max(1, int(abandoned_seconds * self.fps))
        self.move_confirm_req = max(1, int(move_confirm_seconds * self.fps))
        self.zone_grace_req = max(1, int(zone_grace_seconds * self.fps))
        self.track_buffer = int(track_buffer_frames)

        self.bags: dict[int, BagStatus] = {}
        self.frame_idx = 0

    # ---------- géométrie ----------

    def to_world(self, points) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        if pts.size == 0:
            return np.empty((0, 2))
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def to_image(self, world_points) -> np.ndarray:
        pts = np.asarray(world_points, dtype=np.float64).reshape(-1, 1, 2)
        if pts.size == 0:
            return np.empty((0, 2))
        return cv2.perspectiveTransform(pts, self.H_inv).reshape(-1, 2)

    # ---------- détection du porteur ----------

    def _is_elevated(self, bag_box, persons):
        """Détermine si un bagage semble être porté.

        Le bas du bagage doit être situé significativement au-dessus du niveau
        des pieds de la personne, avec un recouvrement vertical cohérent.
        """
        bx1, by1, bx2, by2 = bag_box
        bcx = 0.5 * (bx1 + bx2)

        for p in persons:
            px1, py1, px2, py2 = p["box"]

            if by2 <= py1 or by1 >= py2:
                continue

            person_h = max(1.0, py2 - py1)

            if px1 <= bcx <= px2 and by2 < py2 - self.carried_elevation * person_h:
                return True

        return False

    def find_carrier(self, bag_box, persons):
        """Retourne l'identifiant du porteur probable.

        Cette méthode sert uniquement à l'affichage :
        - sac porté détecté par élévation ;
        - valise tractée proche d'une personne.

        Elle ne doit pas être utilisée pour modifier l'état métier du bagage.
        """
        bx1, by1, bx2, by2 = bag_box
        bcx = 0.5 * (bx1 + bx2)

        for p in persons:
            px1, py1, px2, py2 = p["box"]

            if by2 <= py1 or by1 >= py2:
                continue

            person_h = max(1.0, py2 - py1)
            person_w = max(1.0, px2 - px1)
            pcx = 0.5 * (px1 + px2)

            if px1 <= bcx <= px2 and by2 < py2 - self.carried_elevation * person_h:
                return p["tid"]

            if (
                abs(by2 - py2) <= 0.10 * person_h
                and abs(bcx - pcx) <= 0.35 * person_w
            ):
                return p["tid"]

        return None

    # ---------- helpers ----------

    @staticmethod
    def _reset(st: BagStatus, world: np.ndarray):
        """Réinitialise l'état autour d'une nouvelle position de référence."""
        st.anchor_world = world.copy()
        st.stop_frames = 0
        st.unattended_frames = 0
        st.move_candidate_frames = 0
        st.owner_id = None

    # ---------- mise à jour ----------

    def update(self, bags, persons) -> dict[int, BagStatus]:
        """
        bags:
            liste de dicts {"tid", "box": (x1, y1, x2, y2), "point": (cx, cy)}

        persons:
            liste de dicts {"tid", "box": (x1, y1, x2, y2), "point": (cx, cy)}
        """
        self.frame_idx += 1

        bag_world = self.to_world([b["point"] for b in bags])
        person_ids = np.array([p["tid"] for p in persons], dtype=int)
        person_world = self.to_world([p["point"] for p in persons])

        out = {}
        n_persons = len(person_world)

        for i, b in enumerate(bags):
            tid = b["tid"]
            world = bag_world[i]

            st = self.bags.get(tid)

            if st is None:
                st = BagStatus(track_id=tid)
                self.bags[tid] = st

            st.last_seen = self.frame_idx

            if st.anchor_world is None:
                self._reset(st, world)

            # Recherche des personnes proches de la position de référence.
            if n_persons:
                dists = np.linalg.norm(person_world - st.anchor_world, axis=1)
                near_mask = dists <= self.radius_m
                st.nearby_ids = tuple(int(p) for p in person_ids[near_mask])
            else:
                dists = None
                st.nearby_ids = ()

            person_in_zone = len(st.nearby_ids) > 0

            if person_in_zone:
                st.last_zone_frame = self.frame_idx

            person_recent = (
                self.frame_idx - st.last_zone_frame <= self.zone_grace_req
            )
            established = st.stop_frames >= self.stop_frames_req

            # Un bagage porté repart immédiatement en état MOVING.
            elevated = self._is_elevated(b["box"], persons)
            st.carried = elevated and (not established or person_recent)

            if st.carried:
                self._reset(st, world)
                st.occluded = False
                st.nearby_ids = ()
                st.state = BagState.MOVING
                out[tid] = st
                continue

            displaced = (
                np.linalg.norm(world - st.anchor_world) > self.move_threshold_m
            )

            # Gestion du mouvement au sol.
            if displaced:
                if established and not person_recent:
                    # Un bagage déjà posé ne doit pas "sauter" tout seul si
                    # personne n'est passé à proximité : on garde l'ancre.
                    st.move_candidate_frames = 0
                    st.stop_frames += 1
                    st.occluded = True
                else:
                    # Le déplacement est validé seulement après quelques frames.
                    st.move_candidate_frames += 1

                    if st.move_candidate_frames >= self.move_confirm_req:
                        self._reset(st, world)

                    st.occluded = False
            else:
                st.move_candidate_frames = 0
                st.stop_frames += 1
                st.occluded = False

            stopped = st.stop_frames >= self.stop_frames_req

            # Première attribution du propriétaire : personne la plus proche
            # lorsque le bagage devient immobile.
            if stopped and st.owner_id is None and dists is not None:
                j = int(np.argmin(dists))

                if dists[j] <= self.radius_m:
                    st.owner_id = int(person_ids[j])

            # Temps passé sans personne à proximité.
            if stopped and not person_in_zone:
                st.unattended_frames += 1
            else:
                st.unattended_frames = 0

            if st.unattended_frames >= self.abandoned_frames_req:
                st.state = BagState.ABANDONED
            elif st.unattended_frames > 0:
                st.state = BagState.UNATTENDED
            elif stopped:
                st.state = BagState.STOPPED
            else:
                st.state = BagState.MOVING

            out[tid] = st

        # Suppression des bagages non observés depuis trop longtemps.
        stale = [
            t for t, s in self.bags.items()
            if self.frame_idx - s.last_seen > self.track_buffer
        ]

        for t in stale:
            del self.bags[t]

        return out
