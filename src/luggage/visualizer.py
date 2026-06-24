import cv2
import numpy as np

from src.luggage.luggage_monitor import BagState

FONT = cv2.FONT_HERSHEY_SIMPLEX

C_BAG = (0, 255, 255)
C_WARN = (0, 140, 255)
C_ALERT = (0, 0, 255)
C_CARRY = (255, 200, 80)
C_PERSON = (0, 255, 0)
C_PERSON_NEAR = (120, 255, 120)
C_HIDDEN = (200, 200, 200)
C_TEXT = (235, 235, 235)


class Visualizer:
    def __init__(self, monitor, n_circle_pts=72):
        self.mon = monitor
        angles = np.linspace(0, 2 * np.pi, n_circle_pts, endpoint=False)
        self._unit_circle = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    # ---------- helpers ----------

    def _state_color(self, st, carrier):
        if st.state == BagState.ABANDONED:
            return C_ALERT
        if st.state == BagState.UNATTENDED:
            return C_WARN
        if st.state == BagState.MOVING and carrier is not None:
            return C_CARRY
        return C_BAG

    def _radius_polygon(self, center_world):
        world_pts = center_world + self.mon.radius_m * self._unit_circle
        return self.mon.to_image(world_pts).astype(np.int32)

    def _label(self, frame, text, x, y, color):
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.45, 1)
        cv2.rectangle(frame, (x, y - th - 10), (x + tw + 10, y), (25, 25, 25), -1)
        cv2.rectangle(frame, (x, y - th - 10), (x + tw + 10, y), color, 1)
        cv2.putText(frame, text, (x + 5, y - 5), FONT, 0.45, color, 1, cv2.LINE_AA)

    # ---------- rendu principal ----------

    def draw(self, frame, bags, persons, statuses, fps=None):
        person_pts = {p["tid"]: p["point"] for p in persons}
        near_ids = set()

        # Porteur (affichage) : seulement pour les sacs MOVING
        carriers = {}
        for b in bags:
            st = statuses[b["tid"]]
            near_ids.update(st.nearby_ids)
            if st.state == BagState.MOVING:
                carriers[b["tid"]] = self.mon.find_carrier(b["box"], persons)
            else:
                carriers[b["tid"]] = None

        # 1) Cercles des bagages poses (non MOVING) sur un seul overlay
        polygons = []
        for b in bags:
            st = statuses[b["tid"]]
            if st.state != BagState.MOVING and st.anchor_world is not None:
                polygons.append((self._radius_polygon(st.anchor_world),
                                 self._state_color(st, None)))

        if polygons:
            overlay = frame.copy()
            for poly, color in polygons:
                cv2.fillPoly(overlay, [poly], color)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            for poly, color in polygons:
                cv2.polylines(frame, [poly], True, color, 2, cv2.LINE_AA)

        # 2) Bagages
        for b in bags:
            st = statuses[b["tid"]]
            carrier = carriers[b["tid"]]
            color = self._state_color(st, carrier)
            x1, y1, x2, y2 = b["box"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color,
                          2 if st.state != BagState.MOVING else 1)

            # Point au sol : sur l'ancre quand la position est epinglée
            if st.occluded and st.anchor_world is not None:
                ax, ay = self.mon.to_image(st.anchor_world.reshape(1, 2))[0]
                cv2.circle(frame, (int(ax), int(ay)), 4, color, -1, cv2.LINE_AA)
                cv2.circle(frame, (int(ax), int(ay)), 8, C_HIDDEN, 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, b["point"], 4, color, -1, cv2.LINE_AA)

            # Label : classe + ID, timer pendant le décompte
            cat = b.get("category", "bagage").upper()
            if st.state == BagState.UNATTENDED:
                label = f"{cat} ID:{st.track_id}  {st.timer_seconds(self.mon.fps):.1f}s"
            else:
                label = f"{cat} ID:{st.track_id}"
            self._label(frame, label, x1, max(28, y1 - 4), color)

            # Barre de progression vers ABANDONNE
            if st.state == BagState.UNATTENDED:
                ratio = min(1.0, st.unattended_frames
                            / self.mon.abandoned_frames_req)
                cv2.rectangle(frame, (x1, y2 + 4), (x2, y2 + 9), (60, 60, 60), -1)
                cv2.rectangle(frame, (x1, y2 + 4),
                              (x1 + int((x2 - x1) * ratio), y2 + 9), color, -1)

            # SEUL trait : point bas du sac MOVING vers son porteur
            if carrier is not None and carrier in person_pts:
                cv2.line(frame, b["point"], person_pts[carrier],
                         C_CARRY, 2, cv2.LINE_AA)

        # 3) Personnes — l'anneau vert marque l'entrée dans une zone
        for p in persons:
            tid = p["tid"]
            near = tid in near_ids
            color = C_PERSON_NEAR if near else C_PERSON
            x1, y1, x2, y2 = p["box"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            cv2.circle(frame, p["point"], 5, color, -1, cv2.LINE_AA)
            if near:
                cv2.circle(frame, p["point"], 10, color, 2, cv2.LINE_AA)
            self._label(frame, f"PERSONNE ID:{tid}", x1, max(28, y1 - 4), color)

        self._panel(frame, len(persons), len(bags), statuses, fps)
        self._alert_banner(frame, statuses)
        return frame

    # ---------- panneau & bandeau ----------

    def _panel(self, frame, n_persons, n_bags, statuses, fps):
        n_alert = sum(1 for s in statuses.values()
                      if s.state == BagState.ABANDONED)
        x, y, w, h = 15, 15, 330, 92

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (90, 90, 90), 1)

        cv2.putText(frame, "LUGGAGE MONITOR", (x + 12, y + 26),
                    FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        if fps is not None:
            cv2.putText(frame, f"{fps:.0f} FPS", (x + w - 80, y + 26),
                        FONT, 0.5, (160, 220, 160), 1, cv2.LINE_AA)
        cv2.putText(frame, f"PERSONNES {n_persons}", (x + 12, y + 56),
                    FONT, 0.48, C_TEXT, 1, cv2.LINE_AA)
        cv2.putText(frame, f"BAGAGES {n_bags}", (x + 170, y + 56),
                    FONT, 0.48, C_TEXT, 1, cv2.LINE_AA)
        cv2.putText(frame, f"ALERTES {n_alert}", (x + 12, y + 80),
                    FONT, 0.48, C_ALERT if n_alert else C_TEXT, 1, cv2.LINE_AA)

    def _alert_banner(self, frame, statuses):
        ids = [s.track_id for s in statuses.values()
               if s.state == BagState.ABANDONED]
        if not ids:
            return
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 42), (w, h), (0, 0, 180), -1)
        txt = "ALERTE - BAGAGE ABANDONNE : ID " + ", ".join(map(str, ids))
        cv2.putText(frame, txt, (16, h - 14), FONT, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
