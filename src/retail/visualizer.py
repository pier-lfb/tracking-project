# src/retail/visualizer.py
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

C_ZONE = (60, 180, 255)
C_PERSON = (0, 255, 0)
C_IN_ZONE = (60, 180, 255)
C_TEXT = (235, 235, 235)
C_PANEL = (18, 18, 18)


class RetailVisualizer:
    def __init__(self, monitor):
        self.mon = monitor

    def _label(self, frame, text, x, y, color):
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.45, 1)
        cv2.rectangle(frame, (x, y - th - 10), (x + tw + 10, y), (25, 25, 25), -1)
        cv2.rectangle(frame, (x, y - th - 10), (x + tw + 10, y), color, 1)
        cv2.putText(frame, text, (x + 5, y - 5), FONT, 0.45, color, 1, cv2.LINE_AA)

    def draw(self, frame, persons, statuses, fps=None):
        # Zone d'intérêt affichée en transparence.
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.mon.polygon], C_ZONE)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
        cv2.polylines(frame, [self.mon.polygon], True, C_ZONE, 2, cv2.LINE_AA)

        # Personnes détectées.
        for p in persons:
            st = statuses[p["tid"]]
            color = C_IN_ZONE if st.in_zone else C_PERSON
            x1, y1, x2, y2 = p["box"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color,
                          2 if st.in_zone else 1)
            cv2.circle(frame, p["point"], 5, color, -1, cv2.LINE_AA)

            # Affiche le temps cumulé dès que la personne est passée en zone.
            if st.zone_frames > 0:
                label = f"ID {st.track_id}  {st.zone_seconds(self.mon.fps):.1f}s"
            else:
                label = f"ID {st.track_id}"
            self._label(frame, label, x1, max(28, y1 - 4), color)

        self._panel(frame, len(persons), statuses, fps)
        return frame

    def _panel(self, frame, n_total, statuses, fps):
        in_zone = [s for s in statuses.values() if s.in_zone]
        x, y, w = 15, 15, 330
        h = 92 + 22 * min(len(in_zone), 6)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), C_PANEL, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (90, 90, 90), 1)

        cv2.putText(frame, "RETAIL MONITOR", (x + 12, y + 26),
                    FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        if fps is not None:
            cv2.putText(frame, f"{fps:.0f} FPS", (x + w - 80, y + 26),
                        FONT, 0.5, (160, 220, 160), 1, cv2.LINE_AA)
        cv2.putText(frame, f"PERSONNES {n_total}", (x + 12, y + 56),
                    FONT, 0.48, C_TEXT, 1, cv2.LINE_AA)
        cv2.putText(frame, f"EN ZONE {len(in_zone)}", (x + 170, y + 56),
                    FONT, 0.48, C_IN_ZONE if in_zone else C_TEXT, 1, cv2.LINE_AA)

        # Occupants affichés par temps de présence décroissant.
        in_zone.sort(key=lambda s: -s.zone_frames)
        for i, s in enumerate(in_zone[:6]):
            txt = f"ID {s.track_id}   {s.zone_seconds(self.mon.fps):.1f}s"
            cv2.putText(frame, txt, (x + 12, y + 82 + 22 * i),
                        FONT, 0.45, C_IN_ZONE, 1, cv2.LINE_AA)
