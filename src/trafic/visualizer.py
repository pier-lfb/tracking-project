# src/trafic/visualizer.py
import cv2
import numpy as np
from collections import defaultdict, deque

FONT = cv2.FONT_HERSHEY_SIMPLEX

C_OK = (0, 220, 80)
C_SPEEDER = (0, 0, 255)
C_TEXT = (235, 235, 235)
C_PANEL = (18, 18, 18)

# Couleurs du portique de comptage.
C_GATE_FILL = (90, 75, 30)
C_GATE_EDGE = (200, 180, 90)
C_GATE_DASH = (230, 220, 160)
C_PILL_HOT = (120, 255, 120)


def _roi_blend(frame, x1, y1, x2, y2, draw_fn, alpha):
    """Applique un fondu translucide uniquement sur une petite zone."""
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    overlay = roi.copy()
    draw_fn(overlay, -x1, -y1)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)


class TraficVisualizer:
    def __init__(self, speed_limit=95, trail_length=25, pill_flash_frames=14):
        self.speed_limit = float(speed_limit)
        self.trail_length = int(trail_length)
        self.pill_flash_frames = int(pill_flash_frames)
        self.trails = defaultdict(lambda: deque(maxlen=self.trail_length))
        self._trail_seen = {}
        self._last_count_frame = {"left": -10**9, "right": -10**9}
        self._gate_cache = None         # géométrie fixe du portique

    # ---------- helpers ----------

    def _label(self, frame, text, x, y, color):
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.45, 1)
        cv2.rectangle(frame, (x, y - th - 10), (x + tw + 10, y), (25, 25, 25), -1)
        cv2.rectangle(frame, (x, y - th - 10), (x + tw + 10, y), color, 1)
        cv2.putText(frame, text, (x + 5, y - 5), FONT, 0.45, color, 1, cv2.LINE_AA)

    @staticmethod
    def _triangle(frame, center, direction, size, color):
        d = np.asarray(direction, dtype=np.float64)
        d /= max(np.linalg.norm(d), 1e-9)
        p = np.array([-d[1], d[0]])
        c = np.asarray(center, dtype=np.float64)
        pts = np.array([
            c + d * size,
            c - d * size * 0.7 + p * size * 0.8,
            c - d * size * 0.7 - p * size * 0.8,
        ], dtype=np.int32)
        cv2.fillPoly(frame, [pts], color, cv2.LINE_AA)

    def _pill(self, frame, center, count, direction, hot):
        text = str(count)
        cx, cy = int(center[0]), int(center[1])
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.5, 1)
        tri_w = 16
        w, h = tw + tri_w + 26, th + 14
        x1, y1 = cx - w // 2, cy - h // 2
        x2, y2 = x1 + w, y1 + h
        r = h // 2

        fill = (35, 35, 35)
        edge = C_PILL_HOT if hot else (110, 110, 110)
        accent = C_PILL_HOT if hot else (200, 180, 90)
        txt_c = C_PILL_HOT if hot else C_TEXT

        def draw_bg(img, ox, oy):
            cv2.rectangle(img, (x1 + r + ox, y1 + oy), (x2 - r + ox, y2 + oy), fill, -1)
            cv2.circle(img, (x1 + r + ox, cy + oy), r, fill, -1)
            cv2.circle(img, (x2 - r + ox, cy + oy), r, fill, -1)

        _roi_blend(frame, x1, y1, x2 + 1, y2 + 1, draw_bg, 0.85)

        cv2.ellipse(frame, (x1 + r, cy), (r, r), 0, 90, 270, edge, 1, cv2.LINE_AA)
        cv2.ellipse(frame, (x2 - r, cy), (r, r), 0, -90, 90, edge, 1, cv2.LINE_AA)
        cv2.line(frame, (x1 + r, y1), (x2 - r, y1), edge, 1, cv2.LINE_AA)
        cv2.line(frame, (x1 + r, y2), (x2 - r, y2), edge, 1, cv2.LINE_AA)

        self._triangle(frame, (x1 + r + 6, cy), direction, 6, accent)
        cv2.putText(frame, text, (x1 + r + tri_w + 2, cy + th // 2),
                    FONT, 0.5, txt_c, 1, cv2.LINE_AA)

    @staticmethod
    def _dash_segments(p1, p2, dash=14, gap=10):
        p1 = np.asarray(p1, dtype=np.float64)
        p2 = np.asarray(p2, dtype=np.float64)
        length = np.linalg.norm(p2 - p1)
        if length < 1e-6:
            return []
        u = (p2 - p1) / length
        segs, pos = [], 0.0
        while pos < length:
            a = p1 + u * pos
            b = p1 + u * min(pos + dash, length)
            segs.append((tuple(a.astype(int)), tuple(b.astype(int))))
            pos += dash + gap
        return segs

    @staticmethod
    def _chevron_pts(base, direction, size):
        d = np.asarray(direction, dtype=np.float64)
        d /= max(np.linalg.norm(d), 1e-9)
        p = np.array([-d[1], d[0]])
        b = np.asarray(base, dtype=np.float64)
        return np.array([
            b - d * size * 0.5 + p * size,
            b + d * size * 0.5,
            b - d * size * 0.5 - p * size,
        ], dtype=np.int32)

    # ---------- événements ----------

    def notify_count(self, frame_id, direction):
        if direction in self._last_count_frame:
            self._last_count_frame[direction] = frame_id

    # ---------- rendu ----------

    def draw_track(self, frame, tid, box, point, speed_kmh, frame_id):
        x1, y1, x2, y2 = box

        # Pendant le warm-up ou hors zone valide, on affiche simplement l'ID.
        if speed_kmh is None:
            color, label = C_OK, f"ID {tid}"
        elif speed_kmh > self.speed_limit:
            color, label = C_SPEEDER, f"ID {tid}  {speed_kmh:.0f} km/h"
        else:
            color, label = C_OK, f"ID {tid}  {speed_kmh:.0f} km/h"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, point, 4, color, -1, cv2.LINE_AA)
        self._label(frame, label, x1, max(28, y1 - 4), color)

        self.trails[tid].append(point)
        self._trail_seen[tid] = frame_id
        pts = list(self.trails[tid])
        for i in range(1, len(pts)):
            a = i / len(pts)
            cv2.line(frame, pts[i - 1], pts[i],
                     (0, int(255 * a), int(255 * (1 - a))), 2, cv2.LINE_AA)

    def _build_gate_cache(self, counter):
        """Prépare la géométrie fixe du portique."""
        p1 = counter.p1.astype(np.float64)
        p2 = counter.p2.astype(np.float64)
        line = p2 - p1
        length = np.linalg.norm(line)
        u = line / length
        n = np.array([-u[1], u[0]])
        off = float(counter.offset)
        dir_left, dir_right = n, -n

        quad = np.array([
            p1 + n * off, p2 + n * off,
            p2 - n * off, p1 - n * off,
        ], dtype=np.int32)

        ch = max(7.0, off * 0.8)
        ch_shift = off + ch + 4
        chevrons = []
        for frac in (0.23, 0.31, 0.39):
            base = p1 + u * (length * frac) + dir_left * ch_shift
            chevrons.append(self._chevron_pts(base, dir_left, ch))
        for frac in (0.54, 0.62, 0.70):
            base = p1 + u * (length * frac) + dir_right * ch_shift
            chevrons.append(self._chevron_pts(base, dir_right, ch))

        pill_offset = off + 26
        self._gate_cache = {
            "quad": quad,
            "quad_bbox": (quad[:, 0].min(), quad[:, 1].min(),
                          quad[:, 0].max() + 1, quad[:, 1].max() + 1),
            "dashes": self._dash_segments(p1, p2),
            "chevrons": chevrons,
            "dir_left": dir_left,
            "dir_right": dir_right,
            "pill_left_pos": p1 + u * 40 + dir_right * pill_offset,
            "pill_right_pos": p2 - u * 40 + dir_right * pill_offset,
        }

    def draw_count_gate(self, frame, counter, frame_id):
        if self._gate_cache is None:
            self._build_gate_cache(counter)
        g = self._gate_cache

        # Bande translucide autour de la ligne de comptage.
        bx1, by1, bx2, by2 = g["quad_bbox"]

        def draw_band(img, ox, oy):
            cv2.fillPoly(img, [g["quad"] + np.array([ox, oy])], C_GATE_FILL)

        _roi_blend(frame, bx1, by1, bx2, by2, draw_band, 0.30)
        cv2.polylines(frame, [g["quad"]], True, C_GATE_EDGE, 1, cv2.LINE_AA)

        # Ligne centrale pointillée.
        for a, b in g["dashes"]:
            cv2.line(frame, a, b, C_GATE_DASH, 1, cv2.LINE_AA)

        # Chevrons de direction.
        for pts in g["chevrons"]:
            cv2.polylines(frame, [pts], False, C_GATE_EDGE, 1, cv2.LINE_AA)

        # Compteurs directionnels.
        counts = counter.get_counts()
        hot_l = frame_id - self._last_count_frame["left"] <= self.pill_flash_frames
        hot_r = frame_id - self._last_count_frame["right"] <= self.pill_flash_frames

        self._pill(frame, g["pill_left_pos"], counts["left"], g["dir_left"], hot_l)
        self._pill(frame, g["pill_right_pos"], counts["right"], g["dir_right"], hot_r)

    def draw_dashboard(self, frame, fps, live, counts):
        x, y, w, h = 15, 15, 290, 64

        def draw_panel(img, ox, oy):
            cv2.rectangle(img, (x + ox, y + oy), (x + w + ox, y + h + oy), C_PANEL, -1)

        _roi_blend(frame, x, y, x + w + 1, y + h + 1, draw_panel, 0.7)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (90, 90, 90), 1)
        cv2.line(frame, (x + 10, y + 32), (x + w - 10, y + 32), (70, 70, 70), 1)

        cv2.putText(frame, "TRAFFIC MONITOR", (x + 10, y + 23),
                    FONT, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        if fps is not None:
            cv2.putText(frame, f"{fps:.0f} FPS", (x + w - 68, y + 23),
                        FONT, 0.45, (160, 220, 160), 1, cv2.LINE_AA)

        cv2.putText(frame, f"LIVE {live}", (x + 10, y + 53),
                    FONT, 0.45, C_TEXT, 1, cv2.LINE_AA)
        cv2.putText(frame, f"TOTAL {counts['total']}", (x + 95, y + 53),
                    FONT, 0.45, C_TEXT, 1, cv2.LINE_AA)
        cv2.putText(frame, f"LIMIT {self.speed_limit:.0f}", (x + 195, y + 53),
                    FONT, 0.45, (120, 120, 255), 1, cv2.LINE_AA)

    def prune(self, frame_id, stale_frames=90):
        stale = [tid for tid, f in self._trail_seen.items()
                 if frame_id - f > stale_frames]
        for tid in stale:
            self.trails.pop(tid, None)
            self._trail_seen.pop(tid, None)
