import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX

C_TEAM0 = (0, 0, 255)
C_TEAM1 = (255, 0, 0)
C_UNKNOWN = (128, 128, 128)
C_BALL = (0, 255, 255)
C_TEXT = (235, 235, 235)
C_PANEL = (18, 18, 18)

TEAM_COLORS = {
    0: C_TEAM0,
    1: C_TEAM1,
    -1: C_UNKNOWN,
}


class FootballVisualizer:
    def __init__(self, class_names):
        self.class_names = class_names

    def class_name(self, track):
        cid = int(track.category)
        return self.class_names[cid] if 0 <= cid < len(self.class_names) else "unknown"

    def draw(self, frame, tracks, ball_detections, team_tracker,
             holder_id=None, holder_team=-1, stats=None, fps=None):
        self.draw_tracks(frame, tracks, team_tracker, holder_id)
        self.draw_ball(frame, ball_detections)
        self._panel(frame, stats or {0: 0.0, 1: 0.0}, holder_id, holder_team, fps)
        return frame

    def draw_tracks(self, frame, tracks, team_tracker, holder_id=None):
        for track in tracks:
            name = self.class_name(track)

            if name in ("Player", "Goalkeeper"):
                team = team_tracker.get_team(track.track_id)
                label = f"GK {track.track_id}" if name == "Goalkeeper" else f"ID {track.track_id}"
                self._player(frame, track.tlbr, label, team, track.track_id == holder_id)

            elif name == "Referee":
                self._referee(frame, track.tlbr)

    def _player(self, frame, box, label, team, is_holder=False):
        x1, y1, x2, y2 = [int(v) for v in box]
        color = TEAM_COLORS.get(team, C_UNKNOWN)

        cx = int((x1 + x2) / 2)
        feet_y = int(y2)
        width = max(18, int((x2 - x1) * 0.45))
        height = max(6, int((y2 - y1) * 0.08))
        thickness = 3 if is_holder else 2

        cv2.ellipse(frame, (cx, feet_y), (width, height), 0, 0, 360,
                    color, thickness, cv2.LINE_AA)

        if is_holder:
            cv2.circle(frame, (cx, max(0, y1 - 10)), 6, C_BALL, -1, cv2.LINE_AA)

        self._label(frame, label, cx, feet_y + 18, color)

    def _referee(self, frame, box):
        x1, y1, x2, y2 = [int(v) for v in box]
        cx = int((x1 + x2) / 2)
        feet_y = int(y2)

        cv2.ellipse(
            frame,
            (cx, feet_y),
            (max(18, int((x2 - x1) * 0.45)),
             max(6, int((y2 - y1) * 0.08))),
            0, 0, 360,
            C_UNKNOWN,
            1,
            cv2.LINE_AA,
        )
        self._label(frame, "REF", cx, feet_y + 18, C_UNKNOWN)

    def draw_ball(self, frame, ball_detections):
        for ball in ball_detections:
            x1, y1, x2, y2 = ball.xyxy.astype(int)
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            radius = max(3, int(max(x2 - x1, y2 - y1) / 2))
            cv2.circle(frame, center, radius, C_BALL, 2, cv2.LINE_AA)

    def _panel(self, frame, stats, holder_id, holder_team, fps):
        x, y, w, h = 15, 15, 335, 115

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), C_PANEL, -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (90, 90, 90), 1)

        cv2.putText(frame, "FOOTBALL MONITOR", (x + 12, y + 26),
                    FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        if fps is not None:
            cv2.putText(frame, f"{fps:.0f} FPS", (x + w - 80, y + 26),
                        FONT, 0.5, (160, 220, 160), 1, cv2.LINE_AA)

        t0 = float(stats.get(0, 0.0))
        t1 = float(stats.get(1, 0.0))
        total = max(t0 + t1, 1.0)

        bar_x, bar_y = x + 12, y + 45
        bar_w, bar_h = w - 24, 14
        split = int(bar_w * (t0 / total))

        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (45, 45, 45), -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + split, bar_y + bar_h), C_TEAM0, -1)
        cv2.rectangle(frame, (bar_x + split, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), C_TEAM1, -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (120, 120, 120), 1)

        cursor_x = bar_x + split
        cv2.line(frame, (cursor_x, bar_y - 4),
                 (cursor_x, bar_y + bar_h + 4), (255, 255, 255), 2)

        cv2.putText(frame, f"TEAM 0  {t0:.1f}%", (x + 12, y + 80),
                    FONT, 0.45, C_TEXT, 1, cv2.LINE_AA)

        cv2.putText(frame, f"TEAM 1  {t1:.1f}%", (x + 178, y + 80),
                    FONT, 0.45, C_TEXT, 1, cv2.LINE_AA)

        txt = f"POSSESSION  PLAYER ID {holder_id}" if holder_id is not None else "POSSESSION  --"
        cv2.putText(frame, txt, (x + 12, y + 101), FONT, 0.42, C_TEXT, 1, cv2.LINE_AA)


    @staticmethod
    def _label(frame, text, center_x, y, color):
        scale = 0.45
        thickness = 1
        (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)

        x = int(center_x - tw / 2)
        y = int(y)

        cv2.rectangle(frame, (x - 4, y - th - 5),
                      (x + tw + 4, y + 4), (25, 25, 25), -1)
        cv2.rectangle(frame, (x - 4, y - th - 5),
                      (x + tw + 4, y + 4), color, 1)
        cv2.putText(frame, text, (x, y),
                    FONT, scale, color, thickness, cv2.LINE_AA)
