import numpy as np


class PossessionTracker:
    def __init__(
        self,
        touch_distance_px=35,
        ball_memory_frames=6,
        touch_confirm_frames=3,
        num_teams=2
    ):

        self.touch_distance_px = float(touch_distance_px)
        self.ball_memory_frames = int(ball_memory_frames)
        self.touch_confirm_frames = int(touch_confirm_frames)
        self.num_teams = int(num_teams)

        self.ball_xy = None
        self.ball_frame_id = -10**9

        self.last_touch_player_id = None
        self.last_touch_team = -1

        self.candidate_player_id = None
        self.candidate_team = -1
        self.candidate_count = 0

        self.team_frames = {team: 0 for team in range(self.num_teams)}

    def update(self, tracks, ball_detections, team_tracker, frame_id):
        self._update_ball(ball_detections, frame_id)

        touching_player_id, touching_team = self._find_touch(
            tracks,
            team_tracker,
            frame_id,
        )

        if touching_player_id is not None:
            self._update_touch_candidate(touching_player_id, touching_team)
        else:
            self._reset_candidate()

        if self.last_touch_team != -1:
            self.team_frames[self.last_touch_team] += 1

        return self.last_touch_player_id, self.last_touch_team

    def get_possession_stats(self):
        total = sum(self.team_frames.values())
        if total == 0:
            return {team: 0.0 for team in self.team_frames}

        return {
            team: 100.0 * frames / total
            for team, frames in self.team_frames.items()
        }

    def _update_ball(self, ball_detections, frame_id):
        if not ball_detections:
            return

        best = max(ball_detections, key=lambda d: d.score)
        x1, y1, x2, y2 = best.xyxy

        self.ball_xy = (
            0.5 * (x1 + x2),
            0.5 * (y1 + y2),
        )
        self.ball_frame_id = frame_id

    def _find_touch(self, tracks, team_tracker, frame_id):
        if self.ball_xy is None:
            return None, -1

        if frame_id - self.ball_frame_id > self.ball_memory_frames:
            return None, -1

        candidates = []

        for track in tracks:
            team = team_tracker.get_team(track.track_id)
            if team == -1:
                continue

            distance = self._distance_to_feet(self.ball_xy, track.tlbr)

            if distance <= self.touch_distance_px:
                candidates.append((distance, track.track_id, team))

        if not candidates:
            return None, -1

        candidates.sort(key=lambda x: x[0])
        _, player_id, team = candidates[0]
        return player_id, team

    def _update_touch_candidate(self, player_id, team):
        if player_id == self.candidate_player_id:
            self.candidate_count += 1
        else:
            self.candidate_player_id = player_id
            self.candidate_team = team
            self.candidate_count = 1

        if self.candidate_count >= self.touch_confirm_frames:
            self.last_touch_player_id = self.candidate_player_id
            self.last_touch_team = self.candidate_team

    def _reset_candidate(self):
        self.candidate_player_id = None
        self.candidate_team = -1
        self.candidate_count = 0

    @staticmethod
    def _distance_to_feet(point, box):
        x, y = point
        x1, y1, x2, y2 = box

        foot_x = 0.5 * (x1 + x2)
        foot_y = y2

        return float(np.hypot(x - foot_x, y - foot_y))
