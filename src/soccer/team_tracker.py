from collections import defaultdict, deque
import numpy as np


class TeamTracker:
    def __init__(self, min_votes=5, stale_frames=300, num_teams=2, vote_window=30):
        self.min_votes = int(min_votes)
        self.stale_frames = int(stale_frames)
        self.num_teams = int(num_teams)
        self.vote_window = int(vote_window)

        self.votes = defaultdict(lambda: deque(maxlen=self.vote_window))
        self.last_seen = {}

    def add_vote(self, track_id, team, frame_id):
        self.last_seen[track_id] = frame_id

        if team < 0:
            return

        self.votes[track_id].append(team)

    def get_team(self, track_id):
        votes = self.votes.get(track_id)

        if not votes or len(votes) < self.min_votes:
            return -1

        counts = np.bincount(list(votes), minlength=self.num_teams)
        return int(np.argmax(counts))

    def update_players(self, player_tracks, frame, team_assigner, frame_id):
        """Ajoute un vote couleur pour chaque joueur."""
        if not team_assigner.is_fitted:
            return

        for track in player_tracks:
            team = team_assigner.predict(frame, track.tlbr)
            self.add_vote(track.track_id, team, frame_id)

    def update_goalkeepers(self, goalkeeper_tracks, player_tracks, frame_id):
        """Assigne les gardiens par proximité aux centroides des équipes."""
        if not goalkeeper_tracks:
            return

        centroids = self._team_centroids(player_tracks)
        if 0 not in centroids or 1 not in centroids:
            return

        for track in goalkeeper_tracks:
            point = self._box_center(track.tlbr)
            d0 = np.linalg.norm(point - centroids[0])
            d1 = np.linalg.norm(point - centroids[1])
            team = 0 if d0 < d1 else 1
            self.add_vote(track.track_id, team, frame_id)

    def prune(self, frame_id):
        stale_ids = [
            track_id
            for track_id, last_frame in self.last_seen.items()
            if frame_id - last_frame > self.stale_frames
        ]
        for track_id in stale_ids:
            self.votes.pop(track_id, None)
            self.last_seen.pop(track_id, None)

    def _team_centroids(self, player_tracks):
        points_by_team = {team: [] for team in range(self.num_teams)}

        for track in player_tracks:
            team = self.get_team(track.track_id)
            if team == -1:
                continue
            points_by_team[team].append(self._box_center(track.tlbr))

        return {
            team: np.mean(points, axis=0)
            for team, points in points_by_team.items()
            if points
        }

    @staticmethod
    def _box_center(box):
        x1, y1, x2, y2 = box
        return np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)], dtype=np.float32)
