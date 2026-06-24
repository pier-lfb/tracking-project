import numpy as np
from collections import deque

from src.tracking.basetrack import BaseTrack, TrackState
from src.tracking.kalman_filters.bytetrack_kalman import ByteKalman
from src.tracking.kalman_filters.ocsort_kalman import OCSORTKalman
from src.tracking.kalman_filters.sort_kalman import SORTKalman
from src.tracking.kalman_filters.botsort_kalman import BotKalman


MOTION_MODEL_DICT = {
    'sort': SORTKalman,
    'byte': ByteKalman,
    'bot': BotKalman,
    'ocsort': OCSORTKalman
}

STATE_CONVERT_DICT = {
    'sort': 'xysa',
    'byte': 'xyah',
    'bot': 'xywh',
    'ocsort': 'xysa'
}


class Tracklet(BaseTrack):
    def __init__(self, tlwh, score, category, motion='byte'):
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.is_activated = False
        self.score = float(score)
        self.category = int(category)
        self.motion = motion
        self.kalman_filter = MOTION_MODEL_DICT[motion]()
        self.convert_func = self.__getattribute__('tlwh_to_' + STATE_CONVERT_DICT[motion])
        self.kalman_filter.initialize(self.convert_func(self._tlwh))

    def predict(self):
        self.kalman_filter.predict()
        self.time_since_update += 1

    def activate(self, frame_id):
        self.track_id = self.next_id()
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.kalman_filter.update(self.convert_func(new_track.tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.score = float(new_track.score)
        self.category = int(new_track.category)
        self.time_since_update = 0
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.score = float(new_track.score)
        self.category = int(new_track.category)
        self.kalman_filter.update(self.convert_func(new_track.tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True
        self.time_since_update = 0

    @property
    def tlwh(self):
        return self.__getattribute__(STATE_CONVERT_DICT[self.motion] + '_to_tlwh')()

    def xyah_to_tlwh(self):
        x = self.kalman_filter.kf.x
        ret = x[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def xywh_to_tlwh(self):
        x = self.kalman_filter.kf.x
        ret = x[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    def xysa_to_tlwh(self):
        x = self.kalman_filter.kf.x
        ret = x[:4].copy()
        ret[2] = np.sqrt(x[2] * x[3])
        ret[3] = x[2] / ret[2]
        ret[:2] -= ret[2:] / 2
        return ret


class Tracklet_w_velocity(Tracklet):
    """Tracklet OC-SORT : memorise les observations, la vitesse, et expose
    la boite PREDITE par le Kalman pour l'association (OCM), tout en
    sortant la derniere boite OBSERVEE (conforme a l'original)."""

    def __init__(self, tlwh, score, category, motion='byte', delta_t=3):
        super().__init__(tlwh, score, category, motion)
        self.last_observation = np.array([-1, -1, -1, -1, -1])
        self.observations = dict()
        self.history_observations = []
        self.velocity = None
        self.delta_t = delta_t
        self.age = 0

    @property
    def tlwh(self):
        # Sortie : derniere observation si disponible (comportement original)
        if self.last_observation.sum() < 0:
            return self.__getattribute__(STATE_CONVERT_DICT[self.motion] + '_to_tlwh')()
        return self.tlbr_to_tlwh(self.last_observation[:4])

    @property
    def predicted_tlbr(self):
        """Boite issue de l'etat Kalman courant (apres predict).
        C'est sur CETTE boite que l'association doit se faire."""
        tlwh = self.__getattribute__(STATE_CONVERT_DICT[self.motion] + '_to_tlwh')()
        return self.tlwh_to_tlbr(tlwh)

    def apply_no_observation(self):
        """ORU : a appeler a chaque frame ou le track n'est pas matche.
        Declenche le freeze de l'etat Kalman a la premiere frame manquee."""
        self.kalman_filter.update(None)

    @staticmethod
    def speed_direction(bbox1, bbox2):
        cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
        cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
        speed = np.array([cy2 - cy1, cx2 - cx1])
        norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
        return speed / norm

    def predict(self):
        self.kalman_filter.predict()
        self.age += 1
        self.time_since_update += 1

    def re_activate(self, new_track, frame_id, new_id=False):
        super().re_activate(new_track, frame_id, new_id)
        self._record_observation(new_track)

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.score = float(new_track.score)
        self.category = int(new_track.category)
        self.kalman_filter.update(self.convert_func(new_track.tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True
        self.time_since_update = 0
        self._record_observation(new_track)

    def _record_observation(self, new_track):
        new_tlbr = self.tlwh_to_tlbr(new_track.tlwh)
        if self.last_observation.sum() >= 0:
            previous_box = None
            for dt in range(self.delta_t, 0, -1):
                if self.age - dt in self.observations:
                    previous_box = self.observations[self.age - dt]
                    break
            if previous_box is None:
                previous_box = self.last_observation
            self.velocity = self.speed_direction(previous_box, new_tlbr)

        new_observation = np.r_[new_tlbr, new_track.score]
        self.last_observation = new_observation
        self.observations[self.age] = new_observation
        self.history_observations.append(new_observation)


class Tracklet_w_bbox_buffer(Tracklet):
    """Tracklet C-BIoU conforme au papier : AUCUN filtre de Kalman.
    Le modele de mouvement est le deplacement moyen des n dernieres
    observations, applique pendant les frames non observees (coast)."""

    def __init__(self, tlwh, score, category, motion='byte', b1=0.3, b2=0.4, n=5):
        # Pas d'appel a super().__init__ : on ne veut pas instancier de Kalman
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.is_activated = False
        self.score = float(score)
        self.category = int(category)
        self.motion = motion
        self.b1, self.b2, self.n = float(b1), float(b2), int(n)
        self.origin_bbox_buffer = deque(maxlen=self.n)
        self.origin_bbox_buffer.append(self._tlwh.copy())
        self._refresh_states(self._tlwh)

    def get_buffer_bbox(self, level=1, bbox=None):
        assert level in [1, 2]
        b = self.b1 if level == 1 else self.b2
        src = self._tlwh if bbox is None else bbox
        buffer_bbox = src + np.array([-b * src[2], -b * src[3], 2 * b * src[2], 2 * b * src[3]])
        return np.maximum(0.0, buffer_bbox)

    def _refresh_states(self, bbox):
        self.buffer_bbox1 = self.get_buffer_bbox(level=1, bbox=bbox)
        self.buffer_bbox2 = self.get_buffer_bbox(level=2, bbox=bbox)
        self.motion_state1 = self.buffer_bbox1.copy()
        self.motion_state2 = self.buffer_bbox2.copy()

    @property
    def tlwh(self):
        # Sortie : derniere boite observee brute (pas de lissage Kalman)
        return self._tlwh.copy()

    def predict(self):
        """Pendant le coast, la boite bufferisee avance avec le deplacement
        moyen du buffer — c'est le modele de motion du papier. Le bug de
        l'ancienne version (extrapolation jamais executee car testee apres
        time_since_update = 0) est corrige en la placant ici."""
        self.time_since_update += 1
        if len(self.origin_bbox_buffer) >= 2:
            avg_step = (self.origin_bbox_buffer[-1] - self.origin_bbox_buffer[0]) \
                       / (len(self.origin_bbox_buffer) - 1)
            virtual_bbox = self.origin_bbox_buffer[-1] + self.time_since_update * avg_step
            self._refresh_states(virtual_bbox)

    def re_activate(self, new_track, frame_id, new_id=False):
        self._apply_observation(new_track, frame_id)
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id):
        self._apply_observation(new_track, frame_id)

    def _apply_observation(self, new_track, frame_id):
        self.frame_id = frame_id
        self.score = float(new_track.score)
        self.category = int(new_track.category)
        self._tlwh = np.asarray(new_track.tlwh, dtype=float)
        self.origin_bbox_buffer.append(self._tlwh.copy())
        self._refresh_states(self._tlwh)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.time_since_update = 0
