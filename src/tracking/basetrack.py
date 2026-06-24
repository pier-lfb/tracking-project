import numpy as np
from collections import OrderedDict


class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    _count = 0
    track_id = 0
    is_activated = False
    state = TrackState.New
    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    @staticmethod
    def reset_id():
        BaseTrack._count = 0

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    @property
    def tlwh(self):
        raise NotImplementedError

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @staticmethod
    def tlwh_to_xysa(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] = tlwh[2] * tlwh[3]
        ret[3] = tlwh[2] / tlwh[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"
