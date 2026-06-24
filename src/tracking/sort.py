import numpy as np
from src.tracking.basetrack import TrackState
from src.tracking.tracklet import Tracklet
from src.tracking.matching import iou_distance, linear_assignment
from src.tracking.utils import detections_to_array


class SortTracker:
    def __init__(self, conf_thresh=0.5, track_buffer=30, motion='sort', frame_rate=30,
                 min_hits=3):
        self.tracked_tracklets = []
        self.frame_id = 0
        self.conf_thresh = conf_thresh
        self.det_thresh = conf_thresh + 0.1
        self.motion = motion
        self.min_hits = int(min_hits)
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)
        # NB : le SORT original utilise max_age=1, mettre track_buffer=1

    def update(self, detections):
        output_results = detections_to_array(detections)
        self.frame_id += 1
        activated_tracklets = []

        if len(output_results) > 0:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
            categories = output_results[:, 5]
            remain_inds = scores > self.conf_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            cates = categories[remain_inds]
        else:
            dets = np.empty((0, 4))
            scores_keep = np.array([])
            cates = np.array([])

        det_tracklets = [Tracklet(d, s, c, motion=self.motion) for d, s, c in zip(dets, scores_keep, cates)]

        for t in self.tracked_tracklets:
            t.predict()

        u_detection = list(range(len(det_tracklets)))
        if det_tracklets:
            dists = iou_distance(self.tracked_tracklets, det_tracklets)
            matches, u_track, u_detection = linear_assignment(dists, thresh=0.7)
            for itracked, idet in matches:
                track = self.tracked_tracklets[itracked]
                track.update(det_tracklets[idet], self.frame_id)
                track.hits = getattr(track, 'hits', 0) + 1
                activated_tracklets.append(track)

        self.tracked_tracklets = [t for t in self.tracked_tracklets if self.frame_id - t.end_frame <= self.max_time_lost]

        for inew in u_detection:
            track = det_tracklets[inew]
            track.activate(self.frame_id)
            track.hits = 1
            activated_tracklets.append(track)

        self.tracked_tracklets = [t for t in self.tracked_tracklets if t.is_activated] + activated_tracklets

        # min_hits : un track n'est émis qu'après min_hits matchs consécutifs, sauf en debut de sequence (comme dans le SORT original)
        return [
            t for t in activated_tracklets
            if getattr(t, 'hits', 0) >= self.min_hits or self.frame_id <= self.min_hits
        ]
