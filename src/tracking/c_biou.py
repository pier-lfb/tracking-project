import numpy as np
from src.tracking.basetrack import TrackState
from src.tracking.tracklet import Tracklet_w_bbox_buffer
from src.tracking.matching import buffered_iou_distance, linear_assignment
from src.tracking.utils import (
    detections_to_array,
    joint_tracklets,
    sub_tracklets,
    remove_duplicate_tracklets,
)


class CBIoUTracker:
    """
    C-BIoU (Yang et al.) inspiré du papier :
    - pas de filtre de Kalman, motion = déplacement moyen du buffer
    - cascade à deux tours sur la MEME population de detections :
      tour 1 avec buffer b1, tour 2 (tracks et detections restants) avec b2
    - b1=0.3, b2=0.5 : valeurs d'exemple du papier.
    """

    def __init__(self, conf_thresh=0.5, track_buffer=30, motion='byte', frame_rate=30,
                 match_thresh1=0.7, match_thresh2=0.6, b1=0.3, b2=0.5):
        self.tracked_tracklets = []
        self.lost_tracklets = []
        self.removed_tracklets = []
        self.frame_id = 0
        self.conf_thresh = conf_thresh
        self.det_thresh = conf_thresh + 0.1
        self.motion = motion
        self.match_thresh1 = match_thresh1
        self.match_thresh2 = match_thresh2
        self.b1, self.b2 = b1, b2
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)

    def update(self, detections):
        output_results = detections_to_array(detections)
        self.frame_id += 1
        activated_tracklets, refind_tracklets, lost_tracklets, removed_tracklets = [], [], [], []

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

        detections = [
            Tracklet_w_bbox_buffer(d, s, c, motion=self.motion, b1=self.b1, b2=self.b2)
            for d, s, c in zip(dets, scores_keep, cates)
        ] if len(dets) > 0 else []

        unconfirmed = [t for t in self.tracked_tracklets if not t.is_activated]
        tracked_tracklets = [t for t in self.tracked_tracklets if t.is_activated]
        tracklet_pool = joint_tracklets(tracked_tracklets, self.lost_tracklets)

        for t in tracklet_pool:
            t.predict()

        # --- Tour 1 : BIoU avec buffer étroit (b1) ---
        dists = buffered_iou_distance(tracklet_pool, detections, level=1)
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh1)
        for itracked, idet in matches:
            track, det = tracklet_pool[itracked], detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_tracklets.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind_tracklets.append(track)

        # --- Tour 2 : tracks restants vs mêmes detections restantes (b2) ---
        r_tracked = [tracklet_pool[i] for i in u_track if tracklet_pool[i].state == TrackState.Tracked]
        r_detections = [detections[i] for i in u_detection]

        dists = buffered_iou_distance(r_tracked, r_detections, level=2)
        matches, u_track2, u_det2 = linear_assignment(dists, thresh=self.match_thresh2)
        for itracked, idet in matches:
            track, det = r_tracked[itracked], r_detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_tracklets.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind_tracklets.append(track)
        for it in u_track2:
            track = r_tracked[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_tracklets.append(track)

        # --- Tour 3 : unconfirmed vs detections restantes (b1) ---
        detections_remaining = [r_detections[i] for i in u_det2]
        dists = buffered_iou_distance(unconfirmed, detections_remaining, level=1)
        matches, u_unconfirmed, u_det = linear_assignment(dists, thresh=self.match_thresh1)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections_remaining[idet], self.frame_id)
            activated_tracklets.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            unconfirmed[it].mark_removed()
            removed_tracklets.append(unconfirmed[it])

        for inew in u_det:
            track = detections_remaining[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.frame_id)
            activated_tracklets.append(track)

        for track in self.lost_tracklets:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_tracklets.append(track)

        self.tracked_tracklets = [t for t in self.tracked_tracklets if t.state == TrackState.Tracked]
        self.tracked_tracklets = joint_tracklets(self.tracked_tracklets, activated_tracklets)
        self.tracked_tracklets = joint_tracklets(self.tracked_tracklets, refind_tracklets)
        self.lost_tracklets = sub_tracklets(self.lost_tracklets, self.tracked_tracklets)
        self.lost_tracklets.extend(lost_tracklets)
        self.lost_tracklets = sub_tracklets(self.lost_tracklets, self.removed_tracklets)
        self.removed_tracklets.extend(removed_tracklets)
        self.tracked_tracklets, self.lost_tracklets = remove_duplicate_tracklets(
            self.tracked_tracklets, self.lost_tracklets)

        return [t for t in self.tracked_tracklets if t.is_activated]
