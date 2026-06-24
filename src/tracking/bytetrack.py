import numpy as np
from src.tracking.basetrack import TrackState
from src.tracking.tracklet import Tracklet
from src.tracking.matching import iou_distance, linear_assignment, fuse_det_score
from src.tracking.utils import (
    detections_to_array,
    joint_tracklets,
    sub_tracklets,
    remove_duplicate_tracklets,
)


class ByteTracker:
    def __init__(self, conf_thresh=0.5, track_buffer=30, motion='byte', frame_rate=30,
                 match_thresh=0.8, fuse_score=True):
        self.tracked_tracklets = []
        self.lost_tracklets = []
        self.removed_tracklets = []
        self.frame_id = 0
        self.conf_thresh = conf_thresh
        self.det_thresh = conf_thresh + 0.1
        self.motion = motion
        self.match_thresh = match_thresh   # 0.8 dans l'original (apres fusion du score)
        self.fuse_score = fuse_score
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)

    def update(self, detections):
        output_results = detections_to_array(detections)
        return self._update(output_results)

    def _update(self, output_results):
        self.frame_id += 1
        activated_tracklets, refind_tracklets, lost_tracklets, removed_tracklets = [], [], [], []

        if len(output_results) > 0:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
            categories = output_results[:, 5]
        else:
            scores = np.array([])
            bboxes = np.empty((0, 4))
            categories = np.array([])

        remain_inds = scores > self.conf_thresh
        inds_second = np.logical_and(scores > 0.1, scores <= self.conf_thresh)

        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        cates = categories[remain_inds]
        dets_second = bboxes[inds_second]
        scores_second = scores[inds_second]
        cates_second = categories[inds_second]

        detections = [Tracklet(d, s, c, motion=self.motion) for d, s, c in zip(dets, scores_keep, cates)] if len(dets) > 0 else []
        detections_second = [Tracklet(d, s, c, motion=self.motion) for d, s, c in zip(dets_second, scores_second, cates_second)] if len(dets_second) > 0 else []

        unconfirmed = [t for t in self.tracked_tracklets if not t.is_activated]
        tracked_tracklets = [t for t in self.tracked_tracklets if t.is_activated]

        tracklet_pool = joint_tracklets(tracked_tracklets, self.lost_tracklets)
        for t in tracklet_pool:
            t.predict()

        # --- Association 1 : detections high-score, IoU fusionne au score ---
        dists = iou_distance(tracklet_pool, detections)
        if self.fuse_score:
            dists = fuse_det_score(dists, detections)
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh)
        for itracked, idet in matches:
            track, det = tracklet_pool[itracked], detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_tracklets.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind_tracklets.append(track)

        # --- Association 2 : detections low-score, tracks Tracked restants ---
        r_tracked = [tracklet_pool[i] for i in u_track if tracklet_pool[i].state == TrackState.Tracked]
        dists = iou_distance(r_tracked, detections_second)
        matches, u_track2, _ = linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track, det = r_tracked[itracked], detections_second[idet]
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

        # --- Association 3 : unconfirmed ---
        detections_remaining = [detections[i] for i in u_detection]
        dists = iou_distance(unconfirmed, detections_remaining)
        if self.fuse_score:
            dists = fuse_det_score(dists, detections_remaining)
        matches, u_unconfirmed, u_det = linear_assignment(dists, thresh=0.7)
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
