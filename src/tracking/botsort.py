import cv2
import numpy as np

from src.tracking.basetrack import BaseTrack, TrackState
from src.tracking.tracklet import Tracklet
from src.tracking.matching import iou_distance, linear_assignment, fuse_det_score
from src.tracking.utils import (
    detections_to_array,
    joint_tracklets,
    sub_tracklets,
    remove_duplicate_tracklets,
)


def _gmc_opencv(curr_img, prev_img):
    """Global Motion Compensation via ORB + estimateAffinePartial2D.
    Retourne la matrice de warp 2x3 (identite si echec)."""
    curr_gray = cv2.cvtColor(
        cv2.resize(curr_img, (curr_img.shape[1] // 2, curr_img.shape[0] // 2)),
        cv2.COLOR_BGR2GRAY,
    )
    prev_gray = cv2.cvtColor(
        cv2.resize(prev_img, (prev_img.shape[1] // 2, prev_img.shape[0] // 2)),
        cv2.COLOR_BGR2GRAY,
    )
    orb = cv2.ORB_create(500)
    kp1, desc1 = orb.detectAndCompute(prev_gray, None)
    kp2, desc2 = orb.detectAndCompute(curr_gray, None)
    if desc1 is None or desc2 is None or len(kp1) < 4 or len(kp2) < 4:
        return np.eye(2, 3, dtype=np.float32)
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(desc1, desc2)
    if len(matches) < 4:
        return np.eye(2, 3, dtype=np.float32)
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2) * 2
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2) * 2
    H, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)
    return H.astype(np.float32) if H is not None else np.eye(2, 3, dtype=np.float32)


def _multi_gmc(stracks, H):
    """Compensation du mouvement camera : transforme l'etat ET la covariance
    de chaque tracklet (state xywh + velocities, dim 8)."""
    if len(stracks) == 0:
        return
    R = H[:2, :2].astype(float)
    R8x8 = np.kron(np.eye(4, dtype=float), R)
    t = H[:2, 2].astype(float)

    for st in stracks:
        kf = st.kalman_filter.kf
        ndim_col = kf.x.ndim > 1

        x = kf.x.flatten()
        x = R8x8.dot(x)
        x[:2] += t
        kf.x = x.reshape(-1, 1) if ndim_col else x

        kf.P = R8x8.dot(kf.P).dot(R8x8.T)


class BotSortTracker:
    """BoT-SORT sans branche ReID : IoU + GMC uniquement."""
    def __init__(self, conf_thresh=0.5, track_buffer=30, motion='bot', frame_rate=30, with_gmc=True, match_thresh=0.8,
                 fuse_score=True, proximity_thresh=0.5):
        self.conf_thresh = conf_thresh
        self.det_thresh = conf_thresh + 0.1
        self.match_thresh = match_thresh
        self.fuse_score = fuse_score
        self.motion = motion
        self.with_gmc = with_gmc
        self.proximity_thresh = proximity_thresh
        self.frame_id = 0
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)
        self.tracked_tracklets = []
        self.lost_tracklets = []
        self.removed_tracklets = []
        self.prev_img = None
        BaseTrack.reset_id()

    def update(self, detections, curr_img=None):
        output_results = detections_to_array(detections)
        result = self._update_botsort(output_results, curr_img)
        if curr_img is not None:
            self.prev_img = curr_img
        return result

    def _update_botsort(self, output_results, curr_img):
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

        detections = [Tracklet(d, s, c, motion=self.motion)
                      for d, s, c in zip(dets, scores_keep, cates)] if len(dets) > 0 else []
        detections_second = [Tracklet(d, s, c, motion=self.motion)
                              for d, s, c in zip(dets_second, scores_second, cates_second)] if len(dets_second) > 0 else []

        unconfirmed = [t for t in self.tracked_tracklets if not t.is_activated]
        tracked_tracklets = [t for t in self.tracked_tracklets if t.is_activated]

        # Predict avant GMC
        tracklet_pool = joint_tracklets(tracked_tracklets, self.lost_tracklets)
        for t in tracklet_pool:
            t.predict()

        if self.with_gmc and curr_img is not None and self.prev_img is not None:
            warp = _gmc_opencv(curr_img, self.prev_img)
            _multi_gmc(tracklet_pool, warp)
            _multi_gmc(unconfirmed, warp)

        # Association 1 : high-score dets vs (tracked + lost)
        ious_dists = iou_distance(tracklet_pool, detections)
        # proximity_thresh : toute paire trop loin en IoU est exclue avant meme le fuse
        ious_dists_mask = ious_dists > self.proximity_thresh
        if self.fuse_score:
            ious_dists = fuse_det_score(ious_dists, detections)
        ious_dists[ious_dists_mask] = 1.0

        matches, u_track, u_detection = linear_assignment(ious_dists, thresh=self.match_thresh)
        for itracked, idet in matches:
            track, det = tracklet_pool[itracked], detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_tracklets.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_tracklets.append(track)

        # Association 2 (BYTE)
        r_tracked_tracklets = [tracklet_pool[i] for i in u_track
                                if tracklet_pool[i].state == TrackState.Tracked]
        dists = iou_distance(r_tracked_tracklets, detections_second)
        matches, u_track, u_detection_second = linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track, det = r_tracked_tracklets[itracked], detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_tracklets.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_tracklets.append(track)

        for it in u_track:
            track = r_tracked_tracklets[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_tracklets.append(track)

        # Unconfirmed
        detections = [detections[i] for i in u_detection]
        ious_dists = iou_distance(unconfirmed, detections)
        ious_dists_mask = ious_dists > self.proximity_thresh
        if self.fuse_score:
            ious_dists = fuse_det_score(ious_dists, detections)
        ious_dists[ious_dists_mask] = 1.0

        matches, u_unconfirmed, u_detection = linear_assignment(ious_dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_tracklets.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_tracklets.append(track)

        # Nouveaux tracks
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.frame_id)
            activated_tracklets.append(track)

        # Gestion des lost trop anciens
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