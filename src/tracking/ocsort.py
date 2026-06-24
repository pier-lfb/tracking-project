import numpy as np
from src.tracking.basetrack import TrackState
from src.tracking.tracklet import Tracklet_w_velocity
from src.tracking.matching import iou_distance, observation_centric_association, linear_assignment, ious
from src.tracking.utils import (
    detections_to_array,
    joint_tracklets,
    sub_tracklets,
    remove_duplicate_tracklets,
)


class OcSortTracker:
    def __init__(self, conf_thresh=0.5, track_buffer=30, motion='ocsort', frame_rate=30, delta_t=3):
        self.tracked_tracklets = []
        self.lost_tracklets = []
        self.removed_tracklets = []
        self.frame_id = 0
        self.conf_thresh = conf_thresh
        self.det_thresh = conf_thresh + 0.1
        self.motion = motion
        self.delta_t = delta_t
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)

    @staticmethod
    def k_previous_obs(observations, cur_age, k):
        if len(observations) == 0:
            return [-1, -1, -1, -1, -1]
        for i in range(k):
            dt = k - i
            if cur_age - dt in observations:
                return observations[cur_age - dt]
        return observations[max(observations.keys())]

    def update(self, detections):
        output_results = detections_to_array(detections)
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

        detections = [Tracklet_w_velocity(d, s, c, motion=self.motion) for d, s, c in zip(dets, scores_keep, cates)] if len(dets) > 0 else []
        detections_second = [Tracklet_w_velocity(d, s, c, motion=self.motion) for d, s, c in zip(dets_second, scores_second, cates_second)] if len(dets_second) > 0 else []

        unconfirmed = [t for t in self.tracked_tracklets if not t.is_activated]
        tracked_tracklets = [t for t in self.tracked_tracklets if t.is_activated]
        tracklet_pool = joint_tracklets(tracked_tracklets, self.lost_tracklets)

        velocities = np.array([t.velocity if t.velocity is not None else np.array((0, 0)) for t in tracklet_pool])
        k_observations = np.array([self.k_previous_obs(t.observations, t.age, self.delta_t) for t in tracklet_pool])

        for t in tracklet_pool:
            t.predict()

        # --- OCM : association sur boites predites + cout directionnel ---
        matches, u_track, u_detection = observation_centric_association(
            tracklets=tracklet_pool, detections=detections,
            iou_threshold=0.3, velocities=velocities,
            previous_obs=k_observations, vdc_weight=0.05
        )
        for itracked, idet in matches:
            track, det = tracklet_pool[itracked], detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_tracklets.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind_tracklets.append(track)

        # --- Association (low-score), sur boites prédites ---
        r_tracked = [tracklet_pool[i] for i in u_track if tracklet_pool[i].state == TrackState.Tracked]
        dists = iou_distance(
            [t.predicted_tlbr for t in r_tracked],
            [d.tlbr for d in detections_second]
        )
        matches, u_track2, _ = linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track, det = r_tracked[itracked], detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_tracklets.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind_tracklets.append(track)

        # --- OCR : dernière chance sur la dernière observation ---
        r_tracked2 = [r_tracked[i] for i in u_track2]
        r_detections = [detections[i] for i in u_detection]

        if len(r_tracked2) > 0 and len(r_detections) > 0:
            dists = 1.0 - ious(
                atlbrs=[t.last_observation[:4] for t in r_tracked2],
                btlbrs=[d.tlbr for d in r_detections]
            )
            matches, u_track3, u_det3 = linear_assignment(dists, thresh=0.5)
            for itracked, idet in matches:
                track, det = r_tracked2[itracked], r_detections[idet]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_tracklets.append(track)
                else:
                    track.re_activate(det, self.frame_id)
                    refind_tracklets.append(track)
            for it in u_track3:
                track = r_tracked2[it]
                if track.state != TrackState.Lost:
                    track.mark_lost()
                    lost_tracklets.append(track)
            detections_remaining = [r_detections[i] for i in u_det3]
        else:
            for t in r_tracked2:
                if t.state != TrackState.Lost:
                    t.mark_lost()
                    lost_tracklets.append(t)
            detections_remaining = r_detections

        # --- Unconfirmed ---
        dists = iou_distance(unconfirmed, detections_remaining)
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

        # --- ORU : signaler l'absence d'observation aux tracks non matches ---
        for t in tracklet_pool:
            if t.time_since_update > 0:
                t.apply_no_observation()

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
