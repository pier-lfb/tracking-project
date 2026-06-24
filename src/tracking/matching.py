import numpy as np
import lap
from scipy.spatial.distance import cdist
from cython_bbox import bbox_overlaps as bbox_ious


chi2inv95 = {
    1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877,
    5: 11.070, 6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919
}


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0]
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    return np.asarray(matches), unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    _ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
    if _ious.size == 0:
        return _ious
    return bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=float),
        np.ascontiguousarray(btlbrs, dtype=float)
    )


def iou_distance(atracks, btracks):
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or \
       (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs, btlbrs = atracks, btracks
    else:
        atlbrs = [t.tlbr for t in atracks]
        btlbrs = [t.tlbr for t in btracks]
    return 1 - ious(atlbrs, btlbrs)


def buffered_iou_distance(atracks, btracks, level=1):
    assert level in [1, 2]
    if level == 1:
        atlbrs = [t.tlwh_to_tlbr(t.motion_state1) for t in atracks]
        btlbrs = [d.tlwh_to_tlbr(d.buffer_bbox1) for d in btracks]
    else:
        atlbrs = [t.tlwh_to_tlbr(t.motion_state2) for t in atracks]
        btlbrs = [d.tlwh_to_tlbr(d.buffer_bbox2) for d in btracks]
    return 1 - ious(atlbrs, btlbrs)


def fuse_det_score(cost_matrix, detections):
    """Fusionne le score de detection dans le cout IoU."""
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([d.score for d in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    return 1 - iou_sim * det_scores


def observation_centric_association(tracklets, detections, velocities, previous_obs,
                                    vdc_weight=0.05, iou_threshold=0.3):
    if len(tracklets) == 0:
        return np.empty((0, 2), dtype=int), tuple(range(len(tracklets))), tuple(range(len(detections)))

    # Association sur la boite prédite par Kalman conforme OC-SORT
    trk_tlbrs = np.array([
        t.predicted_tlbr if hasattr(t, 'predicted_tlbr') else t.tlbr
        for t in tracklets
    ])
    det_tlbrs = np.array([d.tlbr for d in detections])
    det_scores = np.array([d.score for d in detections])

    if trk_tlbrs.ndim == 1:
        trk_tlbrs = trk_tlbrs.reshape(-1, 4)
    if det_tlbrs.ndim == 1:
        det_tlbrs = det_tlbrs.reshape(-1, 4)

    iou_matrix = bbox_ious(
        np.ascontiguousarray(trk_tlbrs, dtype=float),
        np.ascontiguousarray(det_tlbrs, dtype=float)
    )
    iou_matrix[iou_matrix < iou_threshold] = -1e5

    previous_obs = np.array(previous_obs)
    Y, X = _speed_direction_batch(det_tlbrs, previous_obs)
    inertia_Y = np.repeat(velocities[:, 0:1], Y.shape[1], axis=1)
    inertia_X = np.repeat(velocities[:, 1:2], X.shape[1], axis=1)
    diff_angle_cos = np.clip(inertia_X * X + inertia_Y * Y, -1, 1)
    diff_angle = (np.pi / 2.0 - np.abs(np.arccos(diff_angle_cos))) / np.pi

    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0

    scores = np.repeat(det_scores[:, np.newaxis], trk_tlbrs.shape[0], axis=1)
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)
    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight * scores.T

    return linear_assignment(-(iou_matrix + angle_diff_cost), thresh=0.0)


def _speed_direction_batch(dets, tracks):
    tracks = tracks[..., np.newaxis]
    CX1 = (dets[:, 0] + dets[:, 2]) / 2.0
    CY1 = (dets[:, 1] + dets[:, 3]) / 2.0
    CX2 = (tracks[:, 0] + tracks[:, 2]) / 2.0
    CY2 = (tracks[:, 1] + tracks[:, 3]) / 2.0
    dx = CX2 - CX1
    dy = CY2 - CY1
    norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
    return dy / norm, dx / norm
