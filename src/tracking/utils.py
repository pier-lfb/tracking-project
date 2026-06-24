import numpy as np
from src.tracking.matching import iou_distance


def detections_to_array(detections: list) -> np.ndarray:
    """Convertit list[Detection] -> np.ndarray (N, 6)
    [x, y, w, h, score, class_id] en format tlwh
    """
    if len(detections) == 0:
        return np.empty((0, 6))

    arr = []
    for d in detections:
        x1, y1, x2, y2 = d.xyxy
        w = x2 - x1
        h = y2 - y1
        arr.append([x1, y1, w, h, d.score, d.class_id])

    return np.array(arr, dtype=float)


# ---------- operations communes sur les listes de tracklets ----------

def joint_tracklets(a, b):
    """Union de deux listes de tracklets, sans doublon d'ID (priorite a `a`)."""
    exists = {t.track_id: 1 for t in a}
    res = list(a)
    for t in b:
        if not exists.get(t.track_id, 0):
            exists[t.track_id] = 1
            res.append(t)
    return res


def sub_tracklets(a, b):
    """Tracklets de `a` dont l'ID n'apparait pas dans `b`."""
    tracks = {t.track_id: t for t in a}
    for t in b:
        tracks.pop(t.track_id, None)
    return list(tracks.values())


def remove_duplicate_tracklets(a, b, iou_thresh=0.15):
    """Supprime les paires (a, b) quasi superposees (distance IoU < iou_thresh)
    en gardant, pour chaque paire, le tracklet le plus ancien."""
    pdist = iou_distance(a, b)
    pairs = np.where(pdist < iou_thresh)
    dupa, dupb = [], []
    for p, q in zip(*pairs):
        tp = a[p].frame_id - a[p].start_frame
        tq = b[q].frame_id - b[q].start_frame
        if tp > tq:
            dupb.append(q)
        else:
            dupa.append(p)
    return [t for i, t in enumerate(a) if i not in dupa], \
           [t for i, t in enumerate(b) if i not in dupb]
