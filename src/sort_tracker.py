"""A pure-IoU multi-object tracker (SORT minus the Kalman filter).

Association only: each frame, existing tracks are matched to detections by
maximizing IoU (Hungarian assignment), with no motion model. A track keeps its
last box as its prediction. This is the transparent baseline the stronger
ByteTrack (Kalman + two-stage association) is compared against; the gap between
them is exactly what motion modeling and low-confidence recovery buy.

Track lifecycle:
    - new detection -> tentative track (not yet output)
    - confirmed once it reaches `min_hits` consecutive updates
    - dropped after `max_age` consecutive frames with no matching detection
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from iou import iou_matrix


class Track:
    __slots__ = ("id", "box", "conf", "hits", "time_since_update")

    def __init__(self, track_id: int, box: np.ndarray, conf: float):
        self.id = track_id
        self.box = box  # (x1, y1, x2, y2)
        self.conf = conf
        self.hits = 1
        self.time_since_update = 0


class IouTracker:
    def __init__(self, iou_thresh: float = 0.3, max_age: int = 30, min_hits: int = 3,
                 det_thresh: float = 0.25):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.det_thresh = det_thresh
        self.tracks: list[Track] = []
        self._next_id = 1

    def _associate(self, dets_xyxy: np.ndarray):
        """Return (matches, unmatched_tracks, unmatched_dets) index lists."""
        if not self.tracks or len(dets_xyxy) == 0:
            return [], list(range(len(self.tracks))), list(range(len(dets_xyxy)))

        track_boxes = np.array([t.box for t in self.tracks])
        ious = iou_matrix(track_boxes, dets_xyxy)
        # Hungarian assignment maximizing IoU (minimize -IoU).
        track_idx, det_idx = linear_sum_assignment(-ious)

        matches, matched_t, matched_d = [], set(), set()
        for t, d in zip(track_idx, det_idx):
            if ious[t, d] >= self.iou_thresh:
                matches.append((t, d))
                matched_t.add(t)
                matched_d.add(d)
        unmatched_t = [i for i in range(len(self.tracks)) if i not in matched_t]
        unmatched_d = [i for i in range(len(dets_xyxy)) if i not in matched_d]
        return matches, unmatched_t, unmatched_d

    def update(self, dets_xyxy: np.ndarray, confs: np.ndarray) -> list[tuple[int, np.ndarray, float]]:
        """Advance one frame. dets_xyxy is (n, 4) in (x1, y1, x2, y2) pixels.
        Returns confirmed tracks visible this frame as (id, box, conf)."""
        keep = confs >= self.det_thresh
        dets_xyxy, confs = dets_xyxy[keep], confs[keep]

        matches, unmatched_t, unmatched_d = self._associate(dets_xyxy)

        for t, d in matches:
            trk = self.tracks[t]
            trk.box = dets_xyxy[d]
            trk.conf = float(confs[d])
            trk.hits += 1
            trk.time_since_update = 0
        for t in unmatched_t:
            self.tracks[t].time_since_update += 1
        for d in unmatched_d:
            self.tracks.append(Track(self._next_id, dets_xyxy[d], float(confs[d])))
            self._next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return [
            (t.id, t.box, t.conf)
            for t in self.tracks
            if t.time_since_update == 0 and t.hits >= self.min_hits
        ]
