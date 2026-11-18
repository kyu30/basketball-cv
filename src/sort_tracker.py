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


class AppTrack:
    __slots__ = ("id", "box", "conf", "feat", "hits", "time_since_update", "vel")

    def __init__(self, track_id, box, conf, feat):
        self.id = track_id
        self.box = box
        self.conf = conf
        self.feat = feat  # EMA appearance embedding (L2-normalized)
        self.hits = 1
        self.time_since_update = 0
        self.vel = np.zeros(2)  # constant-velocity estimate of the box center

    def predicted_box(self):
        """Box shifted forward by the current velocity (per frame since last update)."""
        shift = self.vel * (self.time_since_update + 1)
        return self.box + np.array([shift[0], shift[1], shift[0], shift[1]])


class AppearanceTracker:
    """IoU tracker augmented with a learned appearance embedding.

    Association cost = alpha*(1 - IoU) + (1 - alpha)*appearance_distance, so
    appearance can re-link a track whose box moved too far for IoU alone. alpha=1
    reduces exactly to the pure-IoU baseline (the clean appearance on/off ablation).
    A center-distance gate blocks cross-court identity teleports.
    """
    def __init__(self, alpha: float = 0.1, iou_thresh: float = 0.3, app_thresh: float = 0.25,
                 max_cost: float = 0.7, gate_px: float = 1e9, max_age: int = 30,
                 min_hits: int = 3, det_thresh: float = 0.25, ema: float = 0.9,
                 motion: bool = False, vel_ema: float = 0.5):
        self.alpha = alpha
        self.iou_thresh = iou_thresh
        self.app_thresh = app_thresh
        self.max_cost = max_cost
        self.gate_px = gate_px
        self.max_age = max_age
        self.min_hits = min_hits
        self.det_thresh = det_thresh
        self.ema = ema
        self.motion = motion        # constant-velocity prediction before IoU (BoT-SORT-style)
        self.vel_ema = vel_ema
        self.tracks: list[AppTrack] = []
        self._next_id = 1

    @staticmethod
    def _centers(boxes: np.ndarray) -> np.ndarray:
        return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)

    def update(self, dets_xyxy: np.ndarray, confs: np.ndarray, embs: np.ndarray):
        keep = confs >= self.det_thresh
        dets_xyxy, confs, embs = dets_xyxy[keep], confs[keep], embs[keep]

        if self.tracks and len(dets_xyxy):
            tboxes = np.array([t.predicted_box() if self.motion else t.box for t in self.tracks])
            tfeat = np.array([t.feat for t in self.tracks])
            ious = iou_matrix(tboxes, dets_xyxy)
            app_dist = (1.0 - tfeat @ embs.T) / 2.0           # cosine distance in [0,1]
            cost = self.alpha * (1.0 - ious) + (1.0 - self.alpha) * app_dist
            # Center-distance gate: forbid matches between far-apart boxes.
            cdist = np.linalg.norm(self._centers(tboxes)[:, None] - self._centers(dets_xyxy)[None], axis=2)
            cost[cdist > self.gate_px] = 1e6

            track_idx, det_idx = linear_sum_assignment(cost)
            matches, matched_t, matched_d = [], set(), set()
            for t, d in zip(track_idx, det_idx):
                # Accept if geometry agrees, or appearance strongly agrees, within budget.
                ok = cost[t, d] <= self.max_cost and (ious[t, d] >= self.iou_thresh or app_dist[t, d] <= self.app_thresh)
                if ok and cdist[t, d] <= self.gate_px:
                    matches.append((t, d)); matched_t.add(t); matched_d.add(d)
            unmatched_t = [i for i in range(len(self.tracks)) if i not in matched_t]
            unmatched_d = [i for i in range(len(dets_xyxy)) if i not in matched_d]
        else:
            matches, unmatched_t, unmatched_d = [], list(range(len(self.tracks))), list(range(len(dets_xyxy)))

        for t, d in matches:
            trk = self.tracks[t]
            new_box = dets_xyxy[d]
            if self.motion:
                old_c = np.array([(trk.box[0] + trk.box[2]) / 2, (trk.box[1] + trk.box[3]) / 2])
                new_c = np.array([(new_box[0] + new_box[2]) / 2, (new_box[1] + new_box[3]) / 2])
                step = (new_c - old_c) / (trk.time_since_update + 1)
                trk.vel = self.vel_ema * trk.vel + (1 - self.vel_ema) * step
            trk.box = new_box
            trk.conf = float(confs[d])
            trk.feat = self.ema * trk.feat + (1 - self.ema) * embs[d]
            trk.feat /= (np.linalg.norm(trk.feat) + 1e-12)
            trk.hits += 1
            trk.time_since_update = 0
        for t in unmatched_t:
            self.tracks[t].time_since_update += 1
        for d in unmatched_d:
            self.tracks.append(AppTrack(self._next_id, dets_xyxy[d], float(confs[d]), embs[d]))
            self._next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return [
            (t.id, t.box, t.conf)
            for t in self.tracks
            if t.time_since_update == 0 and t.hits >= self.min_hits
        ]
