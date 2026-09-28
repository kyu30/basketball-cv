import numpy as np


def pairwise_iou(boxes: np.ndarray) -> np.ndarray:
    """Pairwise IoU for boxes in (x, y, w, h) format.

    Returns an (n, n) matrix; diagonal is self-IoU (1.0).
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = boxes[:, 2] * boxes[:, 3]

    inter_x1 = np.maximum(x1[:, None], x1[None, :])
    inter_y1 = np.maximum(y1[:, None], y1[None, :])
    inter_x2 = np.minimum(x2[:, None], x2[None, :])
    inter_y2 = np.minimum(y2[:, None], y2[None, :])

    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter_area = inter_w * inter_h

    union = areas[:, None] + areas[None, :] - inter_area
    return np.where(union > 0, inter_area / union, 0.0)


def max_offdiag_iou(boxes: np.ndarray) -> float:
    """Max pairwise IoU among boxes, excluding self-overlap. 0 if <2 boxes."""
    n = len(boxes)
    if n < 2:
        return 0.0
    iou = pairwise_iou(boxes)
    np.fill_diagonal(iou, 0.0)
    return float(iou.max())
