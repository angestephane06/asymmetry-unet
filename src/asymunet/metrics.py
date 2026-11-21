"""Segmentation metrics, computed per image then averaged."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Bool = NDArray[np.bool_]


def dice_iou(pred: Bool, truth: Bool, valid: Bool) -> tuple[float, float]:
    """Dice and IoU inside the valid area. An empty prediction of an empty truth scores 1."""
    p, t = pred & valid, truth & valid
    inter = float((p & t).sum())
    total = float(p.sum() + t.sum())
    if total == 0:
        return 1.0, 1.0
    union = total - inter
    return 2 * inter / total, inter / union


def summarize(preds: Bool, truths: Bool, valids: Bool) -> dict[str, float]:
    scores = np.array([dice_iou(p, t, v) for p, t, v in zip(preds, truths, valids, strict=True)])
    deformed = truths.reshape(len(truths), -1).any(axis=1)
    # On symmetric samples, any positive pixel is a false alarm.
    false_alarm = (preds & valids).reshape(len(preds), -1).any(axis=1)
    return {
        "dice": float(scores[:, 0].mean()),
        "iou": float(scores[:, 1].mean()),
        "dice_deformed": float(scores[deformed, 0].mean()) if deformed.any() else float("nan"),
        "false_alarm_rate_symmetric": (
            float(false_alarm[~deformed].mean()) if (~deformed).any() else float("nan")
        ),
    }
