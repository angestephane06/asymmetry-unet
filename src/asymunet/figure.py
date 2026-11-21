"""Example figure: inputs, ground truth, ICP baseline and U-Net on test samples."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from asymunet.model import UNet
from asymunet.train import load, normalise, predict


def make_figure(data_dir: Path, run_dir: Path, out: Path, n: int = 4) -> None:
    te_x, te_y = load(data_dir / "test.npz")
    tau = json.loads((run_dir / "results.json").read_text())["baseline_icp_threshold"]["threshold"]
    model = UNet()
    model.load_state_dict(torch.load(run_dir / "unet.pt", map_location="cpu"))
    deformed = np.flatnonzero(te_y.reshape(len(te_y), -1).any(axis=1))[:n]
    pred = predict(model, normalise(te_x[deformed]), "cpu")
    cols = ["Depth", "ICP asymmetry", "Ground truth", "ICP threshold", "U-Net"]
    fig, axes = plt.subplots(n, len(cols), figsize=(2.2 * len(cols), 2.6 * n))
    for row, i in enumerate(deformed):
        valid = te_x[i, 2] > 0
        maps = [
            te_x[i, 0],
            te_x[i, 1],
            te_y[i],
            (te_x[i, 1] > tau) & valid,
            (pred[row] > 0.5) & valid,
        ]
        for col, (title, img) in enumerate(zip(cols, maps, strict=True)):
            ax = axes[row, col]
            ax.imshow(
                np.where(valid, img, np.nan),
                cmap="viridis" if col < 2 else "gray_r",
                origin="lower",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
