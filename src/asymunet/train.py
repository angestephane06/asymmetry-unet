"""Train the U-Net, and evaluate it against the thresholded ICP asymmetry map."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from asymunet.metrics import summarize
from asymunet.model import UNet


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["inputs"], data["mask"]


def normalise(inputs: np.ndarray) -> np.ndarray:
    """Depth centred per image (its absolute offset carries no information)."""
    out = inputs.copy()
    valid = out[:, 2:3] > 0
    depth = out[:, 0:1]
    mean = (depth * valid).sum(axis=(2, 3), keepdims=True) / np.maximum(
        valid.sum(axis=(2, 3), keepdims=True), 1
    )
    out[:, 0:1] = (depth - mean) * valid * 5.0
    out[:, 1:2] = out[:, 1:2] * 10.0
    return out


def baseline_threshold(val_x: np.ndarray, val_y: np.ndarray) -> float:
    """Best threshold of the ICP asymmetry map on the validation set (by mean Dice)."""
    best, best_dice = 0.0, -1.0
    valid = val_x[:, 2] > 0
    for tau in np.linspace(0.005, 0.2, 40):
        dice = summarize(val_x[:, 1] > tau, val_y > 0.5, valid)["dice"]
        if dice > best_dice:
            best, best_dice = float(tau), dice
    return best


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits) * valid
    inter = (prob * target).sum(dim=(1, 2, 3))
    total = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + 1) / (total + 1)).mean()


def predict(model: torch.nn.Module, x: np.ndarray, device: str, batch: int = 256) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i : i + batch]).to(device)
            outs.append(torch.sigmoid(model(xb)).cpu().numpy()[:, 0])
    return np.concatenate(outs)


def train(data_dir: Path, out_dir: Path, epochs: int = 40, seed: int = 0) -> dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = (
        load(data_dir / f"{s}.npz") for s in ("train", "val", "test")
    )
    tr_xn, va_xn, te_xn = normalise(tr_x), normalise(va_x), normalise(te_x)

    model = UNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=2e-3, total_steps=epochs * ((len(tr_x) + 31) // 32)
    )
    xt = torch.from_numpy(tr_xn)
    yt = torch.from_numpy(tr_y)[:, None]
    best_dice, best_state, history = -1.0, None, []
    start = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 32):
            idx = perm[i : i + 32]
            xb, yb = xt[idx].to(device), yt[idx].to(device)
            if torch.rand(1).item() < 0.5:  # vertical flip keeps the left/right geometry
                xb, yb = xb.flip(2), yb.flip(2)
            valid = xb[:, 2:3]
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb, weight=valid) + soft_dice_loss(
                logits, yb, valid
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        val_pred = predict(model, va_xn, device) > 0.5
        val = summarize(val_pred, va_y > 0.5, va_x[:, 2] > 0)
        history.append({"epoch": epoch + 1, "loss": loss.item(), **val})
        if val["dice"] > best_dice:
            best_dice = val["dice"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch + 1:2d} loss {loss.item():.3f} val dice {val['dice']:.3f}")
    train_time = time.time() - start

    assert best_state is not None
    model.load_state_dict(best_state)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / "unet.pt")

    tau = baseline_threshold(va_x, va_y)
    valid = te_x[:, 2] > 0
    results = {
        "test_samples": len(te_x),
        "baseline_icp_threshold": {
            "threshold": tau,
            **summarize(te_x[:, 1] > tau, te_y > 0.5, valid),
        },
        "unet": summarize(predict(model, te_xn, device) > 0.5, te_y > 0.5, valid),
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_seconds": round(train_time, 1),
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "history": history,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    return results


def evaluate_on(run_dir: Path, data_dir: Path) -> dict[str, object]:
    """Test-set scores of an already trained model on another dataset, with the ICP baseline
    threshold re-tuned on that dataset's validation split."""
    model = UNet()
    model.load_state_dict(torch.load(run_dir / "unet.pt", map_location="cpu"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    (va_x, va_y), (te_x, te_y) = (load(data_dir / f"{s}.npz") for s in ("val", "test"))
    tau = baseline_threshold(va_x, va_y)
    valid = te_x[:, 2] > 0
    return {
        "model": str(run_dir),
        "data": str(data_dir),
        "baseline_icp_threshold": {
            "threshold": tau,
            **summarize(te_x[:, 1] > tau, te_y > 0.5, valid),
        },
        "unet": summarize(predict(model, normalise(te_x), device) > 0.5, te_y > 0.5, valid),
    }
