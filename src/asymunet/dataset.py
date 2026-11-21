"""Build the train/val/test sets: synthesis, ICP, rendering. Deterministic per seed."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from asymunet.face import make_face_sample
from asymunet.icp import estimate_symmetry_plane
from asymunet.render import canonical_frame, render
from asymunet.synth import make_sample

SIZE = 64


def build_one(seed: int, base: str = "field") -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    sample = make_face_sample(rng) if base == "face" else make_sample(rng)
    plane = estimate_symmetry_plane(sample.points)
    local = canonical_frame(sample.points, plane)
    valid, (depth, asym, mask) = render(
        local, [local[:, 2], plane.asymmetry, sample.labels.astype(np.float64)], size=SIZE
    )
    # Angle between the estimated normal and the true one (world x axis after the motion).
    true_normal = sample.rotation[:, 0]
    angle = np.degrees(np.arccos(min(1.0, abs(float(plane.normal @ true_normal)))))
    return {
        "inputs": np.stack([depth, asym, valid]).astype(np.float32),
        "mask": (mask > 0.5).astype(np.float32),
        "plane_error_deg": np.float32(angle),
        "icp_iterations": np.int32(plane.iterations),
    }


def build_split(
    first_seed: int, count: int, base: str = "field", workers: int = 8
) -> dict[str, np.ndarray]:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        items = list(
            pool.map(
                partial(build_one, base=base), range(first_seed, first_seed + count), chunksize=16
            )
        )
    return {key: np.stack([it[key] for it in items]) for key in items[0]}


def build_all(out_dir: Path, n_train: int, n_val: int, n_test: int, base: str = "field") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Disjoint seed ranges: no sample can appear in two splits.
    splits = {"train": (0, n_train), "val": (1_000_000, n_val), "test": (2_000_000, n_test)}
    for name, (first, count) in splits.items():
        data = build_split(first, count, base)
        np.savez_compressed(out_dir / f"{name}.npz", **data)
        print(
            f"{name}: {count} samples, median plane error "
            f"{np.median(data['plane_error_deg']):.2f} deg"
        )
