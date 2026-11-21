"""Synthetic symmetric surfaces with known local deformations.

Each sample is a smooth height field that is exactly symmetric about x = 0 (pairs of Gaussian
features mirrored across the midline, plus features centred on it), sampled as a noisy point
cloud. Most samples then receive one or two local bumps or dents on one side only; the points
that moved are the ground truth. The cloud is finally given a random rigid motion, so the
symmetry plane has to be estimated, as with a real scan.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
X_HALF, Y_HALF = 1.0, 1.3


@dataclass(frozen=True)
class Sample:
    points: Array  # (N, 3) after the rigid motion
    labels: NDArray[np.bool_]  # (N,) True where the surface was deformed
    rotation: Array  # rigid motion applied: world = R @ canonical + t
    translation: Array


def _gaussians(x: Array, y: Array, params: Array) -> Array:
    """Sum of Gaussians; params rows are (cx, cy, amplitude, sigma)."""
    z = np.zeros_like(x)
    for cx, cy, amp, sigma in params:
        z += amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
    return z


def _rotation(rng: np.random.Generator, max_deg: float) -> Array:
    ax, ay, az = np.deg2rad(rng.uniform(-max_deg, max_deg, 3))
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return rz @ ry @ rx


def make_sample(
    rng: np.random.Generator,
    n_points: int = 6000,
    p_deformed: float = 0.85,
    noise: float = 0.004,
    max_rotation_deg: float = 12.0,
) -> Sample:
    x = rng.uniform(-X_HALF, X_HALF, n_points)
    y = rng.uniform(-Y_HALF, Y_HALF, n_points)

    # Symmetric base: a dome, mirrored feature pairs, features on the midline.
    z = -rng.uniform(0.15, 0.35) * (x**2 + 0.6 * y**2)
    pairs = []
    for _ in range(rng.integers(3, 7)):
        cx, cy = rng.uniform(0.15, 0.8), rng.uniform(-1.0, 1.0)
        amp, sigma = rng.uniform(-0.25, 0.25), rng.uniform(0.1, 0.35)
        pairs += [(cx, cy, amp, sigma), (-cx, cy, amp, sigma)]
    for _ in range(rng.integers(1, 3)):
        pairs.append((0.0, rng.uniform(-0.8, 0.8), rng.uniform(-0.3, 0.35), rng.uniform(0.08, 0.3)))
    z += _gaussians(x, y, np.array(pairs))

    return deform_and_place(x, y, z, rng, p_deformed, noise, max_rotation_deg)


def deform_and_place(
    x: Array,
    y: Array,
    z: Array,
    rng: np.random.Generator,
    p_deformed: float = 0.85,
    noise: float = 0.004,
    max_rotation_deg: float = 12.0,
) -> Sample:
    """Add one-sided local deformations (with their labels), noise and a rigid motion to a
    surface that is symmetric about x = 0. Shared by every kind of base surface."""
    labels = np.zeros(len(x), dtype=bool)
    z = z.copy()
    if rng.random() < p_deformed:
        side = rng.choice([-1.0, 1.0])  # both deformations of a sample on the same side
        for _ in range(rng.integers(1, 3)):
            cx = side * rng.uniform(0.2, 0.8)
            cy = rng.uniform(-1.0, 1.0)
            amp = rng.choice([-1.0, 1.0]) * rng.uniform(0.04, 0.2)
            sigma = rng.uniform(0.08, 0.2)
            delta = _gaussians(x, y, np.array([(cx, cy, amp, sigma)]))
            z += delta
            # A point is "deformed" if it moved by more than a quarter of the bump height.
            labels |= np.abs(delta) > 0.25 * abs(amp)

    z += rng.normal(0.0, noise, len(x))
    canonical = np.stack([x, y, z], axis=1)
    rotation = _rotation(rng, max_rotation_deg)
    translation = rng.uniform(-0.3, 0.3, 3)
    return Sample(canonical @ rotation.T + translation, labels, rotation, translation)
