"""Face surfaces built from MediaPipe's canonical face mesh (Apache 2.0, see assets/NOTICE).

The mesh is a generic face, not a scan of a real person, and is exactly symmetric about x = 0.
Each sample varies its proportions and adds mirrored local morphology, so no two faces are
identical, then goes through the same one-sided deformations as the other surfaces.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from asymunet.synth import Sample, _gaussians, deform_and_place

Array = NDArray[np.float64]
MESH = Path(__file__).parent / "assets" / "canonical_face_model.obj"


@lru_cache(maxsize=1)
def load_face() -> tuple[Array, NDArray[np.int64]]:
    """Vertices normalised to a face width of 2 (x in [-1, 1]), centred, and triangles."""
    vertices, faces = [], []
    for line in MESH.read_text().splitlines():
        parts = line.split()
        if parts and parts[0] == "v":
            vertices.append([float(p) for p in parts[1:4]])
        elif parts and parts[0] == "f":
            faces.append([int(p.split("/")[0]) - 1 for p in parts[1:4]])
    v = np.array(vertices)
    v -= [0.0, v[:, 1].mean(), v[:, 2].min()]
    v *= 2.0 / np.ptp(v[:, 0])
    return v, np.array(faces, dtype=np.int64)


def sample_surface(
    rng: np.random.Generator, vertices: Array, faces: NDArray[np.int64], n_points: int
) -> Array:
    """Uniform points on the triangle mesh (area-weighted triangle, then barycentric)."""
    a, b, c = (vertices[faces[:, k]] for k in range(3))
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    tri = rng.choice(len(faces), size=n_points, p=areas / areas.sum())
    r1, r2 = rng.random(n_points), rng.random(n_points)
    flip = r1 + r2 > 1
    r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
    return a[tri] + r1[:, None] * (b[tri] - a[tri]) + r2[:, None] * (c[tri] - a[tri])


def make_face_sample(rng: np.random.Generator, n_points: int = 6000, **kwargs: float) -> Sample:
    vertices, faces = load_face()
    pts = sample_surface(rng, vertices, faces, n_points)
    # Symmetric identity variation: proportions, then mirrored bumps (the face stays symmetric).
    pts = pts * rng.uniform([0.92, 0.92, 0.85], [1.08, 1.08, 1.15])
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    pairs = []
    for _ in range(rng.integers(2, 5)):
        cx, cy = rng.uniform(0.15, 0.8), rng.uniform(-0.9, 0.9)
        amp, sigma = rng.uniform(-0.06, 0.06), rng.uniform(0.1, 0.3)
        pairs += [(cx, cy, amp, sigma), (-cx, cy, amp, sigma)]
    z = z + _gaussians(x, y, np.array(pairs))
    return deform_and_place(x, y, z, rng, **kwargs)
