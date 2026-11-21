"""From a point cloud and its estimated symmetry plane to fixed-size 2D maps.

The cloud is expressed in a frame where the estimated plane is x = 0, then interpolated on a
regular grid: depth, ICP asymmetry, validity, and (for training) the deformation mask. The U-Net
and the baseline therefore see exactly the same ICP output.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import griddata

from asymunet.icp import SymmetryPlane
from asymunet.synth import X_HALF, Y_HALF

Array = NDArray[np.float64]


def canonical_frame(points: Array, plane: SymmetryPlane) -> Array:
    """Rotate/translate so that the plane becomes x = 0 and depth is the z axis.

    The in-plane axes come from a PCA of the points projected on the plane (depth is the axis
    of smallest spread). Signs are fixed with the world axes, i.e. the scan is assumed to face
    the sensor roughly, as a face scan does.
    """
    n = plane.normal / np.linalg.norm(plane.normal)
    if n[0] < 0:
        n = -n
    centred = points - points.mean(axis=0)
    in_plane = centred - np.outer(centred @ n, n)
    eigvals, eigvecs = np.linalg.eigh(np.cov(in_plane.T))
    order = np.argsort(eigvals)
    depth, vertical = eigvecs[:, order[1]], eigvecs[:, order[2]]  # order[0] is ~n itself
    depth -= (depth @ n) * n
    depth /= np.linalg.norm(depth)
    vertical = np.cross(depth, n)
    if depth[2] < 0:
        depth = -depth
    if vertical[1] < 0:
        vertical = -vertical
    basis = np.stack([n, vertical, depth])  # rows: new x, y, z
    on_plane = points - (points @ n - plane.offset)[:, None] * n  # projection origin on plane
    origin = on_plane.mean(axis=0)
    return (points - origin) @ basis.T


def render(
    local: Array, values: list[Array], size: int = 64, margin: float = 0.1
) -> tuple[Array, list[Array]]:
    """Interpolate per-point values on a size x size grid over the sampling window.

    Returns the validity map (inside the convex hull of the points) and one map per value
    array, with 0 outside.
    """
    xs = np.linspace(-X_HALF - margin, X_HALF + margin, size)
    ys = np.linspace(-Y_HALF - margin, Y_HALF + margin, size)
    gx, gy = np.meshgrid(xs, ys)
    xy = local[:, :2]
    maps = []
    valid = None
    for v in values:
        grid = griddata(xy, v, (gx, gy), method="linear")
        if valid is None:
            valid = ~np.isnan(grid)
        maps.append(np.nan_to_num(grid, nan=0.0))
    assert valid is not None
    return valid.astype(np.float64), maps
