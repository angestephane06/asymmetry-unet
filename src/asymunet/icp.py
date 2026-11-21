"""Symmetry-plane estimation by ICP.

Python port of a MATLAB implementation: reflect the cloud across the current plane, match
every reflected point to its nearest neighbour, then update the plane in closed form (the
eigenvector of the smallest eigenvalue of A = sum(s s^T - d d^T)). The asymmetry of a point is
the distance between its reflection and that nearest neighbour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

Array = NDArray[np.float64]


@dataclass(frozen=True)
class SymmetryPlane:
    normal: Array  # unit vector, plane is normal . x = offset
    offset: float
    asymmetry: Array  # per-point distance after convergence
    iterations: int
    converged: bool


def reflect(points: Array, normal: Array, offset: float) -> Array:
    """Mirror points across the plane normal . x = offset."""
    distance = points @ normal - offset
    return points - 2.0 * distance[:, None] * normal[None, :]


def _icp_from(
    points: Array, tree: cKDTree, normal: Array, offset: float, max_iter: int, tol: float
) -> SymmetryPlane:
    previous = np.append(normal, offset)
    converged = False
    iterations = 0
    for iterations in range(1, max_iter + 1):  # noqa: B007 - kept after the loop
        _, idx = tree.query(reflect(points, normal, offset))
        matched = points[idx]
        xg, yg = points.mean(axis=0), matched.mean(axis=0)
        s = (points - xg) + (matched - yg)
        d = points - matched
        a = s.T @ s - d.T @ d  # vectorised form of sum_i (s_i s_i^T - d_i d_i^T)
        eigvals, eigvecs = np.linalg.eigh(a)
        normal = eigvecs[:, np.argmin(eigvals)]
        if normal[0] < 0:  # same sign convention as the MATLAB code
            normal = -normal
        offset = float(0.5 * (xg + yg) @ normal)
        current = np.append(normal, offset)
        if np.linalg.norm(current - previous) < tol:
            converged = True
            break
        previous = current
    reflected = reflect(points, normal, offset)
    dist, _ = tree.query(reflected)
    return SymmetryPlane(normal, offset, dist, iterations, converged)


def estimate_symmetry_plane(
    points: Array,
    max_iter: int = 60,
    tol: float = 1e-3,
    init: Literal["pca-min", "pca-all"] = "pca-all",
) -> SymmetryPlane:
    """Estimate the symmetry plane of a point cloud.

    init="pca-min" reproduces a first MATLAB draft: the initial normal is the principal axis of
    smallest variance. On a surface patch such as a face, that axis is usually the depth
    direction, not the left-right one, and ICP can settle on a wrong plane. "pca-all" starts
    from each of the three principal axes and keeps the plane of lowest mean asymmetry.
    """
    centre = points.mean(axis=0)
    eigvals, eigvecs = np.linalg.eigh(np.cov((points - centre).T))
    order = np.argsort(eigvals)
    starts = [order[0]] if init == "pca-min" else list(order)
    tree = cKDTree(points)
    results = []
    for k in starts:
        normal = eigvecs[:, k] / np.linalg.norm(eigvecs[:, k])
        results.append(_icp_from(points, tree, normal, float(centre @ normal), max_iter, tol))
    return min(results, key=lambda r: float(r.asymmetry.mean()))
