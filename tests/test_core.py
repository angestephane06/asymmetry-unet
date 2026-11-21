from __future__ import annotations

import numpy as np
import torch

from asymunet.dataset import build_one
from asymunet.icp import estimate_symmetry_plane, reflect
from asymunet.metrics import dice_iou, summarize
from asymunet.model import UNet
from asymunet.synth import make_sample


def test_reflection_is_an_involution() -> None:
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(100, 3))
    n = np.array([0.6, 0.8, 0.0])
    assert np.allclose(reflect(reflect(pts, n, 0.3), n, 0.3), pts)


def test_icp_recovers_the_plane_of_a_symmetric_cloud() -> None:
    rng = np.random.default_rng(1)
    half = rng.uniform([0.1, -1, -0.2], [1, 1, 0.2], size=(2000, 3))
    cloud = np.vstack([half, half * [-1, 1, 1]])  # exactly symmetric about x = 0
    plane = estimate_symmetry_plane(cloud)
    assert abs(abs(plane.normal[0]) - 1) < 1e-6 and plane.asymmetry.max() < 1e-9


def test_multi_start_beats_smallest_axis_init_on_a_flat_surface() -> None:
    # A wide, thin surface: the smallest PCA axis is the depth, not the symmetry normal.
    sample = make_sample(np.random.default_rng(3), p_deformed=0.0, max_rotation_deg=0.0)
    single = estimate_symmetry_plane(sample.points, init="pca-min")
    multi = estimate_symmetry_plane(sample.points, init="pca-all")
    assert multi.asymmetry.mean() <= single.asymmetry.mean()
    assert abs(multi.normal[0]) > 0.99


def test_deformations_are_one_sided_and_labelled() -> None:
    sample = make_sample(np.random.default_rng(5), p_deformed=1.0, max_rotation_deg=0.0)
    xs = sample.points[sample.labels, 0] - sample.translation[0]
    assert sample.labels.any() and (np.all(xs > 0) or np.all(xs < 0))


def test_dataset_sample_shapes_and_determinism() -> None:
    a, b = build_one(7), build_one(7)
    assert a["inputs"].shape == (3, 64, 64) and a["mask"].shape == (64, 64)
    assert np.array_equal(a["inputs"], b["inputs"])


def test_metrics() -> None:
    t = np.zeros((4, 4), bool)
    t[:2, :2] = True
    v = np.ones((4, 4), bool)
    assert dice_iou(t, t, v) == (1.0, 1.0)
    assert dice_iou(np.zeros_like(t), np.zeros_like(t), v) == (1.0, 1.0)
    p = np.zeros_like(t)
    p[:1, :2] = True
    assert dice_iou(p, t, v) == (2 * 2 / 6, 2 / 4)
    s = summarize(np.stack([p, t]), np.stack([t, np.zeros_like(t)]), np.stack([v, v]))
    assert s["false_alarm_rate_symmetric"] == 1.0


def test_unet_output_shape() -> None:
    out = UNet()(torch.zeros(2, 3, 64, 64))
    assert out.shape == (2, 1, 64, 64)


def test_face_mesh_is_symmetric_and_normalised() -> None:
    from scipy.spatial import cKDTree

    from asymunet.face import load_face

    vertices, faces = load_face()
    assert len(vertices) == 468 and faces.max() < len(vertices)
    assert abs(np.ptp(vertices[:, 0]) - 2.0) < 1e-9
    mirrored = vertices * [-1, 1, 1]
    assert cKDTree(vertices).query(mirrored)[0].max() < 1e-6


def test_face_samples_are_deformed_on_one_side() -> None:
    from asymunet.face import make_face_sample

    sample = make_face_sample(np.random.default_rng(2), p_deformed=1.0, max_rotation_deg=0.0)
    xs = sample.points[sample.labels, 0] - sample.translation[0]
    assert sample.labels.any() and (np.all(xs > 0) or np.all(xs < 0))
    assert build_one(4, "face")["inputs"].shape == (3, 64, 64)
