#!/usr/bin/env python

import math
import os
import sys
import unittest

import numpy as np


PYTHON_DIRECTORY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "python"))
if PYTHON_DIRECTORY not in sys.path:
    sys.path.insert(0, PYTHON_DIRECTORY)

from omni_hemisphere_boundary import (  # noqa: E402
    DEFAULT_FOV_ANGLES,
    draw_hemisphere_boundary,
    project_fov_boundary,
    project_hemisphere_boundary,
    select_yaml_camera,
)


def example_camera():
    return {
        "camera_model": "omni",
        "intrinsics": [1.0, 100.0, 120.0, 160.0, 120.0],
        "distortion_model": "radtan",
        "distortion_coeffs": [0.0, 0.0, 0.0, 0.0],
        "resolution": [320, 240],
    }


class OmniHemisphereBoundaryTest(unittest.TestCase):
    def test_no_distortion_boundary_has_expected_ellipse_radii(self):
        pixels, finite = project_hemisphere_boundary(example_camera(), 720)
        self.assertTrue(np.all(finite))
        normalized_u = (pixels[:, 0] - 160.0) / 100.0
        normalized_v = (pixels[:, 1] - 120.0) / 120.0
        np.testing.assert_allclose(normalized_u ** 2 + normalized_v ** 2,
                                   1.0, rtol=0.0, atol=1e-12)

    def test_full_fov_90_uses_a_45_degree_ray_cone(self):
        pixels, finite = project_fov_boundary(example_camera(), 90.0, 720)
        self.assertTrue(np.all(finite))
        expected_radius = math.sqrt(2.0) - 1.0
        normalized_u = (pixels[:, 0] - 160.0) / 100.0
        normalized_v = (pixels[:, 1] - 120.0) / 120.0
        np.testing.assert_allclose(normalized_u ** 2 + normalized_v ** 2,
                                   expected_radius ** 2,
                                   rtol=0.0, atol=1e-12)

    def test_overlay_draws_boundary_and_principal_point(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        overlay, stats = draw_hemisphere_boundary(
            image, example_camera(), sample_count=720)
        self.assertGreater(stats["rendered_segments"], 600)
        self.assertEqual([ring["fov_degrees"] for ring in stats["rings"]],
                         list(DEFAULT_FOV_ANGLES))
        self.assertGreater(np.count_nonzero(overlay), 0)
        self.assertEqual(overlay.shape[1], image.shape[1])
        self.assertGreater(overlay.shape[0], image.shape[0])
        self.assertEqual(stats["label_height"],
                         overlay.shape[0] - image.shape[0])
        self.assertTrue(np.all(image == 0))

        without_label, no_label_stats = draw_hemisphere_boundary(
            image, example_camera(), sample_count=720, draw_label=False)
        self.assertEqual(without_label.shape, image.shape)
        self.assertEqual(no_label_stats["label_height"], 0)

        _custom, custom_stats = draw_hemisphere_boundary(
            image, example_camera(), sample_count=720, draw_label=False,
            fov_angles=[90.0, 180.0])
        self.assertEqual(len(custom_stats["rings"]), 2)

    def test_camchain_requires_selection_when_multiple_omni_cameras(self):
        configuration = {"cam0": example_camera(), "cam1": example_camera()}
        with self.assertRaises(ValueError):
            select_yaml_camera(configuration)
        label, camera = select_yaml_camera(configuration, "cam1")
        self.assertEqual(label, "cam1")
        self.assertIs(camera, configuration["cam1"])


if __name__ == "__main__":
    unittest.main()
