#!/usr/bin/env python

import os
import sys
import unittest

import numpy as np


PYTHON_DIRECTORY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "python"))
if PYTHON_DIRECTORY not in sys.path:
    sys.path.insert(0, PYTHON_DIRECTORY)

from omni_radtan_to_pinhole_equi import (  # noqa: E402
    convert_configuration,
    fit_camera,
    projection_change_samples,
    project_omni_radtan,
    sample_hemisphere,
)


def example_camera():
    return {
        "camera_model": "omni",
        "intrinsics": [0.82, 410.0, 405.0, 319.0, 241.0],
        "distortion_model": "radtan",
        "distortion_coeffs": [-0.015, 0.002, 0.0003, -0.0002],
        "resolution": [640, 480],
        "rostopic": "/camera/image_raw",
    }


class ConversionTest(unittest.TestCase):
    def test_hemisphere_sampling_is_unit_length_and_front_facing(self):
        rays = sample_hemisphere(1000)
        np.testing.assert_allclose(np.linalg.norm(rays, axis=1), 1.0,
                                   rtol=0.0, atol=1e-14)
        self.assertTrue(np.all(rays[:, 2] > 0.0))
        self.assertLess(np.min(rays[:, 2]), 0.001)

    def test_sampling_can_extend_slightly_beyond_the_front_hemisphere(self):
        rays = sample_hemisphere(1000, max_angle_degrees=92.0)
        np.testing.assert_allclose(np.linalg.norm(rays, axis=1), 1.0,
                                   rtol=0.0, atol=1e-14)
        self.assertLess(np.min(rays[:, 2]), 0.0)

    def test_source_projection_marks_only_sensor_pixels_valid(self):
        camera = example_camera()
        rays = sample_hemisphere(2000)
        pixels, valid = project_omni_radtan(camera, rays)
        width, height = camera["resolution"]
        self.assertGreater(np.count_nonzero(valid), 500)
        self.assertTrue(np.all(pixels[valid, 0] >= 0.0))
        self.assertTrue(np.all(pixels[valid, 0] < width))
        self.assertTrue(np.all(pixels[valid, 1] >= 0.0))
        self.assertTrue(np.all(pixels[valid, 1] < height))

    def test_fit_produces_kalibr_pinhole_equidistant_camera(self):
        camera = example_camera()
        converted, report = fit_camera(camera, sample_count=4000)
        self.assertEqual(converted["camera_model"], "pinhole")
        self.assertEqual(converted["distortion_model"], "equidistant")
        self.assertEqual(len(converted["intrinsics"]), 4)
        self.assertEqual(len(converted["distortion_coeffs"]), 4)
        self.assertEqual(converted["rostopic"], camera["rostopic"])
        self.assertGreater(report["valid_samples"], 1000)
        self.assertLess(report["rms_pixels"], 0.2)

    def test_camchain_conversion_preserves_unselected_camera(self):
        source = example_camera()
        other = {
            "camera_model": "pinhole",
            "intrinsics": [300.0, 300.0, 320.0, 240.0],
            "distortion_model": "equidistant",
            "distortion_coeffs": [0.0, 0.0, 0.0, 0.0],
            "resolution": [640, 480],
        }
        configuration = {"cam0": source, "cam1": other}
        converted, reports = convert_configuration(
            configuration, camera_name="cam0", sample_count=2000)
        self.assertEqual(converted["cam0"]["camera_model"], "pinhole")
        self.assertIs(converted["cam1"], other)
        self.assertEqual(set(reports), {"cam0"})

    def test_visualization_uses_only_requested_subset(self):
        source = example_camera()
        converted, _report = fit_camera(source, sample_count=2000)
        rays, source_pixels, converted_pixels = projection_change_samples(
            source, converted, candidate_count=2000, display_count=37)
        self.assertEqual(rays.shape, (37, 3))
        self.assertEqual(source_pixels.shape, (37, 2))
        self.assertEqual(converted_pixels.shape, (37, 2))
        self.assertTrue(np.all(np.isfinite(converted_pixels)))


if __name__ == "__main__":
    unittest.main()
