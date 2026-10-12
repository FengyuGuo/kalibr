"""Project and draw full-FOV guide rings for a Kalibr omni camera."""

from __future__ import print_function

import math

import cv2
import numpy as np


DEFAULT_FOV_ANGLES = (90.0, 120.0, 150.0, 180.0)


def normalized_camera(camera, image_resolution=None):
    """Validate an omni camera mapping and normalize no-distortion cameras."""
    if not isinstance(camera, dict):
        raise ValueError("camera must be a mapping")
    if camera.get("camera_model") != "omni":
        raise ValueError("camera_model must be 'omni'")
    intrinsics = camera.get("intrinsics")
    if not isinstance(intrinsics, (list, tuple)) or len(intrinsics) != 5:
        raise ValueError("omni intrinsics must be [xi, fu, fv, cu, cv]")
    intrinsics = [float(value) for value in intrinsics]
    if not np.all(np.isfinite(intrinsics)):
        raise ValueError("intrinsics must be finite")
    if intrinsics[0] <= 0.0:
        raise ValueError("xi must be positive to project the z=0 boundary")
    if intrinsics[1] <= 0.0 or intrinsics[2] <= 0.0:
        raise ValueError("focal lengths must be positive")

    distortion_model = camera.get("distortion_model", "none")
    distortion = camera.get("distortion_coeffs", [])
    if distortion_model == "none":
        if distortion not in ([], (), None):
            raise ValueError("distortion_model 'none' requires no coefficients")
        distortion_model = "radtan"
        distortion = [0.0, 0.0, 0.0, 0.0]
    elif distortion_model == "radtan":
        if not isinstance(distortion, (list, tuple)) or len(distortion) != 4:
            raise ValueError("radtan coefficients must be [k1, k2, p1, p2]")
        distortion = [float(value) for value in distortion]
        if not np.all(np.isfinite(distortion)):
            raise ValueError("distortion coefficients must be finite")
    else:
        raise ValueError("omni boundary tool supports radtan or no distortion")

    resolution = camera.get("resolution", image_resolution)
    if resolution is None:
        raise ValueError("camera resolution is required")
    if len(resolution) != 2 or int(resolution[0]) <= 0 or int(resolution[1]) <= 0:
        raise ValueError("resolution must be [width, height]")

    result = dict(camera)
    result["intrinsics"] = intrinsics
    result["distortion_model"] = distortion_model
    result["distortion_coeffs"] = distortion
    result["resolution"] = [int(resolution[0]), int(resolution[1])]
    return result


def _validated_fov_angles(fov_angles):
    if fov_angles is None:
        fov_angles = DEFAULT_FOV_ANGLES
    values = [float(value) for value in fov_angles]
    if not values:
        raise ValueError("at least one FOV angle is required")
    if not np.all(np.isfinite(values)):
        raise ValueError("FOV angles must be finite")
    if any(value <= 0.0 or value > 180.0 for value in values):
        raise ValueError("full FOV angles must be in (0, 180]")
    return sorted(set(values))


def project_fov_boundary(camera, fov_degrees, sample_count=1440):
    """Project a full-FOV cone boundary and return pixels plus a finite mask.

    ``fov_degrees`` is the complete cone angle. Its ray angle from the optical
    axis is therefore theta = fov_degrees / 2.
    """
    if sample_count < 32:
        raise ValueError("at least 32 boundary samples are required")
    fov_degrees = float(fov_degrees)
    if not np.isfinite(fov_degrees) or not 0.0 < fov_degrees <= 180.0:
        raise ValueError("full FOV angle must be in (0, 180]")
    camera = normalized_camera(camera)
    xi, fu, fv, cu, cv = camera["intrinsics"]
    k1, k2, p1, p2 = camera["distortion_coeffs"]

    theta = math.radians(0.5 * fov_degrees)
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    denominator = cos_theta + xi
    if denominator <= 0.0:
        raise ValueError("FOV boundary is outside the omni projection domain")
    radius = sin_theta / denominator
    azimuth = np.linspace(0.0, 2.0 * math.pi, int(sample_count), endpoint=False)
    x = radius * np.cos(azimuth)
    y = radius * np.sin(azimuth)
    x2 = x * x
    y2 = y * y
    xy = x * y
    r2 = x2 + y2
    radial = k1 * r2 + k2 * r2 * r2
    xd = x + x * radial + 2.0 * p1 * xy + p2 * (r2 + 2.0 * x2)
    yd = y + y * radial + 2.0 * p2 * xy + p1 * (r2 + 2.0 * y2)
    pixels = np.column_stack((fu * xd + cu, fv * yd + cv))
    return pixels, np.all(np.isfinite(pixels), axis=1)


def project_hemisphere_boundary(camera, sample_count=1440):
    """Project the z=0 ring, equivalent to a full FOV of 180 degrees."""
    return project_fov_boundary(camera, 180.0, sample_count)


def _safe_int_point(point, coordinate_limit=1000000.0):
    point = np.clip(point, -coordinate_limit, coordinate_limit)
    return int(round(float(point[0]))), int(round(float(point[1])))


def draw_hemisphere_boundary(image, camera, sample_count=1440,
                             color=(0, 255, 0), thickness=2,
                             draw_label=True, draw_principal_point=True,
                             fov_angles=None):
    """Draw multiple full-FOV guide rings and return image/statistics."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a non-empty numpy array")
    if image.ndim == 2:
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 3:
        output = image.copy()
    elif image.ndim == 3 and image.shape[2] == 4:
        output = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        raise ValueError("image must be grayscale, BGR, or BGRA")
    if thickness < 1:
        raise ValueError("thickness must be positive")
    if len(color) != 3 or any(value < 0 or value > 255 for value in color):
        raise ValueError("color must contain three values in [0, 255]")

    height, width = output.shape[:2]
    camera = normalized_camera(camera, image_resolution=[width, height])
    angles = _validated_fov_angles(fov_angles)
    base_color = tuple(int(value) for value in color)
    rectangle = (0, 0, width, height)
    rings = []
    ring_colors = []

    for ring_index, fov_degrees in enumerate(angles):
        if len(angles) == 1:
            brightness = 1.0
        else:
            brightness = 0.4 + 0.6 * ring_index / float(len(angles) - 1)
        ring_color = tuple(int(round(value * brightness)) for value in base_color)
        ring_colors.append(ring_color)
        pixels, finite = project_fov_boundary(
            camera, fov_degrees, sample_count)
        rendered_segments = 0
        for index in range(pixels.shape[0]):
            following = (index + 1) % pixels.shape[0]
            if not finite[index] or not finite[following]:
                continue
            first = _safe_int_point(pixels[index])
            second = _safe_int_point(pixels[following])
            visible, clipped_first, clipped_second = cv2.clipLine(
                rectangle, first, second)
            if visible:
                cv2.line(output, clipped_first, clipped_second, ring_color,
                         int(thickness), cv2.LINE_AA)
                rendered_segments += 1

        theta = math.radians(0.5 * fov_degrees)
        normalized_radius = (math.sin(theta) /
                             (math.cos(theta) + camera["intrinsics"][0]))
        rings.append({
            "fov_degrees": float(fov_degrees),
            "theta_degrees": float(0.5 * fov_degrees),
            "finite_samples": int(np.count_nonzero(finite)),
            "rendered_segments": int(rendered_segments),
            "ideal_radius_u": float(camera["intrinsics"][1] * normalized_radius),
            "ideal_radius_v": float(camera["intrinsics"][2] * normalized_radius),
        })

    xi, _fu, _fv, cu, cv = camera["intrinsics"]
    if draw_principal_point and 0.0 <= cu < width and 0.0 <= cv < height:
        center = int(round(cu)), int(round(cv))
        cv2.drawMarker(output, center, (0, 0, 255), cv2.MARKER_CROSS,
                       max(10, 4 * int(thickness)), int(thickness), cv2.LINE_AA)

    label_height = 0
    if draw_label:
        title = "omni FOV guide rings (full FOV; ray angle theta=FOV/2)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        title_scale = max(0.42, min(width, height) / 1300.0)
        title_size, title_baseline = cv2.getTextSize(
            title, font, title_scale, 1)
        available_width = max(1, width - 20)
        if title_size[0] > available_width:
            title_scale *= float(available_width) / float(title_size[0])
            title_size, title_baseline = cv2.getTextSize(
                title, font, title_scale, 1)

        legend_scale = max(0.34, 0.85 * title_scale)
        legend_items = []
        legend_y = 10 + title_size[1] + title_baseline + 8
        legend_x = 10
        legend_row_height = 0
        for fov_degrees, ring_color in zip(angles, ring_colors):
            text = "FOV {:g} deg".format(fov_degrees)
            text_size, baseline = cv2.getTextSize(text, font, legend_scale, 1)
            item_width = 18 + text_size[0] + 14
            row_height = text_size[1] + baseline + 6
            legend_row_height = max(legend_row_height, row_height)
            if legend_x > 10 and legend_x + item_width > width - 10:
                legend_x = 10
                legend_y += legend_row_height
                legend_row_height = row_height
            legend_items.append((legend_x, legend_y, text, text_size,
                                 baseline, ring_color))
            legend_x += item_width
        label_height = legend_y + legend_row_height + 6

        canvas = np.full((height + label_height, width, 3), 24,
                         dtype=output.dtype)
        canvas[label_height:label_height + height, :width] = output
        cv2.putText(canvas, title, (10, 8 + title_size[1]), font,
                    title_scale, base_color, 1, cv2.LINE_AA)
        for x, y, text, text_size, _baseline, ring_color in legend_items:
            center_y = y + text_size[1] // 2
            cv2.line(canvas, (x, center_y), (x + 10, center_y),
                     ring_color, max(1, int(thickness)), cv2.LINE_AA)
            cv2.putText(canvas, text, (x + 14, y + text_size[1]), font,
                        legend_scale, ring_color, 1, cv2.LINE_AA)
        output = canvas

    outer_ring = rings[-1]
    stats = {
        "sample_count": int(sample_count),
        "finite_samples": int(sum(ring["finite_samples"] for ring in rings)),
        "rendered_segments": int(sum(
            ring["rendered_segments"] for ring in rings)),
        "image_width": int(width),
        "image_height": int(height),
        "label_height": int(label_height),
        "output_height": int(output.shape[0]),
        "xi": float(xi),
        "ideal_boundary_radius_u": outer_ring["ideal_radius_u"],
        "ideal_boundary_radius_v": outer_ring["ideal_radius_v"],
        "rings": rings,
    }
    return output, stats


def select_yaml_camera(configuration, camera_name=None):
    """Select one omni camera from a single-camera or camchain mapping."""
    if not isinstance(configuration, dict):
        raise ValueError("top-level YAML value must be a mapping")
    if "camera_model" in configuration:
        if camera_name is not None:
            raise ValueError("--camera cannot be used with a single-camera YAML")
        return "camera", configuration

    entries = [(name, value) for name, value in configuration.items()
               if isinstance(value, dict) and value.get("camera_model") == "omni"]
    if camera_name is not None:
        for name, camera in entries:
            if name == camera_name:
                return name, camera
        raise ValueError("omni camera '{}' was not found".format(camera_name))
    if len(entries) == 1:
        return entries[0]
    if not entries:
        raise ValueError("no omni camera entries found")
    raise ValueError("multiple omni cameras found; select one with --camera")
