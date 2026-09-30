"""Fit a Kalibr pinhole-equidistant camera to an omni-radtan camera.

The fit is based on ray/pixel correspondences. Unit rays are sampled with
uniform solid-angle density on a spherical cap centered on the optical axis,
projected by the source omni-radtan model, and only projections inside the
source image are retained. A 90 degree cap is a front hemisphere; slightly
larger caps are also supported for fisheye-model fitting.
The target pinhole-equidistant parameters are then estimated by least squares
in pixel space.
"""

from __future__ import print_function

import math

import numpy as np
from scipy.optimize import least_squares


_GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))


def sample_hemisphere(count, max_angle_degrees=90.0):
    """Return deterministic spherical-cap rays uniform by solid angle."""
    if count < 1:
        raise ValueError("sample count must be positive")
    if not 0.0 < max_angle_degrees < 180.0:
        raise ValueError("max_angle_degrees must be in (0, 180)")

    # Midpoint sampling avoids landing exactly on either the 90-degree
    # pinhole boundary or the requested cap boundary.
    z_min = math.cos(math.radians(max_angle_degrees))
    indices = np.arange(count, dtype=np.float64)
    z = 1.0 - (indices + 0.5) * (1.0 - z_min) / float(count)
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    azimuth = indices * _GOLDEN_ANGLE
    return np.column_stack((radius * np.cos(azimuth),
                            radius * np.sin(azimuth), z))


def _camera_values(camera):
    """Validate a source camera dictionary and return numeric parameters."""
    if not isinstance(camera, dict):
        raise ValueError("camera entry must be a mapping")
    if camera.get("camera_model") != "omni":
        raise ValueError("camera_model must be 'omni'")
    if camera.get("distortion_model") != "radtan":
        raise ValueError("distortion_model must be 'radtan'")

    intrinsics = camera.get("intrinsics")
    distortion = camera.get("distortion_coeffs")
    resolution = camera.get("resolution")
    if not isinstance(intrinsics, (list, tuple)) or len(intrinsics) != 5:
        raise ValueError("omni intrinsics must be [xi, fu, fv, cu, cv]")
    if not isinstance(distortion, (list, tuple)) or len(distortion) != 4:
        raise ValueError("radtan distortion_coeffs must be [k1, k2, p1, p2]")
    if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
        raise ValueError("resolution must be [width, height]")

    values = np.asarray(list(intrinsics) + list(distortion), dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("intrinsics and distortion coefficients must be finite")
    xi, fu, fv, cu, cv, k1, k2, p1, p2 = values
    width, height = int(resolution[0]), int(resolution[1])
    if xi < 0.0:
        raise ValueError("omni xi must be non-negative")
    if fu <= 0.0 or fv <= 0.0:
        raise ValueError("source focal lengths must be positive")
    if width <= 0 or height <= 0:
        raise ValueError("resolution values must be positive")
    return xi, fu, fv, cu, cv, k1, k2, p1, p2, width, height


def project_omni_radtan(camera, rays):
    """Project rays with Kalibr's omni-radtan equations.

    Returns ``(pixels, valid)``. ``valid`` includes the omni projection domain,
    finite-value checks, and source image bounds.
    """
    (xi, fu, fv, cu, cv, k1, k2, p1, p2,
     width, height) = _camera_values(camera)
    rays = np.asarray(rays, dtype=np.float64)
    if rays.ndim != 2 or rays.shape[1] != 3:
        raise ValueError("rays must have shape (N, 3)")

    norm = np.linalg.norm(rays, axis=1)
    fov_parameter = xi if xi <= 1.0 else 1.0 / xi
    domain = (norm > 0.0) & (rays[:, 2] > -fov_parameter * norm)
    denominator = rays[:, 2] + xi * norm

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        x = rays[:, 0] / denominator
        y = rays[:, 1] / denominator
        x2 = x * x
        y2 = y * y
        xy = x * y
        r2 = x2 + y2
        radial = k1 * r2 + k2 * r2 * r2
        xd = x + x * radial + 2.0 * p1 * xy + p2 * (r2 + 2.0 * x2)
        yd = y + y * radial + 2.0 * p2 * xy + p1 * (r2 + 2.0 * y2)
        u = fu * xd + cu
        v = fv * yd + cv

    pixels = np.column_stack((u, v))
    finite = np.all(np.isfinite(pixels), axis=1)
    inside = ((u >= 0.0) & (u < width) &
              (v >= 0.0) & (v < height))
    return pixels, domain & finite & inside


def project_pinhole_equidistant(parameters, rays):
    """Project rays using Kalibr's pinhole + equidistant model.

    This polar-angle form is equivalent to pinhole projection followed by
    EquidistantDistortion, and remains stable near and beyond 90 degrees.
    """
    parameters = np.asarray(parameters, dtype=np.float64)
    if parameters.shape != (8,):
        raise ValueError("parameters must be [fu, fv, cu, cv, k1, k2, k3, k4]")
    rays = np.asarray(rays, dtype=np.float64)
    if rays.ndim != 2 or rays.shape[1] != 3:
        raise ValueError("rays must have shape (N, 3)")

    fu, fv, cu, cv = parameters[:4]
    coefficients = parameters[4:]
    rho = np.hypot(rays[:, 0], rays[:, 1])
    theta = np.arctan2(rho, rays[:, 2])
    direction_x = np.divide(rays[:, 0], rho, out=np.zeros_like(rho),
                            where=rho > 0.0)
    direction_y = np.divide(rays[:, 1], rho, out=np.zeros_like(rho),
                            where=rho > 0.0)
    theta2 = theta * theta
    polynomial = (1.0 + coefficients[0] * theta2
                  + coefficients[1] * theta2 ** 2
                  + coefficients[2] * theta2 ** 3
                  + coefficients[3] * theta2 ** 4)
    theta_distorted = theta * polynomial
    return np.column_stack((fu * direction_x * theta_distorted + cu,
                            fv * direction_y * theta_distorted + cv))


def _residuals(parameters, rays, source_pixels):
    return (project_pinhole_equidistant(parameters, rays) -
            source_pixels).reshape(-1)


def _jacobian(parameters, rays, source_pixels):
    del source_pixels
    fu, fv = parameters[:2]
    rho = np.hypot(rays[:, 0], rays[:, 1])
    theta = np.arctan2(rho, rays[:, 2])
    direction_x = np.divide(rays[:, 0], rho, out=np.zeros_like(rho),
                            where=rho > 0.0)
    direction_y = np.divide(rays[:, 1], rho, out=np.zeros_like(rho),
                            where=rho > 0.0)
    theta2 = theta * theta
    powers = np.column_stack((theta * theta2,
                              theta * theta2 ** 2,
                              theta * theta2 ** 3,
                              theta * theta2 ** 4))
    radial = theta + powers.dot(parameters[4:])

    jacobian = np.zeros((2 * rays.shape[0], 8), dtype=np.float64)
    jacobian[0::2, 0] = direction_x * radial
    jacobian[1::2, 1] = direction_y * radial
    jacobian[0::2, 2] = 1.0
    jacobian[1::2, 3] = 1.0
    jacobian[0::2, 4:] = fu * direction_x[:, None] * powers
    jacobian[1::2, 4:] = fv * direction_y[:, None] * powers
    return jacobian


def fit_camera(camera, sample_count=50000, max_angle_degrees=90.0,
               max_function_evaluations=200):
    """Fit one omni-radtan camera and return ``(converted, statistics)``."""
    (xi, source_fu, source_fv, source_cu, source_cv, _k1, _k2, _p1, _p2,
     _width, _height) = _camera_values(camera)
    if sample_count < 100:
        raise ValueError("at least 100 spherical-cap samples are required")

    candidate_rays = sample_hemisphere(sample_count, max_angle_degrees)
    candidate_pixels, valid = project_omni_radtan(camera, candidate_rays)
    rays = candidate_rays[valid]
    source_pixels = candidate_pixels[valid]
    if rays.shape[0] < 100:
        raise ValueError(
            "only {} sampled rays project inside the source image; increase "
            "--samples or check the calibration".format(rays.shape[0]))

    # Around the optical axis, omni projection has scale f / (1 + xi).
    initial = np.array([source_fu / (1.0 + xi),
                        source_fv / (1.0 + xi),
                        source_cu, source_cv, 0.0, 0.0, 0.0, 0.0])
    lower = np.array([np.finfo(np.float64).eps, np.finfo(np.float64).eps,
                      -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
    upper = np.full(8, np.inf, dtype=np.float64)
    result = least_squares(
        _residuals, initial, jac=_jacobian, args=(rays, source_pixels),
        bounds=(lower, upper), method="trf", loss="linear", x_scale="jac",
        ftol=1e-12, xtol=1e-12, gtol=1e-12,
        max_nfev=max_function_evaluations)
    if not result.success:
        raise RuntimeError("least-squares fitting failed: {}".format(result.message))

    residual_vectors = (project_pinhole_equidistant(result.x, rays) -
                        source_pixels)
    pixel_errors = np.linalg.norm(residual_vectors, axis=1)
    converted = dict(camera)
    converted["camera_model"] = "pinhole"
    converted["intrinsics"] = [float(value) for value in result.x[:4]]
    converted["distortion_model"] = "equidistant"
    converted["distortion_coeffs"] = [float(value) for value in result.x[4:]]

    statistics = {
        "candidate_samples": int(sample_count),
        "valid_samples": int(rays.shape[0]),
        "max_angle_degrees": float(max_angle_degrees),
        "sse_pixels_squared": float(np.sum(residual_vectors ** 2)),
        "rms_pixels": float(np.sqrt(np.mean(pixel_errors ** 2))),
        "mean_pixels": float(np.mean(pixel_errors)),
        "p95_pixels": float(np.percentile(pixel_errors, 95.0)),
        "max_pixels": float(np.max(pixel_errors)),
        "function_evaluations": int(result.nfev),
    }
    return converted, statistics


def projection_change_samples(source_camera, converted_camera,
                              candidate_count=50000,
                              max_angle_degrees=90.0, display_count=250):
    """Return a small, evenly selected set of before/after pixel positions."""
    if display_count < 1:
        raise ValueError("visualization point count must be positive")
    if converted_camera.get("camera_model") != "pinhole":
        raise ValueError("converted camera_model must be 'pinhole'")
    if converted_camera.get("distortion_model") != "equidistant":
        raise ValueError("converted distortion_model must be 'equidistant'")
    intrinsics = converted_camera.get("intrinsics")
    distortion = converted_camera.get("distortion_coeffs")
    if not isinstance(intrinsics, (list, tuple)) or len(intrinsics) != 4:
        raise ValueError("pinhole intrinsics must be [fu, fv, cu, cv]")
    if not isinstance(distortion, (list, tuple)) or len(distortion) != 4:
        raise ValueError("equidistant distortion_coeffs must contain four values")

    rays = sample_hemisphere(candidate_count, max_angle_degrees)
    source_pixels, valid = project_omni_radtan(source_camera, rays)
    parameters = np.asarray(list(intrinsics) + list(distortion),
                            dtype=np.float64)
    converted_pixels = project_pinhole_equidistant(parameters, rays)
    valid &= np.all(np.isfinite(converted_pixels), axis=1)
    rays = rays[valid]
    source_pixels = source_pixels[valid]
    converted_pixels = converted_pixels[valid]
    if rays.shape[0] == 0:
        raise ValueError("no valid projection correspondences to visualize")

    shown = min(int(display_count), rays.shape[0])
    indices = np.linspace(0, rays.shape[0] - 1, shown).astype(np.int64)
    return (rays[indices], source_pixels[indices],
            converted_pixels[indices])


def plot_projection_changes(camera_pairs, candidate_count=50000,
                            max_angle_degrees=90.0, display_count=250,
                            output_path=None, show=False):
    """Plot actual projections and a readable displacement vector field.

    ``camera_pairs`` contains ``(label, source, converted)`` tuples. The left
    panel uses the true before/after pixel positions. The right panel magnifies
    only the arrows (with the factor stated in its title) so subpixel changes
    remain visible; arrow color always represents the true pixel displacement.
    """
    if not output_path and not show:
        raise ValueError("either output_path or show must be requested")

    # Keep save-only use functional on headless calibration machines.
    import matplotlib
    if output_path and not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    camera_pairs = list(camera_pairs)
    if not camera_pairs:
        raise ValueError("no converted cameras to visualize")
    figure, axes = plt.subplots(len(camera_pairs), 2,
                               figsize=(15.0, 6.5 * len(camera_pairs)),
                               squeeze=False)
    for row, pair in enumerate(camera_pairs):
        position_axis, vector_axis = axes[row]
        label, source_camera, converted_camera = pair
        _rays, source_pixels, converted_pixels = projection_change_samples(
            source_camera, converted_camera,
            candidate_count=candidate_count,
            max_angle_degrees=max_angle_degrees,
            display_count=display_count)
        displacement = converted_pixels - source_pixels
        errors = np.linalg.norm(displacement, axis=1)
        segments = np.stack((source_pixels, converted_pixels), axis=1)
        lines = LineCollection(segments, cmap="viridis", linewidths=1.0,
                               alpha=0.85)
        lines.set_array(errors)
        position_axis.add_collection(lines)
        position_axis.scatter(
            source_pixels[:, 0], source_pixels[:, 1], s=24,
            facecolors="none", edgecolors="#1769aa", linewidths=0.8,
            label="omni-radtan")
        position_axis.scatter(
            converted_pixels[:, 0], converted_pixels[:, 1], s=18,
            marker="x", color="#d84315", linewidths=0.8,
            label="pinhole-equidistant")

        # Make the 95th-percentile arrow roughly 20 pixels long. This changes
        # only arrow rendering, never the fitted data or reported error.
        reference_error = max(float(np.percentile(errors, 95.0)), 1e-12)
        arrow_factor = min(500.0, max(1.0, 20.0 / reference_error))
        vector_axis.scatter(source_pixels[:, 0], source_pixels[:, 1], s=5,
                            color="0.75", alpha=0.5)
        arrows = vector_axis.quiver(
            source_pixels[:, 0], source_pixels[:, 1],
            displacement[:, 0] * arrow_factor,
            displacement[:, 1] * arrow_factor,
            errors, cmap="viridis", angles="xy", scale_units="xy", scale=1.0,
            width=0.004, headwidth=3.5, headlength=4.5)

        width, height = source_camera["resolution"]
        for axis in (position_axis, vector_axis):
            axis.set_xlim(0.0, float(width))
            axis.set_ylim(float(height), 0.0)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("u [px]")
            axis.set_ylabel("v [px]")
            axis.grid(True, color="0.9", linewidth=0.6)
        position_axis.set_title(
            "{}: actual before/after positions ({} of {} candidates)".format(
                label, source_pixels.shape[0], candidate_count))
        position_axis.legend(loc="best")
        vector_axis.set_title(
            "displacement field (arrows x{:.1f}; RMS {:.4f} px)".format(
                arrow_factor, math.sqrt(np.mean(errors ** 2))))
        colorbar = figure.colorbar(arrows, ax=vector_axis,
                                  fraction=0.035, pad=0.02)
        colorbar.set_label("true pixel displacement [px]")

    figure.tight_layout()
    if output_path:
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    return figure


def _camera_entries(configuration):
    """Return camera entries for a single-camera or camchain YAML tree."""
    if not isinstance(configuration, dict):
        raise ValueError("top-level YAML value must be a mapping")
    if "camera_model" in configuration:
        return [(None, configuration)]
    return [(name, value) for name, value in configuration.items()
            if isinstance(value, dict) and "camera_model" in value]


def convert_configuration(configuration, camera_name=None, **fit_options):
    """Return a converted copy of a Kalibr YAML configuration and fit reports."""
    output = dict(configuration)
    entries = _camera_entries(configuration)
    if not entries:
        raise ValueError("no camera entries found in the YAML configuration")

    if camera_name is not None:
        selected = [(name, camera) for name, camera in entries
                    if name == camera_name]
        if not selected:
            if entries[0][0] is None:
                raise ValueError("--camera cannot be used with a single-camera YAML")
            raise ValueError("camera '{}' was not found".format(camera_name))
    else:
        selected = [(name, camera) for name, camera in entries
                    if camera.get("camera_model") == "omni" and
                    camera.get("distortion_model") == "radtan"]
        if not selected:
            raise ValueError("no omni-radtan camera entries found")

    reports = {}
    for name, camera in selected:
        label = name if name is not None else "camera"
        converted, statistics = fit_camera(camera, **fit_options)
        if name is None:
            output = converted
        else:
            output[name] = converted
        reports[label] = statistics
    return output, reports
