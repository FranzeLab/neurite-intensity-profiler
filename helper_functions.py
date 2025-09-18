import numpy as np
import imageio
from scipy.ndimage import map_coordinates
import xml.etree.ElementTree as ET
import gzip
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import pandas as pd
from scipy.optimize import curve_fit


def load_tiff_channels(tiff_path):
    img = imageio.volread(tiff_path)  # Expected shape: (channels, height, width)
    if img.shape[0] != 2:
        raise ValueError("Expected a 2-channel image.")
    return img[0], img[1]


def load_and_filter_traces(traces_path):
    traces, metadata = load_snt_traces(traces_path)
    filtered = [(t, m) for t, m in zip(traces, metadata)
                if len(t) >= 2 and m.get('primary') == 'true']
    return zip(*filtered)


def load_snt_traces(traces_file):
    try:
        with gzip.open(traces_file, 'rt', encoding='utf-8') as f:
            tree = ET.parse(f)
    except (OSError, ET.ParseError):
        with open(traces_file, 'r', encoding='utf-8') as f:
            tree = ET.parse(f)

    root = tree.getroot()
    traces = []
    metadata = []

    for path in root.findall(".//path"):
        # Parse metadata
        tags = path.attrib.copy()

        # Parse 2D image coordinates
        points = []
        for pt in path.findall(".//point"):
            x = pt.attrib.get('x')
            y = pt.attrib.get('y')
            if x is None or y is None:
                continue
            try:
                points.append([float(x), float(y)])
            except ValueError:
                continue

        if points:
            traces.append(np.array(points))
            metadata.append(tags)

    return traces, metadata


def get_neurite_types(metadata):
    return [
        "axon" if m.get("swctype") == "2"
        else "dendrite" if m.get("swctype") == "3"
        else "Unknown"
        for m in metadata
    ]


def interpolate_and_smooth(path, window, spacing, order=3):
    if path.shape[0] < window or window <= order:
        return path

    # Smooth x and y
    x = savgol_filter(path[:, 0], window, order)
    y = savgol_filter(path[:, 1], window, order)
    smoothed = np.column_stack((x, y))

    # Compute cumulative arc length
    deltas = np.diff(smoothed, axis=0)
    dist = np.sqrt((deltas ** 2).sum(axis=1))
    arc_length = np.insert(np.cumsum(dist), 0, 0)

    # Resample at uniform spacing (default: 1 pixel)
    n_samples = int(np.floor(arc_length[-1] / spacing)) + 1
    uniform_dist = np.linspace(0, arc_length[-1], n_samples)

    # Interpolate x and y separately
    interp_x = interp1d(arc_length, smoothed[:, 0], kind='linear')
    interp_y = interp1d(arc_length, smoothed[:, 1], kind='linear')
    resampled = np.column_stack((interp_x(uniform_dist), interp_y(uniform_dist)))

    return resampled


def compute_normals(trace, n_samples):
    normals = []
    for i in range(len(trace)):
        start = max(i - n_samples // 2, 0)
        end = min(i + n_samples // 2 + 1, len(trace))
        window = trace[start:end]

        if len(window) < 3:
            normals.append(np.array([np.nan, np.nan]))
            continue

        coords = window - window.mean(axis=0)
        _, _, vh = np.linalg.svd(coords, full_matrices=False)
        tangent = vh[0]
        tangent /= np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]])
        normals.append(normal)

    return np.array(normals)


def sample_profile_along_normal(image, point, normal, width_px, mode='constant'):
    offsets = np.linspace(-width_px / 2, width_px / 2, width_px)
    coords = [(point + offset * normal)[::-1] for offset in offsets]
    coords = np.array(coords).T  # shape (2, N)
    profile = map_coordinates(image, coords, order=1, mode=mode)
    return profile


def average_profile(image, trace, normals, width, n_samples):
    profiles = []
    for i in range(len(trace)):
        start = max(i - n_samples // 2, 0)
        end = min(i + n_samples // 2 + 1, len(trace))
        window = trace[start:end]
        normal = normals[i]

        if np.isnan(normal).any() or len(window) < 3:
            profiles.append(np.full(width, np.nan))
            continue

        sampled_profiles = [sample_profile_along_normal(image, pt, normal, width_px=width) for pt in window]
        avg_profile = np.mean(sampled_profiles, axis=0)
        profiles.append(avg_profile)

    return profiles


def fit_gaussian(x_vals, profile, max_sigma, min_sigma=0.5, r_squared_lim=0.5):
    # Detrend profile using a linear fit
    coeffs = np.polyfit(x_vals, profile, deg=1)
    profile_detrended = profile - (coeffs[0] * x_vals + coeffs[1])

    # Gaussian model
    def gaussian(x, a, x0, sigma):
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    # Initial guess
    a0 = profile_detrended.max()
    x0 = 0
    sigma0 = max_sigma / 2
    p0 = [a0, x0, sigma0]

    bounds = (
        [0, -max_sigma, min_sigma],
        [np.inf, max_sigma, max_sigma]
    )

    popt, _ = curve_fit(gaussian, x_vals, profile_detrended, p0=p0, bounds=bounds)
    a_fit, x0_fit, sigma_fit = popt

    if abs(x0_fit) > max_sigma / 2:
        return np.nan, np.nan, f"Asymmetry: {x0_fit}"

    # Fit quality check
    residual = profile_detrended - gaussian(x_vals, *popt)
    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum((profile_detrended - np.mean(profile_detrended)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    if r_squared < r_squared_lim:
        return np.nan, np.nan, f"FitQuality: {r_squared}"

    return sigma_fit, x0_fit, "Passed"


def fit_profiles(profiles, fit_width, min_sigma, r_squared_lim):
    sigmas = []
    translations = []
    fit_results = []

    x_vals = np.linspace(-fit_width / 2, fit_width / 2, fit_width)

    for profile in profiles:
        if np.isnan(profile).any():
            sigmas.append(np.nan)
            translations.append(np.nan)
            fit_results.append("NaN in profile")
            continue
        try:
            sigma, x0, mssg = fit_gaussian(x_vals, profile, max_sigma=fit_width / 4,
                                           min_sigma=min_sigma, r_squared_lim=r_squared_lim)
            sigmas.append(sigma)
            translations.append(x0)
            fit_results.append(mssg)
        except RuntimeError:
            sigmas.append(np.nan)
            translations.append(np.nan)
            fit_results.append("RuntimeError")

    return sigmas, translations, fit_results


def measure_intensities(image, trace, normals, widths):
    intensities = []
    for pt, normal, width in zip(trace, normals, widths):
        if np.isnan(width) or np.isnan(normal).any():
            intensities.append(np.nan)
            continue

        width_px = int(width)
        profile = sample_profile_along_normal(image, pt, normal, width_px)
        intensities.append(np.max(profile))

    return intensities


def fit_trace_width(image, trace, fit_width, window, min_width=1, r_squared_lim=0.5):
    assert window % 2 == 1, "n_samples must be an odd number"

    normals = compute_normals(trace, window)
    profiles = average_profile(image, trace, normals, fit_width, window)
    sigmas, translations, fit_results = fit_profiles(profiles, fit_width, min_width, r_squared_lim)

    avg_sigma = np.nanmedian(sigmas)
    filled_sigmas = [avg_sigma if np.isnan(s) else s for s in sigmas]  # Replace missing sigmas
    scale_fwhm = 2 * np.sqrt(2 * np.log(2))
    scaled_widths = [scale_fwhm * s for s in filled_sigmas]  # Scale sigmas to widths
    intensities = measure_intensities(image, trace, normals, scaled_widths)

    return intensities, scaled_widths, translations, fit_results, normals


def translate_trace(trace, translation, normal):
    out = []
    for pt, t, n in zip(trace, translation, normal):
        if np.isnan(t) or np.isnan(n).any():
            out.append(pt)
        else:
            out.append(pt + n * t)
    return np.array(out)


def assemble_dataframe(traces, types, withs, translations, normals, intensities, fit_mssgs):
    all_rows = []
    for tid, (trace, w, t, n, ints, fit) in enumerate(zip(traces, withs, translations, normals, intensities, fit_mssgs)):
        for i, (pt, w_loc, t_loc, n_loc, i_loc, fit_loc) in enumerate(zip(trace, w, t, n, ints, fit)):
            if np.isnan(t_loc) or np.isnan(n_loc[0]) or np.isnan(n_loc[1]):
                new_x, new_y = pt
            else:
                new_x, new_y = pt + n_loc * t_loc
            all_rows.append([
                tid, types[tid], i, pt[0], pt[1], n_loc[0], n_loc[1],
                new_x, new_y, w_loc, i_loc, fit_loc
            ])
    return pd.DataFrame(all_rows, columns=[
        "trace_id", "neurite", "point_index", "x", "y",
        "normal_x", "normal_y", "new_x", "new_y",
        "width", "intensity", "fit_mssg"
    ])
