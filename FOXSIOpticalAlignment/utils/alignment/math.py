import numpy as np
from scipy.ndimage import gaussian_filter

def calculate_intensity_weighted_centroid(
    x, y, roi_bounds, bins=150, plot_range=(-1000.0, 1000.0), sigma=2.0, threshold_fraction=0.5
):
    """
    Calculates the intensity-weighted centroid, spatial spreads, and threshold mask
    within a rectangular region of interest (ROI).
    """
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    if not 0 <= threshold_fraction <= 1:
        raise ValueError("threshold_fraction must be between 0 and 1.")

    xmin, xmax, ymin, ymax = roi_bounds
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("The ROI must have positive width and height.")

    event_mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    roi_x = x[event_mask]
    roi_y = y[event_mask]
    n_events = len(roi_x)

    if n_events == 0:
        raise ValueError("No flight events were found inside the selected ROI.")

    hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[plot_range, plot_range])
    smoothed = hist.astype(float) if sigma == 0 else gaussian_filter(hist.astype(float), sigma=sigma)

    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2

    x_in_roi = (x_centers >= xmin) & (x_centers <= xmax)
    y_in_roi = (y_centers >= ymin) & (y_centers <= ymax)
    roi_pixel_mask = x_in_roi[:, np.newaxis] & y_in_roi[np.newaxis, :]

    roi_values = smoothed[roi_pixel_mask]
    if roi_values.size == 0 or np.max(roi_values) <= 0:
        raise ValueError("No positive smoothed intensity was found inside the ROI.")

    threshold = threshold_fraction * float(np.max(roi_values))
    threshold_mask = roi_pixel_mask & (smoothed >= threshold) & (smoothed > 0)

    weights = np.where(threshold_mask, smoothed, 0.0)
    total_weight = float(np.sum(weights))

    if total_weight <= 0:
        raise ValueError("No positive histogram intensity passed the threshold.")

    centroid_x = float(np.sum(weights * x_centers[:, np.newaxis]) / total_weight)
    centroid_y = float(np.sum(weights * y_centers[np.newaxis, :]) / total_weight)

    variance_x = float(np.sum(weights * (x_centers[:, np.newaxis] - centroid_x) ** 2) / total_weight)
    variance_y = float(np.sum(weights * (y_centers[np.newaxis, :] - centroid_y) ** 2) / total_weight)

    std_x = float(np.sqrt(max(variance_x, 0.0)))
    std_y = float(np.sqrt(max(variance_y, 0.0)))

    return {
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "std_x": std_x,
        "std_y": std_y,
        "sem_x": np.nan,
        "sem_y": np.nan,
        "event_mean_x": float(np.mean(roi_x)),
        "event_mean_y": float(np.mean(roi_y)),
        "n_events": n_events,
        "smoothed_hist": smoothed,
        "hist": hist,
        "xedges": xedges,
        "yedges": yedges,
        "threshold": threshold,
        "threshold_mask": threshold_mask,
        "total_weight": total_weight,
        "n_threshold_pixels": int(np.count_nonzero(threshold_mask)),
        "roi_bounds": (xmin, xmax, ymin, ymax),
        "sigma": sigma,
        "threshold_fraction": threshold_fraction,
    }

def calculate_polar_offsets(centroid_x, centroid_y, optical_x, optical_y):
    """Calculates detector-center and optical-center offsets."""
    detector_radius = float(np.hypot(centroid_x, centroid_y))
    dx = centroid_x - optical_x
    dy = centroid_y - optical_y
    radial_offset = float(np.hypot(dx, dy))
    azimuth_deg = float(np.degrees(np.arctan2(dx, dy)) % 360)

    return {
        "detector_radius_arcsec": detector_radius,
        "offset_x_arcsec": float(dx),
        "offset_y_arcsec": float(dy),
        "radial_offset_arcsec": radial_offset,
        "radial_offset_arcmin": radial_offset / 60.0,
        "azimuth_deg": azimuth_deg,
    }