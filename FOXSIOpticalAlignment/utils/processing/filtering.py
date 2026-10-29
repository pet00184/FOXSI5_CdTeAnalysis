import numpy as np
import cdte_tools_py.calibration.filter_convert_extract as fce

def filter_cdte_events(data, qualityflag=True, energy_range=None):
    """Applies quality flag (flag_quality == 1) and optional energy range filters."""
    filtered = data.copy()
    if qualityflag and "flag_quality" in filtered.colnames:
        filtered = filtered[filtered["flag_quality"] == 1]

    if energy_range is not None:
        filtered = fce.energy_range_filter(filtered.copy(), *energy_range)

    x = np.asarray(filtered["solar_x_unaligned"], dtype=float)
    y = np.asarray(filtered["solar_y_unaligned"], dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    return x[valid], y[valid]

def build_solar_histogram(x, y, bins=150, plot_range=(-1000.0, 1000.0)):
    """Generates a 2D histogram in Solar-X / Solar-Y coordinates."""
    hist, xedges, yedges = np.histogram2d(
        x, y, bins=bins, range=[plot_range, plot_range]
    )
    return hist, xedges, yedges