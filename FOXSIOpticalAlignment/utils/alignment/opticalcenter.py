import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter

from utils.processing.filtering import filter_cdte_events, build_solar_histogram

class FOXSIAlignmentImage:
    """
    Prepare and plot a FOXSI CdTe X-ray alignment image using unaligned solar coordinates.
    """

    def __init__(self, cdte_data, cdte):
        self.cdte_data = cdte_data
        self.cdte = cdte

        valid_detectors = ("CdTe1", "CdTe3", "CdTe4", "CdTe5")
        if cdte not in valid_detectors:
            raise ValueError(f"cdte must be one of {valid_detectors}; received {cdte!r}")

        self.hist = None
        self.xedges = None
        self.yedges = None
        self.x = None
        self.y = None

        self.centroid_x = None
        self.centroid_y = None
        self.smoothed_hist = None
        self.centroid_sigma = None
        self.centroid_threshold_fraction = None
        self.centroid_x_range = None
        self.centroid_y_range = None

        self.circle_radius = None
        self.centroid_circle = None

        self.fig = None
        self.ax = None
        self.image = None
        self.origin_artist = None
        self.centroid_artist = None

    def filter_events(self, qualityflag=True, energy_range=None):
        """Apply optional quality and energy filters."""
        return filter_cdte_events(self.cdte_data, qualityflag=qualityflag, energy_range=energy_range)

    def prepare_image(self, qualityflag=True, energy_range=None, bins=200, plot_range=(-1000.0, 1000.0)):
        if bins < 1:
            raise ValueError("bins must be a positive integer.")
        if len(plot_range) != 2 or plot_range[0] >= plot_range[1]:
            raise ValueError("plot_range must be (minimum, maximum), with minimum < maximum.")

        required_columns = ("solar_x_unaligned", "solar_y_unaligned")
        missing = [col for col in required_columns if col not in self.cdte_data.colnames]
        if missing:
            raise ValueError(f"Input data missing required columns: {missing}.")

        self.x, self.y = self.filter_events(qualityflag=qualityflag, energy_range=energy_range)
        self.hist, self.xedges, self.yedges = build_solar_histogram(self.x, self.y, bins=bins, plot_range=plot_range)

        self.centroid_x = None
        self.centroid_y = None
        self.smoothed_hist = None
        return self.hist, self.xedges, self.yedges

    def smooth_image(self, sigma=2.0):
        if self.hist is None:
            raise RuntimeError("Call prepare_image() before smoothing.")
        if sigma < 0:
            raise ValueError("sigma must be non-negative.")
        return self.hist.astype(float, copy=True) if sigma == 0 else gaussian_filter(self.hist.astype(float), sigma=sigma)

    def find_centroid(self, sigma=2.0, threshold_fraction=0.5, x_range=(-120.0, 120.0), y_range=(-120.0, 120.0)):
        if self.hist is None:
            raise RuntimeError("Call prepare_image() before finding the centroid.")

        smoothed = self.smooth_image(sigma=sigma)
        x_centers = (self.xedges[:-1] + self.xedges[1:]) / 2
        y_centers = (self.yedges[:-1] + self.yedges[1:]) / 2

        x_in_region = (x_centers >= x_range[0]) & (x_centers <= x_range[1])
        y_in_region = (y_centers >= y_range[0]) & (y_centers <= y_range[1])
        region_mask = x_in_region[:, np.newaxis] & y_in_region[np.newaxis, :]

        region_values = smoothed[region_mask]
        if region_values.size == 0 or np.max(region_values) <= 0:
            raise ValueError("No positive image intensity found inside centroid search region.")

        threshold = threshold_fraction * np.max(region_values)
        mask = region_mask & (smoothed >= threshold)
        weights = smoothed * mask
        total_weight = np.sum(weights)

        if total_weight <= 0:
            raise ValueError("No pixels passed the centroid threshold.")

        self.centroid_x = np.sum(weights * x_centers[:, np.newaxis]) / total_weight
        self.centroid_y = np.sum(weights * y_centers[np.newaxis, :]) / total_weight

        self.smoothed_hist = smoothed
        self.centroid_sigma = sigma
        self.centroid_threshold_fraction = threshold_fraction
        self.centroid_x_range = tuple(x_range)
        self.centroid_y_range = tuple(y_range)

        print(f"{self.cdte} centroid: Solar-X = {self.centroid_x:.2f} arcsec, Solar-Y = {self.centroid_y:.2f} arcsec")
        return self.centroid_x, self.centroid_y, smoothed

    def plot(self, ax=None, title=None, colorbar=True, xrange=None, yrange=None, rings=True):
        if self.hist is None:
            raise RuntimeError("Call prepare_image() before plotting.")

        fig = ax.figure if ax is not None else plt.subplots(figsize=(9, 8))[0]
        ax = ax or fig.gca()

        vmax = float(self.hist.max()) if self.hist.size else 0.0
        norm = LogNorm(vmin=1, vmax=vmax) if vmax > 1 else None
        extent = [self.xedges[0], self.xedges[-1], self.yedges[0], self.yedges[-1]]

        im = ax.imshow(self.hist.T, origin="lower", extent=extent, aspect="equal", cmap="magma", norm=norm, interpolation="nearest") if self.hist.sum() > 0 else None

        if xrange is not None: ax.set_xlim(-xrange, xrange)
        if yrange is not None: ax.set_ylim(-yrange, yrange)

        if rings:
            ring_limit = min(abs(self.xedges[0]), abs(self.xedges[-1]), abs(self.yedges[0]), abs(self.yedges[-1]))
            for radius_arcsec in np.arange(120, ring_limit, 120):
                ax.add_patch(Circle((0, 0), radius_arcsec, fill=False, color="cyan", linewidth=1.3))

        self.origin_artist = ax.plot(0, 0, marker="+", color="cyan", markersize=9, markeredgewidth=1.5, linestyle="none", label="(0, 0)", zorder=10)[0]
        ax.set_title(title or f"{self.cdte} X-ray Alignment Image")
        ax.set_xlabel("Solar-X [arcsec]")
        ax.set_ylabel("Solar-Y [arcsec]")
        if colorbar and im is not None: fig.colorbar(im, ax=ax, label="Counts", fraction=0.046, pad=0.04)

        self.fig, self.ax, self.image = fig, ax, im
        return fig, ax, im

    def plot_with_centroid(self, sigma=2.0, threshold_fraction=0.5, x_range=(-120.0, 120.0), y_range=(-120.0, 120.0), circle_radius=60.0, ax=None, title=None, colorbar=True, cross_size=9, xrange=None, yrange=None, rings=True):
        self.find_centroid(sigma=sigma, threshold_fraction=threshold_fraction, x_range=x_range, y_range=y_range)
        self.circle_radius = circle_radius

        fig, ax, im = self.plot(ax=ax, title=title, colorbar=colorbar, xrange=xrange, yrange=yrange, rings=rings)
        self.centroid_artist = ax.plot(self.centroid_x, self.centroid_y, marker="+", color="lime", markersize=cross_size, markeredgewidth=1.8, linestyle="none", label="Centroid", zorder=11)[0]
        self.centroid_circle = Circle((self.centroid_x, self.centroid_y), radius=circle_radius, fill=False, edgecolor="lime", linestyle="--", linewidth=1.5, zorder=9)

        ax.add_patch(self.centroid_circle)
        ax.legend()
        return fig, ax, im