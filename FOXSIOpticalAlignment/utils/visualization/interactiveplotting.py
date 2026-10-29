import sys
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import RectangleSelector

from utils.processing.filtering import filter_cdte_events, build_solar_histogram
from utils.alignment.math import calculate_intensity_weighted_centroid, calculate_polar_offsets


class InteractiveCdTeFlightCentroid:
    """Interactively select a rectangular ROI in a FOXSI CdTe flight image."""

    def __init__(self, cdtedict):
        self.cdtedict = cdtedict

    @staticmethod
    def _to_arcsec(value):
        return float(value.to(u.arcsec).value) if hasattr(value, "to") else float(value)

    def _get_flight_events(self, target, cdtekey, qualityflag=True, energy_range=None):
        data = self.cdtedict[cdtekey][target]["Data"]
        return filter_cdte_events(data, qualityflag=qualityflag, energy_range=energy_range)

    def interactive_flight_centroid(
        self,
        target,
        cdtekey,
        optical_center,
        bins=150,
        energy_range=None,
        qualityflag=True,
        xrange=None,
        yrange=None,
        ring_spacing=120,
        cmap="viridis",
        sigma=2.0,
        threshold_fraction=0.5,
    ):
        optical_x = self._to_arcsec(optical_center[0])
        optical_y = self._to_arcsec(optical_center[1])

        x, y = self._get_flight_events(
            target=target, cdtekey=cdtekey, qualityflag=qualityflag, energy_range=energy_range
        )
        plot_range = (-1000.0, 1000.0)
        hist, xedges, yedges = build_solar_histogram(x, y, bins=bins, plot_range=plot_range)

        fig, ax = plt.subplots(figsize=(10, 8))
        vmax = float(hist.max()) if hist.size else 0.0
        norm = LogNorm(vmin=1, vmax=vmax) if vmax > 1 else None

        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        if hist.sum() > 0:
            image = ax.imshow(
                hist.T,
                origin="lower",
                extent=extent,
                aspect="equal",
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
            )
            fig.colorbar(image, ax=ax, label="Counts", fraction=0.046, pad=0.04)
        else:
            ax.text(0.5, 0.5, "No Data", transform=ax.transAxes, color="red", fontsize=14, ha="center", va="center")

        # Set axis limits prior to drawing concentric rings and text annotations
        if xrange is not None:
            ax.set_xlim(-xrange, xrange)
        if yrange is not None:
            ax.set_ylim(-yrange, yrange)

        # Draw red concentric rings centered on the optical center
        ring_limit = min(
            abs(xedges[0] - optical_x),
            abs(xedges[-1] - optical_x),
            abs(yedges[0] - optical_y),
            abs(yedges[-1] - optical_y),
        )

        for radius in np.arange(ring_spacing, ring_limit, ring_spacing):
            ax.add_patch(
                Circle(
                    (optical_x, optical_y),
                    radius,
                    fill=False,
                    color="red",
                    linewidth=1.3,
                    zorder=8,
                )
            )

            label_x = optical_x
            label_y = optical_y + radius + 8
            ax.text(
                label_x,
                label_y,
                f"{radius / 60:.0f}′",
                color="red",
                fontsize=8,
                ha="center",
                va="bottom",
                clip_on=True,
                zorder=9,
            )

        # Mark detector origin and optical center
        ax.plot(
            0,
            0,
            marker="+",
            color="cyan",
            markersize=10,
            markeredgewidth=1.7,
            linestyle="none",
            label="Detector center (0, 0)",
            zorder=10,
        )
        ax.plot(
            optical_x,
            optical_y,
            marker="+",
            color="red",
            markersize=10,
            markeredgewidth=1.7,
            linestyle="none",
            label="Optical center",
            zorder=10,
        )

        ax.set_title(f"{cdtekey} Flight Centroid — Target {target}")
        ax.set_xlabel("Solar-X [arcsec]")
        ax.set_ylabel("Solar-Y [arcsec]")
        ax.legend(loc="upper right")

        # Top-left persistent annotation box
        annotation = ax.text(
            0.02,
            0.98,
            f"Draw a rectangle around the source.\nGaussian sigma = {sigma:g} px\nThreshold fraction = {threshold_fraction:g}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            color="white",
            fontsize=9,
            bbox={"facecolor": "black", "alpha": 0.75, "edgecolor": "none"},
            zorder=25,
        )

        fig.source_offset = None
        fig.centroid_results = None
        fig._roi_artists = []

        def onselect(eclick, erelease):
            if any(v is None for v in (eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata)):
                return

            xmin, xmax = sorted([float(eclick.xdata), float(erelease.xdata)])
            ymin, ymax = sorted([float(eclick.ydata), float(erelease.ydata)])

            if xmin == xmax or ymin == ymax:
                print("Please select an ROI with nonzero width and height.")
                sys.stdout.flush()
                return

            try:
                results = calculate_intensity_weighted_centroid(
                    x=x,
                    y=y,
                    roi_bounds=(xmin, xmax, ymin, ymax),
                    bins=bins,
                    plot_range=plot_range,
                    sigma=sigma,
                    threshold_fraction=threshold_fraction,
                )
            except ValueError as exc:
                print(f"Centroid calculation failed: {exc}")
                sys.stdout.flush()
                return

            centroid_x, centroid_y = results["centroid_x"], results["centroid_y"]
            offsets = calculate_polar_offsets(centroid_x, centroid_y, optical_x, optical_y)

            # Clear previous ROI drawings
            for artist in fig._roi_artists:
                try:
                    artist.remove()
                except ValueError:
                    pass

            roi_patch = Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                edgecolor="lime",
                linewidth=1.5,
                zorder=12,
            )
            ax.add_patch(roi_patch)

            offset_line, = ax.plot(
                [optical_x, centroid_x],
                [optical_y, centroid_y],
                color="yellow",
                linewidth=1.5,
                zorder=13,
            )

            centroid_artist, = ax.plot(
                centroid_x,
                centroid_y,
                marker="+",
                color="lime",
                markersize=12,
                markeredgewidth=2,
                linestyle="none",
                zorder=14,
            )

            fig._roi_artists = [roi_patch, offset_line, centroid_artist]

            # Update top-left text box
            annotation.set_text(
                f"{cdtekey} | Target {target}\n"
                f"Events in ROI: {results['n_events']}\n"
                f"Centroid X: {centroid_x:.2f} arcsec\n"
                f"Centroid Y: {centroid_y:.2f} arcsec\n"
                f"Standard deviation X: {results['std_x']:.2f} arcsec\n"
                f"Standard deviation Y: {results['std_y']:.2f} arcsec\n"
                f"Offset X: {offsets['offset_x_arcsec']:.2f} arcsec\n"
                f"Offset Y: {offsets['offset_y_arcsec']:.2f} arcsec\n"
                f"Radial offset: {offsets['radial_offset_arcsec']:.2f} arcsec\n"
                f"Azimuth: {offsets['azimuth_deg']:.1f}°"
            )

            # Print results to console and flush stdout immediately
            print(f"\n{cdtekey} — Target {target}")
            print(f"Intensity-weighted centroid: X = {centroid_x:.2f}, Y = {centroid_y:.2f} arcsec")
            print(f"ROI: X = [{xmin:.2f}, {xmax:.2f}], Y = [{ymin:.2f}, {ymax:.2f}] arcsec")
            print(f"Raw events inside ROI: {results['n_events']}")
            print(f"Thresholded weighted standard deviation: X = {results['std_x']:.2f}, Y = {results['std_y']:.2f} arcsec")
            print(f"Offset from optical center: dX = {offsets['offset_x_arcsec']:.2f}, dY = {offsets['offset_y_arcsec']:.2f} arcsec")
            print(f"Radial offset: {offsets['radial_offset_arcsec']:.2f} arcsec ({offsets['radial_offset_arcmin']:.3f} arcmin)")
            print(f"Azimuth: {offsets['azimuth_deg']:.1f}° (0° = +Solar-Y; clockwise toward +Solar-X)")
            sys.stdout.flush()

            fig.canvas.draw_idle()

        selector = RectangleSelector(
            ax,
            onselect,
            useblit=True,
            button=[1],
            minspanx=1,
            minspany=1,
            spancoords="data",
            interactive=True,
            props={"facecolor": "lime", "edgecolor": "lime", "alpha": 0.2, "fill": True},
        )
        fig.roi_selector = selector

        plt.tight_layout()
        plt.show()
        return fig, ax


class InteractiveCdTeOpticalOffset:
    """Interactive flight-data plot with concentric circles centered on optical center."""

    CDTE_LIST = ["CdTe1", "CdTe3", "CdTe4", "CdTe5"]

    def __init__(self, cdtedict):
        self.cdtedict = cdtedict

    def make_cdte_histogram(self, target, cdtekey, bins=150, energy_range=None, qualityflag=True):
        cdtedata = self.cdtedict[cdtekey].get(target, {}).get("Data")
        if cdtedata is None or len(cdtedata) == 0:
            edges = np.linspace(-1000, 1000, bins + 1)
            return np.zeros((bins, bins)), edges, edges

        x, y = filter_cdte_events(cdtedata, qualityflag=qualityflag, energy_range=energy_range)
        image, xedges, yedges = build_solar_histogram(x, y, bins=bins, plot_range=(-1000, 1000))
        return image.T, xedges, yedges

    def interactive_optical_offset(
        self,
        target,
        cdtekey,
        optical_center,
        bins=150,
        energy_range=None,
        qualityflag=True,
        xrange=None,
        yrange=None,
        ring_spacing=120,
        cmap="viridis",
    ):
        optical_x, optical_y = [
            float(v.to_value(u.arcsec) if isinstance(v, u.Quantity) else v) for v in optical_center
        ]
        image, xedges, yedges = self.make_cdte_histogram(
            target, cdtekey, bins=bins, energy_range=energy_range, qualityflag=qualityflag
        )

        fig, ax = plt.subplots(figsize=(9, 8))
        vmax = image.max()
        norm = LogNorm(vmin=1, vmax=vmax) if vmax > 1 else None

        im = ax.imshow(
            image,
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            aspect="equal",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )

        if xrange is not None:
            ax.set_xlim(-xrange, xrange)
        if yrange is not None:
            ax.set_ylim(-yrange, yrange)

        # Draw red concentric rings centered on optical center
        ring_limit = max(
            0,
            min(
                optical_x - xedges[0],
                xedges[-1] - optical_x,
                optical_y - yedges[0],
                yedges[-1] - optical_y,
            ),
        )

        for radius_arcsec in np.arange(ring_spacing, ring_limit, ring_spacing):
            ax.add_patch(
                Circle((optical_x, optical_y), radius_arcsec, fill=False, color="red", linewidth=1.3, zorder=8)
            )

            label_x = optical_x
            label_y = optical_y + radius_arcsec + 8
            ax.text(
                label_x,
                label_y,
                f"{radius_arcsec / 60:.0f}′",
                color="red",
                fontsize=8,
                ha="center",
                va="bottom",
                clip_on=True,
                zorder=9,
            )

        ax.plot(
            0,
            0,
            marker="+",
            color="cyan",
            markersize=10,
            markeredgewidth=1.5,
            linestyle="none",
            label="Detector center",
            zorder=10,
        )
        ax.plot(
            optical_x,
            optical_y,
            marker="+",
            color="red",
            markersize=11,
            markeredgewidth=2,
            linestyle="none",
            label="Optical center",
            zorder=10,
        )

        peak_marker, = ax.plot([], [], marker="o", color="yellow", markeredgecolor="black", markersize=8, linestyle="none", zorder=15)
        radial_line, = ax.plot([], [], color="yellow", linestyle="--", linewidth=1.3, zorder=14)

        info = ax.text(
            0.03,
            0.97,
            "Click a source location",
            transform=ax.transAxes,
            color="white",
            fontsize=10,
            ha="left",
            va="top",
            bbox=dict(facecolor="black", alpha=0.70, edgecolor="none"),
            zorder=25,
        )

        ax.set_title(f"{target} — {cdtekey}: click to measure optical offset")
        ax.set_xlabel("Solar-X [arcsec]")
        ax.set_ylabel("Solar-Y [arcsec]")
        ax.legend(loc="lower right")
        fig.colorbar(im, ax=ax, label="Counts", fraction=0.046, pad=0.04)

        def on_click(event):
            if event.inaxes != ax or event.xdata is None or event.ydata is None:
                return

            x, y = float(event.xdata), float(event.ydata)
            offsets = calculate_polar_offsets(x, y, optical_x, optical_y)

            peak_marker.set_data([x], [y])
            radial_line.set_data([optical_x, x], [optical_y, y])

            info.set_text(
                f"Detector-center distance: {offsets['detector_radius_arcsec']:.1f} arcsec\n"
                f"Optical-center ΔX: {offsets['offset_x_arcsec']:+.1f} arcsec\n"
                f"Optical-center ΔY: {offsets['offset_y_arcsec']:+.1f} arcsec\n"
                f"Optical radial distance: {offsets['radial_offset_arcsec']:.1f} arcsec\n"
                f"Optical radial distance: {offsets['radial_offset_arcmin']:.2f} arcmin\n"
                f"Azimuth from north: {offsets['azimuth_deg']:.1f}°"
            )

            print(f"\n{target} — {cdtekey}")
            print(f"Clicked position: ({x:.2f}, {y:.2f}) arcsec")
            print(f"Distance from detector center: {offsets['detector_radius_arcsec']:.2f} arcsec")
            print(f"Optical-center ΔX: {offsets['offset_x_arcsec']:+.2f} arcsec")
            print(f"Optical-center ΔY: {offsets['offset_y_arcsec']:+.2f} arcsec")
            print(f"Radial distance from optical center: {offsets['radial_offset_arcsec']:.2f} arcsec ({offsets['radial_offset_arcmin']:.2f} arcmin)")
            print(f"Azimuth from north: {offsets['azimuth_deg']:.2f}°")
            sys.stdout.flush()

            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("button_press_event", on_click)
        plt.tight_layout()
        plt.show()
        return fig, ax