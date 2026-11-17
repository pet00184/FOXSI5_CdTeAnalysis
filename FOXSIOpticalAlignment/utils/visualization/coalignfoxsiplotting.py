
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from pathlib import Path
import subprocess

from utils.processing.filtering import get_cdte_target_coordinates
from scipy.ndimage import gaussian_filter
from utils.processing.filtering import filter_cdte_events

from glob import glob
from sunpy.map import Map
from sunpy.visualization.colormaps import cm
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.visualization import ImageNormalize, LogStretch
from astropy.wcs.utils import pixel_to_skycoord
from sunpy.map.header_helper import make_fitswcs_header

from utils.alignment.opticalcenter import (
    CoAlignFOXSI,
    SPARCS_DICT,
    TARGET_TIME_DICT,
)


OPTICAL_CENTERS = CoAlignFOXSI.OPTICAL_CENTERS

DETECTORS = ["CdTe1", "CdTe3", "CdTe4", "CdTe5"]

DETECTOR_COLORS = {
    "CdTe1": "tab:blue",
    "CdTe3": "tab:orange",
    "CdTe4": "tab:green",
    "CdTe5": "tab:red",
}


def _make_contour_data(
    x,
    y,
    bins=150,
    plot_range=(-1000.0, 1000.0),
):
    """Generate histogram and bin-center coordinates for contours."""

    hist, xedges, yedges = np.histogram2d(
        x,
        y,
        bins=bins,
        range=[plot_range, plot_range],
    )

    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2

    return hist.T, xcenters, ycenters


def _get_contour_levels(
    hist,
    contour_fractions=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
):
    """Calculate contour levels relative to the histogram maximum."""

    maximum = np.max(hist)

    if maximum == 0:
        return []

    levels = maximum * np.asarray(contour_fractions)

    return np.unique(levels[levels > 0])


def _get_detector_coordinates(
    cdte_target_dict,
    target,
    detector,
    qualityflag=True,
    energy_range=None,
    align=False,
    shifts=None,
):
    """Retrieve coordinates and optionally apply optical-center shifts."""

    x, y = get_cdte_target_coordinates(
        cdte_target_dict=cdte_target_dict,
        target=target,
        detector=detector,
        qualityflag=qualityflag,
        energy_range=energy_range,
    )

    if align:
        dx = shifts[detector]["dx"]
        dy = shifts[detector]["dy"]

        x = x + dx
        y = y + dy

    return x, y


def plot_optical_center_aligned(
    cdte_target_dict,
    target,
    bins=150,
    plot_range=(-1000.0, 1000.0),
    qualityflag=True,
    energy_range=None,
    optical_centers=OPTICAL_CENTERS,
    reference_detector="CdTe4",
    contour_fractions=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    xlim=None,
    ylim=None,
    figsize=(12, 10),
):
    """
    Plot all four detectors in a 2x2 grid after aligning their
    optical centers to the reference detector.
    """

    optical_shifts = CoAlignFOXSI.get_optical_center_shifts(
        optical_centers=optical_centers,
        reference_detector=reference_detector,
    )

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for ax, detector in zip(axes, DETECTORS):

        x, y = _get_detector_coordinates(
            cdte_target_dict=cdte_target_dict,
            target=target,
            detector=detector,
            qualityflag=qualityflag,
            energy_range=energy_range,
            align=True,
            shifts=optical_shifts,
        )

        hist, xcenters, ycenters = _make_contour_data(
            x,
            y,
            bins=bins,
            plot_range=plot_range,
        )

        levels = _get_contour_levels(
            hist,
            contour_fractions=contour_fractions,
        )

        if len(levels) > 0:
            ax.contour(
                xcenters,
                ycenters,
                hist,
                levels=levels,
            )

        # Mark the shared optical center (the reference detector center).
        ax.plot(
            optical_centers[reference_detector]["x"],
            optical_centers[reference_detector]["y"],
            marker="*",
            color="black",
            markersize=12,
            label="Reference optical center",
        )

        ax.set_title(detector)
        ax.set_xlabel("Solar-X [arcsec]")
        ax.set_ylabel("Solar-Y [arcsec]")
        ax.grid(alpha=0.2)
        ax.set_aspect("equal", adjustable="box") #will add this into all plotting so that the aspect ratio stays consistent!!

        if xlim is not None:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

    fig.suptitle(
        f"{target}: Optical Centers Aligned to {reference_detector}",
        fontsize=14,
    )

    fig.tight_layout()

    return fig, axes


def plot_optical_center_contour_comparison(
    cdte_target_dict,
    target,
    bins=150,
    plot_range=(-1000.0, 1000.0),
    qualityflag=True,
    energy_range=None,
    optical_centers=OPTICAL_CENTERS,
    reference_detector="CdTe4",
    contour_fractions=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    xlim=None,
    ylim=None,
    figsize=(16, 12),
):
    """
    Compare detector contours before and after optical-center alignment.

    Top left: All detectors in original, unshifted coordinates.
    Remaining plots: Individual detectors shifted to align with
    the reference detector's optical center.
    """

    optical_shifts = CoAlignFOXSI.get_optical_center_shifts(
        optical_centers=optical_centers,
        reference_detector=reference_detector,
    )

    # Create four subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    axes = axes.flatten()

    # Define which detectors appear in each subplot
    plot_configs = [
        (False, DETECTORS, "Unaligned — All Detectors"),
        (True, ["CdTe1", "CdTe4"], f"CdTe1 Aligned to {reference_detector}"),
        (True, ["CdTe3", "CdTe4"], f"CdTe3 Aligned to {reference_detector}"),
        (True, ["CdTe5", "CdTe4"], f"CdTe5 Aligned to {reference_detector}"),
    ]

    for ax, (align, detectors, title) in zip(axes, plot_configs):

        for detector in detectors:

            x, y = _get_detector_coordinates(
                cdte_target_dict=cdte_target_dict,
                target=target,
                detector=detector,
                qualityflag=qualityflag,
                energy_range=energy_range,
                align=align,
                shifts=optical_shifts,
            )

            hist, xcenters, ycenters = _make_contour_data(
                x,
                y,
                bins=bins,
                plot_range=plot_range,
            )

            levels = _get_contour_levels(
                hist,
                contour_fractions=contour_fractions,
            )

            if len(levels) > 0:
                ax.contour(
                    xcenters,
                    ycenters,
                    hist,
                    levels=levels,
                    colors=DETECTOR_COLORS[detector],
                    linewidths=1.5,
                )

        ax.set_title(title)
        ax.set_xlabel("Solar-X [arcsec]")
        ax.set_ylabel("Solar-Y [arcsec]")
        ax.grid(alpha=0.2)
        ax.set_aspect("equal", adjustable="box")

        if xlim is not None:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

    # Shared legend for the detectors
    handles = [
        plt.Line2D(
            [0],
            [0],
            color=DETECTOR_COLORS[detector],
            label=detector,
        )
        for detector in DETECTORS
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.suptitle(target, y=1.06, fontsize=14)
    fig.tight_layout()

    return fig, axes

def _select_cdte_detectors(data, specific_cdte=None):
    """
    Selects events from one or more CdTe detectors.

    specific_cdte:
        None        -> all detectors
        "CdTe4"     -> only CdTe4
        ("CdTe4", "CdTe3") -> CdTe4 and CdTe3
        ["CdTe1", "CdTe5"] -> CdTe1 and CdTe5
    """

    if specific_cdte is None:
        return data

    # Convert a single detector string into a list
    if isinstance(specific_cdte, str):
        specific_cdte = [specific_cdte]

    # Validate detector names
    invalid = [
        detector for detector in specific_cdte
        if detector not in DETECTORS
    ]

    if invalid:
        raise ValueError(
            f"Invalid detector(s): {invalid}. "
            f"Choose from {DETECTORS} or None."
        )

    return data[np.isin(data["detector"], specific_cdte)]

def plot_combined_optical_center_data(
    cdte_target_dict,
    target,
    bins=150,
    plot_range=(-1000.0, 1000.0),
    qualityflag=True,
    energy_range=None,
    optical_centers=OPTICAL_CENTERS,
    reference_detector="CdTe4",
    contour_fractions=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    gaussian_sigma=None,
    specific_cdte=None,
    xlim=None,
    ylim=None,
    figsize=(9, 8),
):
    """
    Plots the combined CdTe event data after optical-center alignment.

    Expects combined data at:
        cdte_target_dict["Combined Data"][target]["Data"]

    Parameters
    ----------
    specific_cdte : str, list, tuple, or None
        None                 -> all detectors
        "CdTe4"              -> only CdTe4
        ("CdTe4", "CdTe3")   -> CdTe4 and CdTe3
        ["CdTe1", "CdTe5"]   -> CdTe1 and CdTe5

    gaussian_sigma : float or None
        Optional Gaussian smoothing applied to the 2D histogram.
        Sigma is specified in histogram bins.
    """

    # Retrieve the optical-center-aligned combined event table
    data = cdte_target_dict["Combined Data"][target]["Data"]

    # Optionally select one or more detectors
    data = _select_cdte_detectors(
        data,
        specific_cdte=specific_cdte,
    )

    # Apply quality and energy filters
    x, y = filter_cdte_events(
        data,
        qualityflag=qualityflag,
        energy_range=energy_range,
    )

    # Generate the combined histogram
    hist, xcenters, ycenters = _make_contour_data(
        x,
        y,
        bins=bins,
        plot_range=plot_range,
    )

    # Optional Gaussian smoothing
    if gaussian_sigma is not None:
        hist = gaussian_filter(hist, sigma=gaussian_sigma)

    # Calculate contour levels
    levels = _get_contour_levels(
        hist,
        contour_fractions=contour_fractions,
    )

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    if len(levels) > 0:
        ax.contour(
            xcenters,
            ycenters,
            hist,
            levels=levels,
            cmap="viridis",
        )

    # Mark the reference optical center
    reference_center = optical_centers[reference_detector]

    ax.plot(
        reference_center["x"],
        reference_center["y"],
        marker="*",
        color="red",
        markersize=14,
        label=f"{reference_detector} optical center",
    )

    # Set detector label for plot title
    if specific_cdte is None:
        detector_label = "All CdTe Detectors"
    elif isinstance(specific_cdte, str):
        detector_label = specific_cdte
    else:
        detector_label = ", ".join(specific_cdte)

    ax.set_title(
        f"{target}: {detector_label} Optical-Center-Aligned Data"
    )
    ax.set_xlabel("Solar-X [arcsec]")
    ax.set_ylabel("Solar-Y [arcsec]")
    ax.grid(alpha=0.2)
    ax.legend()
    ax.set_aspect("equal", adjustable="box")

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    fig.tight_layout()

    return fig, ax

def plot_sparcs_shifted_data(
    cdte_target_dict,
    target,
    bins=150,
    plot_range=(-1000.0, 1000.0),
    qualityflag=True,
    energy_range=None,
    sparcs_dict=SPARCS_DICT,
    contour_fractions=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    gaussian_sigma=None,
    specific_cdte=None,
    xlim=None,
    ylim=None,
    figsize=(9, 8),
):
    """
    Plots the combined CdTe data after optical-center alignment and
    translation to the SPARCS pointing coordinate.

    Expects data at:
        cdte_target_dict["Combined Data"][target]
                       ["SPARCS shifted data"]["Data"]

    Parameters
    ----------
    specific_cdte : str, list, tuple, or None
        None                 -> all detectors
        "CdTe4"              -> only CdTe4
        ("CdTe4", "CdTe3")   -> CdTe4 and CdTe3
        ["CdTe1", "CdTe5"]   -> CdTe1 and CdTe5

    gaussian_sigma : float or None
        Optional Gaussian smoothing applied to the 2D histogram.
        Sigma is specified in histogram bins.
    """

    # Retrieve the SPARCS-shifted combined event table
    data = (
        cdte_target_dict["Combined Data"][target]
        ["SPARCS shifted data"]["Data"]
    )

    # Optionally select one or more detectors
    data = _select_cdte_detectors(
        data,
        specific_cdte=specific_cdte,
    )

    # Apply quality and energy filters
    x, y = filter_cdte_events(
        data,
        qualityflag=qualityflag,
        energy_range=energy_range,
    )

    # Generate the combined histogram
    hist, xcenters, ycenters = _make_contour_data(
        x,
        y,
        bins=bins,
        plot_range=plot_range,
    )

    # Optional Gaussian smoothing
    if gaussian_sigma is not None:
        hist = gaussian_filter(hist, sigma=gaussian_sigma)

    # Calculate contour levels
    levels = _get_contour_levels(
        hist,
        contour_fractions=contour_fractions,
    )

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    if len(levels) > 0:
        ax.contour(
            xcenters,
            ycenters,
            hist,
            levels=levels,
            cmap="viridis",
        )

    # Mark SPARCS coordinate
    sparcs_x = sparcs_dict[target]["West"].to_value(u.arcsec)
    sparcs_y = sparcs_dict[target]["North"].to_value(u.arcsec)

    ax.plot(
        sparcs_x,
        sparcs_y,
        marker="*",
        color="red",
        markersize=14,
        label="SPARCS pointing",
    )

    # Set detector label for plot title
    if specific_cdte is None:
        detector_label = "All CdTe Detectors"
    elif isinstance(specific_cdte, str):
        detector_label = specific_cdte
    else:
        detector_label = ", ".join(specific_cdte)

    ax.set_title(
        f"{target}: {detector_label} SPARCS-Shifted Data"
    )
    ax.set_xlabel("Solar-X [arcsec]")
    ax.set_ylabel("Solar-Y [arcsec]")
    ax.grid(alpha=0.2)
    ax.legend()
    ax.set_aspect("equal", adjustable="box")

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    fig.tight_layout()

    return fig, ax

### AIA map stuff
def get_closest_aia_map(
    target,
    aia_wavelength,
    aia_dir="/Users/pet00184/FOXSI_Analysis/FOXSI5_CdTeAnalysis/Data/AIA",
    target_time_dict=TARGET_TIME_DICT,
):
    """
    Load the AIA frame closest to the manually specified middle
    time of a CdTe target observation.

    Parameters
    ----------
    target : str
        Target name, e.g. "Target Alpha".

    aia_wavelength : int
        AIA wavelength, e.g. 94, 171, or 211.

    aia_dir : str
        Directory containing the AIA FITS files.

    target_time_dict : dict
        Manually specified target observation times.

    Returns
    -------
    sunpy.map.GenericMap
        AIA map closest in time to the target's specified UTC middle.
    """

    if target not in target_time_dict:
        raise ValueError(f"Unknown target: {target}")

    target_time = target_time_dict[target]["UTC middle"]

    files = sorted(
        glob(f"{aia_dir}/*{aia_wavelength}*.fits")
    )

    if not files:
        raise FileNotFoundError(
            f"No AIA {aia_wavelength} Å FITS files found in {aia_dir}"
        )

    aia_sequence = Map(files, sequence=True)

    aia_times = np.array([
        Time(aia_map.date).unix
        for aia_map in aia_sequence
    ])

    closest_index = np.argmin(
        np.abs(aia_times - target_time.unix)
    )

    closest_map = aia_sequence[closest_index]

    print(
        f"{target}, AIA {aia_wavelength} Å:\n"
        f"Target middle: {target_time.isot}\n"
        f"AIA frame:    {Time(closest_map.date).isot}\n"
        f"Time offset:  "
        f"{(aia_times[closest_index] - target_time.unix):.1f} seconds"
    )

    return closest_map

def make_sparcs_shifted_cdte_map(
    cdte_target_dict,
    target,
    aia_map,
    bins=300,
    plot_range=(-1500, 1500),
    sigma=2,
    qualityflag=True,
    energy_range=None,
    specific_cdte=None,
    data_product="SPARCS shifted data",
):
    """
    Make a SunPy map from a selected shifted combined CdTe product.

    data_product can be:
        "SPARCS shifted data"
        "Manual Pointing"
    """

    data = (
        cdte_target_dict["Combined Data"][target]
        [data_product]["Data"]
    )

    data = _select_cdte_detectors(
        data,
        specific_cdte=specific_cdte,
    )

    x, y = filter_cdte_events(
        data,
        qualityflag=qualityflag,
        energy_range=energy_range,
    )

    if len(x) == 0:
        raise ValueError(
            f"No valid CdTe events remain for {target}."
        )

    image, xedges, yedges = np.histogram2d(
        x,
        y,
        bins=bins,
        range=[plot_range, plot_range],
    )

    image = image.T

    if sigma is not None:
        image = gaussian_filter(image, sigma=sigma)

    pixel_scale = u.Quantity(
        [
            xedges[1] - xedges[0],
            yedges[1] - yedges[0],
        ],
        u.arcsec,
    ) / u.pixel

    reference_coord = SkyCoord(
        0 * u.arcsec,
        0 * u.arcsec,
        frame=aia_map.coordinate_frame,
    )

    header = make_fitswcs_header(
        image,
        reference_coord,
        scale=pixel_scale,
    )

    cdte_map = Map(image, header)

    return cdte_map

def plot_sparcs_cdte_on_aia(
    cdte_target_dict,
    target,
    aia_wavelength,
    aia_map=None,
    aia_dir="/Users/pet00184/FOXSI_Analysis/FOXSI5_CdTeAnalysis/Data/AIA",
    bins=300,
    plot_range=(-1500, 1500),
    sigma=2,
    qualityflag=True,
    energy_range=None,
    specific_cdte=None,
    levels=(20, 30, 40, 50, 60, 70, 80, 90, 95),
    xlim=None,
    ylim=None,
    figsize=(9, 9),
    manual_com_position=None,
    centroid_sigma=2.0,
    threshold_fraction=0.5,
    centroid_x_range=None,
    centroid_y_range=None,
):
    """
    Plot manually adjusted combined CdTe contours over an AIA image.

    The original SPARCS pointing is shown as a red star.
    The final CdTe center of mass is shown as a green star.

    manual_com_position:
        Desired final (Solar-X, Solar-Y) COM position in arcseconds.
        If None, the measured COM is retained without a manual shift.
    """

    from utils.processing.filtering import build_manual_pointing_cdte_data

    # Select the AIA map.
    if aia_map is None:
        aia_map = get_closest_aia_map(
            target=target,
            aia_wavelength=aia_wavelength,
            aia_dir=aia_dir,
        )

    # Calculate the CdTe COM and build the manual-pointing product.
    build_manual_pointing_cdte_data(
        cdte_target_dict=cdte_target_dict,
        target=target,
        manual_com_position=manual_com_position,
        bins=bins,
        plot_range=plot_range,
        qualityflag=qualityflag,
        energy_range=energy_range,
        centroid_sigma=centroid_sigma,
        threshold_fraction=threshold_fraction,
        centroid_x_range=centroid_x_range,
        centroid_y_range=centroid_y_range,
    )

    manual_product = (
        cdte_target_dict["Combined Data"][target]["Manual Pointing"]
    )

    # Make the map using the manually adjusted data.
    cdte_map = make_sparcs_shifted_cdte_map(
        cdte_target_dict=cdte_target_dict,
        target=target,
        aia_map=aia_map,
        bins=bins,
        plot_range=plot_range,
        sigma=sigma,
        qualityflag=qualityflag,
        energy_range=energy_range,
        specific_cdte=specific_cdte,
        data_product="Manual Pointing",
    )

    # Set up the plot.
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection=aia_map)

    cmap = getattr(cm, f"sdoaia{aia_wavelength}")

    norm = ImageNormalize(
        aia_map.data,
        stretch=LogStretch(),
    )

    aia_map.plot(
        axes=ax,
        cmap=cmap,
        norm=norm,
    )

    # Overlay CdTe contours.
    cdte_map.draw_contours(
        axes=ax,
        levels=np.asarray(levels) * u.percent,
        colors="cyan",
        linewidths=1.5,
    )

    # Original SPARCS optical center.
    sparcs_x = SPARCS_DICT[target]["West"]
    sparcs_y = SPARCS_DICT[target]["North"]

    sparcs_coord = SkyCoord(
        sparcs_x,
        sparcs_y,
        frame=aia_map.coordinate_frame,
    )

    ax.plot_coord(
        sparcs_coord,
        marker="*",
        color="red",
        markersize=10,
        linestyle="none",
        label="SPARCS optical center",
        zorder=15,
    )

    # Final CdTe center of mass.
    com_coord = SkyCoord(
        manual_product["final_com_x"] * u.arcsec,
        manual_product["final_com_y"] * u.arcsec,
        frame=aia_map.coordinate_frame,
    )

    ax.plot_coord(
        com_coord,
        marker="*",
        color="lime",
        markersize=10,
        markeredgecolor="black",
        markeredgewidth=0.5,
        linestyle="none",
        label="CdTe center of mass",
        zorder=16,
    )

    # Optional plot limits.
    if xlim is not None and ylim is not None:

        bottom_left = SkyCoord(
            xlim[0] * u.arcsec,
            ylim[0] * u.arcsec,
            frame=aia_map.coordinate_frame,
        )

        top_right = SkyCoord(
            xlim[1] * u.arcsec,
            ylim[1] * u.arcsec,
            frame=aia_map.coordinate_frame,
        )

        bl_pix = aia_map.world_to_pixel(bottom_left)
        tr_pix = aia_map.world_to_pixel(top_right)

        ax.set_xlim(bl_pix.x.value, tr_pix.x.value)
        ax.set_ylim(bl_pix.y.value, tr_pix.y.value)

    target_time = TARGET_TIME_DICT[target]["UTC middle"]

    detector_label = (
        "All CdTe detectors"
        if specific_cdte is None
        else (
            specific_cdte
            if isinstance(specific_cdte, str)
            else ", ".join(specific_cdte)
        )
    )

    ax.set_title(
        f"{target} — AIA {aia_wavelength} Å\n"
        f"{target_time.isot} UTC | {detector_label}"
    )

    ax.set_xlabel("Solar-X [arcsec]")
    ax.set_ylabel("Solar-Y [arcsec]")
    ax.grid(True)
    ax.legend(loc="upper right")

    plt.tight_layout()

    return fig, ax, cdte_map, aia_map

#### TRYING TO MAKE MOVIES STUFF HERE

def _select_cdte_time_range(data, timerange, target_start):
    """
    Select CdTe events using elapsed time relative to the first event.

    The target dictionary provides the authoritative start time.
    Event UTC timestamps are used only to determine relative timing.
    """
    if len(data) == 0:
        return data

    event_times = Time(data["utc"])

    # Shift event timestamps so the first event corresponds to
    # the authoritative target start time.
    relative_times = event_times - event_times.min()
    corrected_times = target_start + relative_times

    start, end = timerange
    mask = (corrected_times >= start) & (corrected_times < end)

    return data[mask]

def make_manual_pointing_cdte_map(
    cdte_target_dict,
    target,
    aia_map,
    timerange=None,
    bins=300,
    plot_range=(-1500, 1500),
    sigma=2,
    qualityflag=True,
    energy_range=None,
    specific_cdte=None,
):
    """
    Make a SunPy map from the manually pointed combined CdTe data.

    Optionally restricts the events to a specified UTC time interval.
    """

    # Retrieve the manually pointed combined data.
    data = (
        cdte_target_dict["Combined Data"][target]
        ["Manual Pointing"]["Data"]
    )

    # Select the time interval for this frame.
    if timerange is not None:
        data = _select_cdte_time_range(
            data,
            timerange,
            target_start=TARGET_TIME_DICT[target]["UTC start"],
        )

    # Optionally select individual detectors.
    data = _select_cdte_detectors(
        data,
        specific_cdte=specific_cdte,
    )

    # Apply quality and energy filters.
    x, y = filter_cdte_events(
        data,
        qualityflag=qualityflag,
        energy_range=energy_range,
    )

    if len(x) == 0:
        raise ValueError(
            f"No valid CdTe events remain for {target} "
            f"during {timerange}."
        )

    # Make the 2D histogram.
    image, xedges, yedges = np.histogram2d(
        x,
        y,
        bins=bins,
        range=[plot_range, plot_range],
    )

    image = image.T

    # Optional Gaussian smoothing.
    if sigma is not None:
        image = gaussian_filter(image, sigma=sigma)

    # Pixel scale in arcseconds per pixel.
    pixel_scale = u.Quantity(
        [
            xedges[1] - xedges[0],
            yedges[1] - yedges[0],
        ],
        u.arcsec,
    ) / u.pixel

    # Use the AIA coordinate frame.
    reference_coord = SkyCoord(
        0 * u.arcsec,
        0 * u.arcsec,
        frame=aia_map.coordinate_frame,
    )

    header = make_fitswcs_header(
        image,
        reference_coord,
        scale=pixel_scale,
    )

    return Map(image, header)

def plot_manual_pointing_cdte_on_aia(
    cdte_target_dict,
    target,
    aia_wavelength,
    aia_map,
    timerange,
    bins=300,
    plot_range=(-1500, 1500),
    sigma=2,
    qualityflag=True,
    energy_range=None,
    specific_cdte=None,
    levels=(20, 30, 40, 50, 60, 70, 80, 90, 95),
    xlim=None,
    ylim=None,
    figsize=(9, 9),
):
    """Plot one time-resolved, manually pointed CdTe frame over AIA."""

    # Make the time-resolved CdTe map.
    cdte_map = make_manual_pointing_cdte_map(
        cdte_target_dict=cdte_target_dict,
        target=target,
        aia_map=aia_map,
        timerange=timerange,
        bins=bins,
        plot_range=plot_range,
        sigma=sigma,
        qualityflag=qualityflag,
        energy_range=energy_range,
        specific_cdte=specific_cdte,
    )

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection=aia_map)

    # Plot AIA background.
    cmap = getattr(cm, f"sdoaia{aia_wavelength}")

    norm = ImageNormalize(
        aia_map.data,
        stretch=LogStretch(),
    )

    aia_map.plot(
        axes=ax,
        cmap=cmap,
        norm=norm,
    )

    # Overlay CdTe contours.
    cdte_map.draw_contours(
        axes=ax,
        levels=np.asarray(levels) * u.percent,
        colors="cyan",
        linewidths=1.5,
    )

    # Original SPARCS pointing.
    sparcs_x = SPARCS_DICT[target]["West"]
    sparcs_y = SPARCS_DICT[target]["North"]

    sparcs_coord = SkyCoord(
        sparcs_x,
        sparcs_y,
        frame=aia_map.coordinate_frame,
    )

    ax.plot_coord(
        sparcs_coord,
        marker="*",
        color="red",
        markersize=13,
        linestyle="none",
        label="SPARCS pointing",
    )

    # Final COM from the manual-pointing product.
    manual_product = (
        cdte_target_dict["Combined Data"][target]["Manual Pointing"]
    )

    final_com = SkyCoord(
        manual_product["final_com_x"] * u.arcsec,
        manual_product["final_com_y"] * u.arcsec,
        frame=aia_map.coordinate_frame,
    )

    ax.plot_coord(
        final_com,
        marker="*",
        color="lime",
        markersize=13,
        linestyle="none",
        label="Final CdTe COM",
    )

    # Apply plot limits.
    if xlim is not None and ylim is not None:

        bottom_left = SkyCoord(
            xlim[0] * u.arcsec,
            ylim[0] * u.arcsec,
            frame=aia_map.coordinate_frame,
        )

        top_right = SkyCoord(
            xlim[1] * u.arcsec,
            ylim[1] * u.arcsec,
            frame=aia_map.coordinate_frame,
        )

        bl_pix = aia_map.world_to_pixel(bottom_left)
        tr_pix = aia_map.world_to_pixel(top_right)

        ax.set_xlim(bl_pix.x.value, tr_pix.x.value)
        ax.set_ylim(bl_pix.y.value, tr_pix.y.value)

    start, end = (Time(t) for t in timerange)

    ax.set_title(
        f"{target} — AIA {aia_wavelength} Å\n"
        f"{start.isot} – {end.isot} UTC"
    )

    ax.set_xlabel("Solar-X [arcsec]")
    ax.set_ylabel("Solar-Y [arcsec]")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    return fig, ax, cdte_map

def get_closest_aia_frame(aia_sequence, target_time):
    """Return the AIA map closest in time to the requested UTC time."""

    target_time = Time(target_time)

    aia_times = np.array([
        Time(aia_map.date).unix
        for aia_map in aia_sequence
    ])

    closest_index = np.argmin(
        np.abs(aia_times - target_time.unix)
    )

    return aia_sequence[closest_index]

def make_manual_pointing_cdte_movie_frames(
    cdte_target_dict,
    target,
    cadence=5 * u.s,
    duration=5 * u.s,
    bins=300,
    sigma=2,
    energy_range=None,
    qualityflag=True,
    aia_wavelength=94,
    aia_dir="/Users/pet00184/FOXSI_Analysis/FOXSI5_CdTeAnalysis/Data/AIA",
    savepath="./movie_frames/",
    plot_range=(-1500, 1500),
    xlim=None,
    ylim=None,
    specific_cdte=None,
    clear_existing=False,
):
    """
    Generate PNG frames of manually pointed combined CdTe data over AIA.

    Each frame contains events within a specified UTC time interval.
    """

    savepath = Path(savepath)
    savepath.mkdir(parents=True, exist_ok=True)

    if cadence <= 0 * u.s or duration <= 0 * u.s:
        raise ValueError("Cadence and duration must both be positive.")

    # Optionally clear old movie frames.
    if clear_existing:
        for old_frame in savepath.glob("frame_*.png"):
            old_frame.unlink()

    # Get the target observation times.
    start = TARGET_TIME_DICT[target]["UTC start"]
    end = TARGET_TIME_DICT[target]["UTC end"]

    # Load the AIA sequence once.
    aia_files = sorted(
        glob(f"{aia_dir}/*{aia_wavelength}*.fits")
    )

    if not aia_files:
        raise FileNotFoundError(
            f"No AIA {aia_wavelength} Å FITS files found in {aia_dir}"
        )

    aia_sequence = Map(aia_files, sequence=True)

    frame = 0
    t = start

    while t < end:

        frame_end = min(t + duration, end)
        frame_midpoint = t + (frame_end - t) / 2

        print(
            f"Creating frame {frame:04d}: "
            f"{t.isot} – {frame_end.isot}"
        )

        # Find the closest AIA frame to the middle of this interval.
        aia_map = get_closest_aia_frame(
            aia_sequence,
            frame_midpoint,
        )

        # Plot this time interval.
        fig, ax, cdte_map = plot_manual_pointing_cdte_on_aia(
            cdte_target_dict=cdte_target_dict,
            target=target,
            aia_wavelength=aia_wavelength,
            aia_map=aia_map,
            timerange=(t, frame_end),
            bins=bins,
            plot_range=plot_range,
            sigma=sigma,
            qualityflag=qualityflag,
            energy_range=energy_range,
            specific_cdte=specific_cdte,
            xlim=xlim,
            ylim=ylim,
        )

        # Save the frame.
        fig.savefig(
            savepath / f"frame_{frame:04d}.png",
            dpi=200,
        )

        plt.close(fig)

        frame += 1
        t += cadence

    print(f"\nFinished creating {frame} frames in {savepath}")

    return frame

def make_movie(
    frame_dir,
    output_name="foxsi_movie.mp4",
    fps=10,
    pattern="frame_%04d.png",
    quality=18,
):
    """
    Convert PNG movie frames into an MP4 using ffmpeg.

    Lower CRF quality values produce higher-quality, larger files.
    """

    frame_dir = Path(frame_dir)

    if not frame_dir.exists():
        raise FileNotFoundError(
            f"Frame directory does not exist: {frame_dir}"
        )

    if len(list(frame_dir.glob("frame_*.png"))) == 0:
        raise FileNotFoundError(
            f"No PNG frames found in {frame_dir}"
        )

    input_pattern = str(frame_dir / pattern)
    output_path = frame_dir / output_name

    command = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-start_number", "0",
        "-i", input_pattern,
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264",
        "-crf", str(quality),
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]

    print("Creating movie:")
    print(" ".join(command))

    subprocess.run(command, check=True)

    print(f"\nSaved movie:\n{output_path}")

    return output_path