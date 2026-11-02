
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

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
    figsize=(16, 7),
):
    """
    Compare detector contours before and after optical-center alignment.

    Left: Original, unshifted coordinates.
    Right: Coordinates shifted to align optical centers.
    """

    optical_shifts = CoAlignFOXSI.get_optical_center_shifts(
        optical_centers=optical_centers,
        reference_detector=reference_detector,
    )

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, align, title in zip(
        axes,
        [False, True],
        ["Unaligned", f"Aligned to {reference_detector}"],
    ):

        for detector in DETECTORS:

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

        if xlim is not None:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

    # Shared legend for the detectors.
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

    fig.suptitle(target, y=1.08, fontsize=14)
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
):
    """
    Make a SunPy map from the SPARCS-shifted combined CdTe data.

    The resulting map uses the coordinate frame of the supplied AIA map.
    Its pixel coordinates represent Solar-X and Solar-Y in arcseconds.

    Parameters
    ----------
    cdte_target_dict : dict
        Dictionary containing the combined CdTe data products.

    target : str
        Target name.

    aia_map : sunpy.map.GenericMap
        AIA map providing the coordinate frame.

    bins : int
        Number of histogram bins along each axis.

    plot_range : tuple
        Solar-X and Solar-Y histogram limits in arcseconds.

    sigma : float or None
        Gaussian smoothing sigma in histogram pixels.
        Set to None to disable smoothing.

    qualityflag : bool
        If True, retain only events with flag_quality == 1.

    energy_range : tuple or None
        Optional energy range passed to filter_cdte_events.

    specific_cdte : str, list, tuple, or None
        Optional detector selection. None uses all detectors.

    Returns
    -------
    sunpy.map.GenericMap
        SPARCS-shifted combined CdTe map.
    """

    # Retrieve the SPARCS-shifted combined data.
    data = (
        cdte_target_dict["Combined Data"][target]
        ["SPARCS shifted data"]["Data"]
    )

    # Optionally select individual detectors.
    data = _select_cdte_detectors(
        data,
        specific_cdte=specific_cdte,
    )

    # Apply the existing quality and energy filters.
    x, y = filter_cdte_events(
        data,
        qualityflag=qualityflag,
        energy_range=energy_range,
    )

    if len(x) == 0:
        raise ValueError(
            f"No valid CdTe events remain for {target}."
        )

    # Make the 2D histogram.
    image, xedges, yedges = np.histogram2d(
        x,
        y,
        bins=bins,
        range=[plot_range, plot_range],
    )

    # Transpose to match the usual image / SunPy map orientation.
    image = image.T

    # Optional smoothing.
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

    # The histogram coordinate system is Solar-X / Solar-Y,
    # with (0, 0) at the center of the histogram.
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
):
    """
    Plot SPARCS-shifted combined CdTe contours over an AIA image.

    The AIA frame is selected using the manually specified UTC middle
    time for the requested target, unless an AIA map is supplied.

    Parameters
    ----------
    cdte_target_dict : dict
        Dictionary containing the combined CdTe data products.

    target : str
        Target name, e.g. "Target Alpha".

    aia_wavelength : int
        AIA wavelength, e.g. 94, 171, or 211.

    aia_map : sunpy.map.GenericMap or None
        Optional AIA map. If None, the closest frame is selected
        using TARGET_TIME_DICT.

    aia_dir : str
        Directory containing AIA FITS files.

    bins : int
        Number of histogram bins along each axis.

    plot_range : tuple
        Solar-X and Solar-Y histogram limits in arcseconds.

    sigma : float or None
        Gaussian smoothing sigma in histogram pixels.
        Set to None to disable smoothing.

    qualityflag : bool
        If True, retain only events with flag_quality == 1.

    energy_range : tuple or None
        Optional energy range passed to filter_cdte_events.

    specific_cdte : str, list, tuple, or None
        Optional detector selection. None uses all detectors.

    levels : tuple
        Contour levels expressed as percentages of the maximum
        smoothed CdTe histogram value.

    xlim, ylim : tuple or None
        Optional Solar-X and Solar-Y plot limits in arcseconds.

    figsize : tuple
        Figure size.

    Returns
    -------
    fig, ax, cdte_map, aia_map
    """

    # ---------------------------------------------------------
    # Select the AIA map
    # ---------------------------------------------------------
    if aia_map is None:
        aia_map = get_closest_aia_map(
            target=target,
            aia_wavelength=aia_wavelength,
            aia_dir=aia_dir,
        )

    # ---------------------------------------------------------
    # Make the SPARCS-shifted CdTe map
    # ---------------------------------------------------------
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
    )

    # ---------------------------------------------------------
    # Set up the plot
    # ---------------------------------------------------------
    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(
        projection=aia_map,
    )

    # AIA colormap.
    cmap = getattr(
        cm,
        f"sdoaia{aia_wavelength}",
    )

    # Logarithmic image normalization.
    norm = ImageNormalize(
        aia_map.data,
        stretch=LogStretch(),
    )

    aia_image = aia_map.plot(
        axes=ax,
        cmap=cmap,
        norm=norm,
    )

    # ---------------------------------------------------------
    # Overlay CdTe contours
    # ---------------------------------------------------------
    cdte_map.draw_contours(
        axes=ax,
        levels=np.asarray(levels) * u.percent,
        colors="cyan",
        linewidths=1.5,
    )

    # ---------------------------------------------------------
    # Optional plot limits
    # ---------------------------------------------------------
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

        ax.set_xlim(
            bl_pix.x.value,
            tr_pix.x.value,
        )

        ax.set_ylim(
            bl_pix.y.value,
            tr_pix.y.value,
        )

    # ---------------------------------------------------------
    # Labels and title
    # ---------------------------------------------------------
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

    plt.tight_layout()

    return fig, ax, cdte_map, aia_map