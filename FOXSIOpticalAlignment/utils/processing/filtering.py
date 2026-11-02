import numpy as np
import cdte_tools_py.calibration.filter_convert_extract as fce
from astropy.table import vstack
from utils.alignment.opticalcenter import CoAlignFOXSI, SPARCS_DICT
import astropy.units as u

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

def get_cdte_target_coordinates(
    cdte_target_dict,
    target,
    detector,
    qualityflag=True,
    energy_range=None,
):
    """
    Retrieve filtered Solar-X and Solar-Y coordinates for a target
    and detector.

    Parameters
    ----------
    cdte_target_dict : dict
        Dictionary containing target and detector event data.
    target : str
        Target name.
    detector : str
        Detector name (e.g. 'CdTe1', 'CdTe3', 'CdTe4', 'CdTe5').
    qualityflag : bool
        Whether to apply the quality flag filter.
    energy_range : tuple or None
        Optional energy range passed to filter_cdte_events().

    Returns
    -------
    x, y : numpy.ndarray
        Filtered Solar-X and Solar-Y event coordinates.
    """
    data = cdte_target_dict[detector][target]["Data"]

    x, y = filter_cdte_events(
        data,
        qualityflag=qualityflag,
        energy_range=energy_range,
    )

    return x, y

def build_combined_cdte_data(
    cdte_target_dict,
    target,
    detectors=("CdTe1", "CdTe3", "CdTe4", "CdTe5"),
    optical_centers=None,
    reference_detector="CdTe4",
    overwrite=True,
):
    """
    Combines CdTe event data after shifting each detector's Solar-X and
    Solar-Y coordinates to align its optical center with the reference
    detector's optical center.

    Adds a 'detector' column identifying the source detector for every event.

    The combined data are stored in:
        cdte_target_dict["Combined Data"][target]["Data"]

    Original detector data are not modified.
    """

    if optical_centers is None:
        optical_centers = CoAlignFOXSI.OPTICAL_CENTERS

    # Calculate the shifts needed to align all optical centers
    shifts = CoAlignFOXSI.get_optical_center_shifts(
        optical_centers=optical_centers,
        reference_detector=reference_detector,
    )

    combined_tables = []

    for detector in detectors:

        # Copy the original event table
        data = cdte_target_dict[detector][target]["Data"].copy()

        # Identify the detector associated with each event
        data["detector"] = np.full(len(data), detector, dtype="U5")

        # Shift the event coordinates to align optical centers
        data["solar_x_unaligned"] = (
            data["solar_x_unaligned"] + shifts[detector]["dx"]
        )

        data["solar_y_unaligned"] = (
            data["solar_y_unaligned"] + shifts[detector]["dy"]
        )

        combined_tables.append(data)

    # Stack all detector event tables
    combined_data = vstack(
        combined_tables,
        join_type="outer",
        metadata_conflicts="silent",
    )

    # Create the combined-data entry if it does not already exist
    if "Combined Data" not in cdte_target_dict:
        cdte_target_dict["Combined Data"] = {}

    if target in cdte_target_dict["Combined Data"] and not overwrite:
        raise ValueError(
            f"Combined data already exist for {target}. "
            "Set overwrite=True to replace them."
        )

    # Store the combined data
    cdte_target_dict["Combined Data"][target] = {
        "Data": combined_data
    }

    return combined_data

def build_sparcs_shifted_cdte_data(
    cdte_target_dict,
    target,
    sparcs_dict=SPARCS_DICT,
    optical_centers=None,
    reference_detector="CdTe4",
    overwrite=True,
):
    """
    Shifts the optical-center-aligned combined CdTe data so that the
    reference detector's optical center is placed at the SPARCS coordinate.

    Expects the optical-center-aligned data at:
        cdte_target_dict["Combined Data"][target]["Data"]

    Stores the shifted data at:
        cdte_target_dict["Combined Data"][target]["SPARCS shifted data"]["Data"]

    Original detector data and optical-center-aligned combined data
    are not modified.
    """

    if optical_centers is None:
        optical_centers = CoAlignFOXSI.OPTICAL_CENTERS

    # Make sure the target has a SPARCS pointing
    if target not in sparcs_dict:
        raise KeyError(f"No SPARCS pointing available for {target}")

    # Retrieve the existing optical-center-aligned combined data
    combined_data = (
        cdte_target_dict["Combined Data"][target]["Data"]
    )

    # Get the reference detector's optical center
    reference_center = optical_centers[reference_detector]

    optical_x = reference_center["x"]
    optical_y = reference_center["y"]

    # Convert SPARCS coordinates to arcseconds
    sparcs_x = sparcs_dict[target]["West"].to_value(u.arcsec)
    sparcs_y = sparcs_dict[target]["North"].to_value(u.arcsec)

    # Calculate the translation needed to move the optical center
    # to the SPARCS coordinate
    dx = sparcs_x - optical_x
    dy = sparcs_y - optical_y

    # Copy the combined data so the original remains unchanged
    shifted_data = combined_data.copy()

    # Apply the same translation to every detector's events
    shifted_data["solar_x_unaligned"] = (
        shifted_data["solar_x_unaligned"] + dx
    )

    shifted_data["solar_y_unaligned"] = (
        shifted_data["solar_y_unaligned"] + dy
    )

    # Check whether a SPARCS-shifted product already exists
    if (
        "SPARCS shifted data" in cdte_target_dict["Combined Data"][target]
        and not overwrite
    ):
        raise ValueError(
            f"SPARCS-shifted data already exist for {target}. "
            "Set overwrite=True to replace them."
        )

    # Store the shifted data separately
    cdte_target_dict["Combined Data"][target]["SPARCS shifted data"] = {
        "Data": shifted_data,
        "dx": dx,
        "dy": dy,
        "reference_detector": reference_detector,
    }

    return shifted_data