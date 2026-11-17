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


def build_manual_pointing_cdte_data(
    cdte_target_dict,
    target,
    manual_pointing,
    optical_centers=None,
    reference_detector="CdTe4",
    overwrite=True,
):
    """
    Shifts the optical-center-aligned combined CdTe data so that the
    reference detector's optical center is placed at a manually
    specified Solar-X / Solar-Y coordinate.

    Parameters
    ----------
    cdte_target_dict : dict
        Dictionary containing the combined CdTe data products.

    target : str
        Target name.

    manual_pointing : tuple
        (Solar-X, Solar-Y) coordinates in arcseconds.

    optical_centers : dict or None
        Optical center dictionary. Defaults to CoAlignFOXSI.OPTICAL_CENTERS.

    reference_detector : str
        Detector whose optical center defines the shared optical center.

    overwrite : bool
        Whether to overwrite an existing manual pointing product.

    Returns
    -------
    astropy.table.Table
        Manually shifted combined CdTe event data.
    """

    if optical_centers is None:
        optical_centers = CoAlignFOXSI.OPTICAL_CENTERS

    if target not in cdte_target_dict.get("Combined Data", {}):
        raise KeyError(
            f"No combined optical-center-aligned data found for {target}. "
            "Run build_combined_cdte_data() first."
        )

    if reference_detector not in optical_centers:
        raise KeyError(
            f"No optical center found for {reference_detector}."
        )

    if len(manual_pointing) != 2:
        raise ValueError(
            "manual_pointing must be (Solar-X, Solar-Y) in arcseconds."
        )

    manual_x, manual_y = manual_pointing

    # Retrieve the optical-center-aligned combined data
    combined_data = (
        cdte_target_dict["Combined Data"][target]["Data"]
    )

    # Get the reference detector's optical center
    reference_center = optical_centers[reference_detector]

    optical_x = reference_center["x"]
    optical_y = reference_center["y"]

    # Calculate the translation needed to move the optical center
    # to the manually specified Solar-X / Solar-Y coordinate
    dx = manual_x - optical_x
    dy = manual_y - optical_y

    # Copy the combined data so the original remains unchanged
    shifted_data = combined_data.copy()

    shifted_data["solar_x_unaligned"] = (
        shifted_data["solar_x_unaligned"] + dx
    )

    shifted_data["solar_y_unaligned"] = (
        shifted_data["solar_y_unaligned"] + dy
    )

    # Check whether a manual pointing product already exists
    if (
        "Manual Pointing" in cdte_target_dict["Combined Data"][target]
        and not overwrite
    ):
        raise ValueError(
            f"Manual pointing data already exist for {target}. "
            "Set overwrite=True to replace them."
        )

    # Store the manually shifted data separately
    cdte_target_dict["Combined Data"][target]["Manual Pointing"] = {
        "Data": shifted_data,
        "dx": dx,
        "dy": dy,
        "reference_detector": reference_detector,
        "Solar-X": manual_x,
        "Solar-Y": manual_y,
    }

    print(
        f"{target}: Manual pointing applied\n"
        f"  Solar-X: {manual_x:.2f} arcsec\n"
        f"  Solar-Y: {manual_y:.2f} arcsec\n"
        f"  Shift: dx = {dx:.2f}, dy = {dy:.2f} arcsec"
    )

    return shifted_data


def build_manual_pointing_cdte_data(
    cdte_target_dict,
    target,
    manual_com_position=None,
    bins=200,
    plot_range=(-1500.0, 1500.0),
    qualityflag=True,
    energy_range=None,
    centroid_sigma=2.0,
    threshold_fraction=0.5,
    centroid_x_range=None,
    centroid_y_range=None,
    overwrite=True,
):
    """
    Calculate the combined CdTe centroid after SPARCS shifting and
    optionally apply an additional manual translation.

    Stores the result separately at:
        cdte_target_dict["Combined Data"][target]["Manual Pointing"]

    manual_com_position is the desired final (Solar-X, Solar-Y)
    centroid position in arcseconds.
    """

    from utils.alignment.opticalcenter import FOXSIAlignmentImage

    target_data = cdte_target_dict["Combined Data"][target]

    # Retrieve the original SPARCS-shifted data.
    sparcs_product = target_data["SPARCS shifted data"]
    sparcs_data = sparcs_product["Data"]

    # Calculate the centroid using the existing centroid algorithm.
    # CdTe4 is used as the class label because the class requires
    # a valid detector name; the input is the combined dataset.
    centroid_image = FOXSIAlignmentImage(
        cdte_data=sparcs_data,
        cdte="CdTe4",
    )

    centroid_image.prepare_image(
        qualityflag=qualityflag,
        energy_range=energy_range,
        bins=bins,
        plot_range=plot_range,
    )

    if centroid_x_range is None:
        centroid_x_range = plot_range

    if centroid_y_range is None:
        centroid_y_range = plot_range

    centroid_image.find_centroid(
        sigma=centroid_sigma,
        threshold_fraction=threshold_fraction,
        x_range=centroid_x_range,
        y_range=centroid_y_range,
    )

    com_x = float(centroid_image.centroid_x)
    com_y = float(centroid_image.centroid_y)

    com_x_err = float(centroid_image.centroid_x_err)
    com_y_err = float(centroid_image.centroid_y_err)

    # Determine the additional manual translation.
    if manual_com_position is None:
        manual_com_x = None
        manual_com_y = None
        dx_manual = 0.0
        dy_manual = 0.0

    else:
        if len(manual_com_position) != 2:
            raise ValueError(
                "manual_com_position must be (Solar-X, Solar-Y)."
            )

        manual_com_x = float(manual_com_position[0])
        manual_com_y = float(manual_com_position[1])

        dx_manual = manual_com_x - com_x
        dy_manual = manual_com_y - com_y

    # Copy the SPARCS-shifted data so the original is preserved.
    manual_data = sparcs_data.copy()

    manual_data["solar_x_unaligned"] = (
        manual_data["solar_x_unaligned"] + dx_manual
    )

    manual_data["solar_y_unaligned"] = (
        manual_data["solar_y_unaligned"] + dy_manual
    )

    # The SPARCS optical axis was placed at the SPARCS coordinates.
    # Apply the manual translation to get its final position.
    optical_axis_x = (
        sparcs_product["dx"]
        + CoAlignFOXSI.OPTICAL_CENTERS["CdTe4"]["x"]
        + dx_manual
    )

    optical_axis_y = (
        sparcs_product["dy"]
        + CoAlignFOXSI.OPTICAL_CENTERS["CdTe4"]["y"]
        + dy_manual
    )

    # The final COM is the requested position if a manual adjustment
    # was supplied; otherwise it remains at the measured position.
    final_com_x = com_x + dx_manual
    final_com_y = com_y + dy_manual

    if (
        "Manual Pointing" in target_data
        and not overwrite
    ):
        raise ValueError(
            f"Manual Pointing data already exist for {target}. "
            "Set overwrite=True to replace them."
        )

    # Store the new product and all relevant pointing metadata.
    target_data["Manual Pointing"] = {
        "Data": manual_data,

        # Original SPARCS translation.
        "dx": sparcs_product["dx"],
        "dy": sparcs_product["dy"],
        "dx_sparcs": sparcs_product["dx"],
        "dy_sparcs": sparcs_product["dy"],

        # Measured COM before manual adjustment.
        "com_x": com_x,
        "com_y": com_y,
        "com_x_err": com_x_err,
        "com_y_err": com_y_err,

        # Requested COM position.
        "manual_com_x": manual_com_x,
        "manual_com_y": manual_com_y,

        # Additional translation relative to SPARCS.
        "dx_manual": dx_manual,
        "dy_manual": dy_manual,

        # Final positions.
        "final_com_x": final_com_x,
        "final_com_y": final_com_y,
        "optical_axis_x": optical_axis_x,
        "optical_axis_y": optical_axis_y,

        "reference_detector": sparcs_product["reference_detector"],
        "centroid_sigma": centroid_sigma,
        "threshold_fraction": threshold_fraction,
        "centroid_x_range": tuple(centroid_x_range),
        "centroid_y_range": tuple(centroid_y_range),
        "total_counts_used": int(centroid_image.total_counts_used),
    }

    return manual_data

