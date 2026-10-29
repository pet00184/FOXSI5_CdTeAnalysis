import numpy as np
from astropy.io import fits
from astropy.table import Table

DEFAULT_INTERVALS = [
    ("TI1", "TI2", "Target Alpha Pre-Shift"),
    ("TI3", "TI4", "Target Alpha"),
    ("TI5", "TI6", "Target Echo"),
    ("TI7", "TI8", "Target Charlie"),
]

def load_cdte_fits(filepath):
    """Loads a CdTe FITS file and returns its data Table and Header."""
    with fits.open(filepath) as hdul:
        header = hdul[0].header
        data = Table(hdul[1].data)
    return data, header

def parse_target_intervals(data, header, intervals=None):
    """
    Parses TI header tags and splits event data into target intervals.
    """
    if intervals is None:
        intervals = DEFAULT_INTERVALS

    ti = data["ti"]
    ti_keys = ["TI1", "TI2", "TI3", "TI4", "TI5", "TI6", "TI7", "TI8", "TI9"]

    indices = {}
    for key in ti_keys:
        value = header.get(key)
        if value is None or str(value).lower() == "nan":
            continue
        indices[key] = np.argmin(np.abs(ti - float(value)))

    target_data = {}
    for start_key, end_key, label in intervals:
        if start_key not in indices or end_key not in indices:
            continue
        i0 = indices[start_key]
        i1 = indices[end_key]
        target_data[label] = {"Data": data[i0:i1]}

    if "TI9" in indices:
        i0 = indices["TI9"]
        target_data["Target Alpha 2"] = {"Data": data[i0:len(ti)]}

    return target_data