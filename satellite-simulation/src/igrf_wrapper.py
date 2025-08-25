import numpy as np
import os
from IGRF import igrf_utils as iut
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# path to the coefficents
IGRF_FILE = os.path.join(SCRIPT_DIR, 'IGRF/SHC_files/IGRF14.SHC')
# load the coefficients
igrf_model = iut.load_shcfile(IGRF_FILE)


def get_b_field_NED(latitude: float, longitude: float, altitude: float, date: datetime) -> np.ndarray:
    """ 
    expects latitude and longitude in degrees, altitude in kilometers
    returns magnetic field vector in Gauss (in NED frame)
    """
    # convert year to decimal year
    year_start = datetime(date.year, 1, 1)
    year_end = datetime(date.year + 1, 1, 1)
    decimal_year = date.year + (date - year_start).total_seconds() / ((year_end - year_start).total_seconds())
    
    # interpolate igrf coefficients linearily
    idx = np.searchsorted(igrf_model.time, decimal_year) - 1
    t0, t1 = igrf_model.time[idx], igrf_model.time[idx+1]
    g0, g1 = igrf_model.coeffs[:, idx], igrf_model.coeffs[:, idx+1]
    coeffs = g0 + (g1 - g0) * (decimal_year - t0) / (t1 - t0)
    
    # convert coefficients to geocentric
    colat_deg = 90.0 - latitude
    radius_gc, colat_gc_deg, sd, cd = iut.gg_to_geo(altitude, colat_deg)
    
    # synthesize the magentic field in geocentric spherical coordinates
    B_r, B_theta, B_phi = iut.synth_values(coeffs, radius_gc, colat_gc_deg, longitude, nmax=13)
    
    # rotate the field components to geodetic frame
    B_x = B_theta * cd + B_r * sd   # north component
    B_z = B_r * cd - B_theta * sd   # vertical component (down) nadir
    B_y = B_phi                     # east component remains the same
    
    return np.array([B_x, B_y, B_z]) / 1e5

