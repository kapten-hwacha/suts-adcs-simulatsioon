import numpy as np


def xyz_to_spherical(vec_xyz: np.ndarray) -> np.ndarray:
    """
    Converts a 3D Cartesian vector [x, y, z] to spherical coordinates [r, longitude_rad, latitude_rad].
    r: Radial distance
    longitude_rad: Longitude (azimuthal angle from positive x-axis)
    latitude_rad: Geocentric latitude (angle from equatorial plane)
    """
    assert vec_xyz.shape == (3,), "Vector must be 3D!"
    x, y, z = vec_xyz
    r = np.linalg.norm(vec_xyz)
    longitude_rad = np.arctan2(y, x)
    latitude_rad = np.arctan2(z, np.linalg.norm(vec_xyz[:2])) # Geocentric latitude
    return np.array([r, longitude_rad, latitude_rad])


def spherical_to_xyz(vec_spherical: np.ndarray) -> np.ndarray:
    """
    Converts spherical coordinates [r, longitude_rad, latitude_rad] to a 3D Cartesian vector [x, y, z].
    r: Radial distance
    longitude_rad: Longitude (azimuthal angle from positive x-axis)
    latitude_rad: Geocentric latitude (angle from equatorial plane)
    """
    assert vec_spherical.shape == (3,), "Vector must be 3D!"
    r, longitude_rad, latitude_rad = vec_spherical
    x = r * np.cos(latitude_rad) * np.cos(longitude_rad)
    y = r * np.cos(latitude_rad) * np.sin(longitude_rad)
    z = r * np.sin(latitude_rad)
    return np.array([x, y, z])
