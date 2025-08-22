import numpy as np

RADIUS_EARTH = 6371.0
MU = 398600.0  # Gravitational constant times Earth's mass (km^3/s^2)

class Orbit:
    def __init__(self, altitude_km: float, inclination_deg: float, 
                latitude_deg: float = 0, longitude_deg: float = 0):
        self.inclination = inclination_deg
        self.latitude = latitude_deg
        self.longitude = longitude_deg

        a = RADIUS_EARTH + altitude_km  # semi-major axis
        self.period = 2 * np.pi * np.sqrt(a**3 / MU)
        self.omega = 2 * np.pi / self.period

    def propagate(self, time_elapsed: float):
        """ 
        returns the latitude and longitude of the satellite at the time
            @todo: implement elliptical orbits
            # if orbit is elliptical, then we are looking for a solution to Kepler's equation -> numerical
            # also then we must use theta + gammma, where gamma is the argument of periapsis instead of theta
        """
        M = self.omega * time_elapsed  # mean anomaly
        theta = M  # true anomaly = mean anomaly if circular orbit 
        
        latitude_rad = np.arcsin(np.sin(np.deg2rad(self.inclination)) * np.sin(theta))
        longitude_rad = np.arctan2(np.cos(theta), np.cos(np.deg2rad(self.inclination)) * np.sin(theta))

        latitude_deg = np.rad2deg(latitude_rad)
        longitude_deg = np.rad2deg(longitude_rad)

        return latitude_deg, longitude_deg
