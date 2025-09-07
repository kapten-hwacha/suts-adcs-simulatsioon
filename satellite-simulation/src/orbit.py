import numpy as np
from scipy.optimize import newton
from coordinates import xyz_to_spherical


RADIUS_EARTH = 6371.0  # km
OMEGA_EARTH = 7.2921159e-5  # Earth's angular rate (rad/s)
MU = 398600.0  # gravitational constant times Earth's mass (km^3/s^2)


def kepler(E: float, M: float, e: float) -> float:
    """ Kepler's equation in the form f(E) = 0, where M = E - e * sin(E) """
    return E - e * np.sin(E) - M


def dE_kepler(E: float, M: float, e: float) -> float:
    """ Derivative of Kepler's equation with respect to Eccentric Anomaly E """
    return 1 - e * np.cos(E)


def calculate_semi_major_axis(altitude_km_periapsis: float, eccentricity: float) -> float:
    return (RADIUS_EARTH + altitude_km_periapsis) / (1 - eccentricity)


def calculate_eccentric_anomaly(nu_deg: float, e: float) -> float:
    """ calculates eccentric anomaly in radians from true anomaly and eccentricity """
    nu = np.deg2rad(nu_deg)
    return 2 * np.arctan2(np.sqrt(1 - e) * np.sin(nu / 2), np.sqrt(1 + e) * np.cos(nu / 2))


def calculate_true_anomaly(E: float, e: float):
    """ calculates true anomaly in radians from eccentric anomaly and eccentricity """
    return 2 * np.atan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))


class Orbit:
    """ class for orbit propagation and modelling """
    def __init__(self, semi_major_axis_km: float, inclination_deg: float, 
                eccentricity: float = 0, argument_of_periapsis_deg: float = 0,
                raan_deg: float = 0, initial_true_anomaly_deg: float = 0,
                earth_rotation_angle_deg: float = 0
                ):
        # check that orbit parameters are valid
        assert semi_major_axis_km > 0, "semi-major axis must be a positive value!"
        assert eccentricity >= 0 and eccentricity < 1, "eccentricity must be in range [0, 1), ie escape orbits are not implemented!"
        assert inclination_deg >= 0 and inclination_deg <= 180, "inclination must be in range of [0, 180] deg"
        assert argument_of_periapsis_deg >= 0 and argument_of_periapsis_deg < 360, "argument of periapsis must be in range of [0, 360) deg"
        assert raan_deg >= 0 and raan_deg <= 360, "raan must be in range of [0, 360] deg"
        assert initial_true_anomaly_deg >= 0 and initial_true_anomaly_deg < 360, "mean anomaly at epoch must be in range of [0, 360) deg"
        assert earth_rotation_angle_deg >= 0 and earth_rotation_angle_deg < 360, "initial Earth's rotation angle must be in range of [0, 360) deg"

        self.inclination = np.deg2rad(inclination_deg)
        self.eccentricity = eccentricity  # e
        self.semi_major_axis = semi_major_axis_km  # a
        self.angular_rate = np.sqrt(MU / self.semi_major_axis**3)
        
        self.raan = np.deg2rad(raan_deg)  # right ascension of the ascending node
        self.omega = np.deg2rad(argument_of_periapsis_deg)  # argument of periapsis
        self.initial_ERA = np.deg2rad(earth_rotation_angle_deg)

        # rotation matrices for rotations from perifocal to ECI frame
        R_Z_raan = np.array((
            [np.cos(self.raan), -np.sin(self.raan), 0],
            [np.sin(self.raan), np.cos(self.raan), 0],
            [0, 0, 1]
        ))

        R_X_incl = np.array((
            [1, 0, 0],
            [0, np.cos(self.inclination), -np.sin(self.inclination)],
            [0, np.sin(self.inclination), np.cos(self.inclination)]
        ))

        R_Z_peri = np.array((
            [np.cos(self.omega), -np.sin(self.omega), 0],
            [np.sin(self.omega), np.cos(self.omega), 0],
            [0, 0, 1]
        ))

        # this rotation matrix will stay constant for a constant orbit
        self.R_perifocal_to_eci = R_Z_raan @ R_X_incl @ R_Z_peri
        
        E = calculate_eccentric_anomaly(initial_true_anomaly_deg, self.eccentricity)
        self.M0 = E - self.eccentricity * np.sin(E)  # initial mean anomaly (in radians!)

    def __solve_kepler(self, mean_anomaly: float) -> float:
        """ class method for solving the transcendental Kepler's equation using Newton's method """
        E0 = mean_anomaly  # a rather efficient initial guess
        E = newton(kepler, E0, dE_kepler, args=(mean_anomaly, self.eccentricity))
        return E
    
    def vector_eci_to_ecef(self, time_elapsed: float, vec_eci: np.ndarray) -> np.ndarray:
        ERA = self.initial_ERA + OMEGA_EARTH * time_elapsed
        R_Z_ERA = np.array((
            [np.cos(ERA), -np.sin(ERA), 0],
            [np.sin(ERA), np.cos(ERA), 0],
            [0, 0, 1]
        ))
        return R_Z_ERA @ vec_eci

    def __get_velocity_perifocal(self, nu, r) -> np.ndarray:
        """ returns a velocity vector [x, y, z] in the perifocal frame """
        h = np.sqrt(MU * self.semi_major_axis * (1 - self.eccentricity**2))
        vx = -(MU / h) * np.sin(nu)
        vy = (MU / h) * (self.eccentricity + np.cos(nu))
        return np.array(([vx, vy, 0]))

    def propagate(self, time_elapsed: float) -> tuple:
        """ 
        function to be iteratively called for orbit propagation:
        returns position and velocity vector [x, y, z] in the eci frame
        """
        M = self.angular_rate * time_elapsed + self.M0  # mean anomaly

        r = self.semi_major_axis
        if self.eccentricity == 0:
            nu = M  # true anomaly = mean anomaly if circular orbit
        else:
            E = self.__solve_kepler(M)
            nu = calculate_true_anomaly(E, self.eccentricity)
            r *= (1 - self.eccentricity * np.cos(E))

        # translate position vector from perifocal to ECI frame
        r_peri = np.array([r * np.cos(nu), r * np.sin(nu), 0])
        r_eci = self.R_perifocal_to_eci @ r_peri

        v_peri = self.__get_velocity_perifocal(nu, r)
        v_eci = self.R_perifocal_to_eci @ v_peri

        return r_eci, v_eci
