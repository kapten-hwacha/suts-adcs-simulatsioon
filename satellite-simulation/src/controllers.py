import numpy as np
from quaternions import Quaternion
from abc import ABC, abstractmethod
from scipy.linalg import sqrtm


class Controller(ABC):
    """ abstract base class for all controller implementations """

    @abstractmethod
    # returns control torque vector (3D)
    def get_control_torque(self, satellite: "Satellite", dt: float) -> np.ndarray:
        """ calculate control torque based on on information in satellite class object """
        pass


class PID(Controller):
    def __init__(self, kp: float, ki: float, kd: float, integral_max: float = -1, tau_derivative: float = 1e-9):
        self.kp = kp  # proportional gain
        self.ki = ki  # integral gain
        self.kd = kd  # derivative gain
        self.integral = np.zeros(3)

        self.integral_max = integral_max
        self.tau_d = tau_derivative
        self.filtered_omega_error = np.zeros(3)

        # for debugging
        self.proportional_terms = []
        self.integral_terms = []
        self.derivative_terms = []

    def get_control_torque(self, satellite: "Satellite", dt: float) -> np.ndarray:
        """ calculates the control torque using the PID control law """

        q = satellite.q_body_to_eci_error
        
        # accumulate error and clip if threshold is set
        self.integral += q.vector * dt
        if self.integral_max > 0:
            np.clip(self.integral, -self.integral_max, self.integral_max, out=self.integral)

        omega_error = satellite.omega - satellite.omega_target
        
        # first order low pass filter for the derivative part
        if self.tau_d > 0 and dt > 0:
            alpha = dt / (self.tau_d + dt)
            self.filtered_omega_error = alpha * omega_error + (1 - alpha) * self.filtered_omega_error

        # PID control law
        proportional_term = -self.kp * q.vector
        integral_term = - self.ki * self.integral
        derivative_term = -self.kd * self.filtered_omega_error
        torque = proportional_term + integral_term + derivative_term

        # for debugging
        self.proportional_terms.append(proportional_term)
        self.integral_terms.append(integral_term)
        self.derivative_terms.append(derivative_term)

        return torque  # return 1d vector (x, y, z)


class BDot(Controller):
    def __init__(self, gain: float, B_initial: np.ndarray):
        self.B_prev = B_initial
        self.gain = gain

    def get_control_torque(self, satellite: "Satellite", dt: float) -> np.ndarray:
        """ calculates control torque using b-dot control law """
        
        # would using a better numerical differentiation method be beneficial?

        if dt == 0:
            return np.zeros(3)
        
        B_dot = (satellite.B_field_gauss - self.B_prev) / dt
        # apply bdot control law
        torque = np.cross(-self.gain * B_dot, satellite.B_field_gauss)
        
        self.B_prev = satellite.B_field_gauss  # update for next iteration
        
        return torque.flatten()  # return 1d vector (x, y, z)



class LQR_Yang(Controller):
    """
    this controller is derived to asymptodically converge to
    align the satellite WITH THE ECI FRAME!
    ie this derivation can not be used to turn the satellite to an arbitrary point

    optimal control is uniquely given by u = -R^-1 @ B @ F @ x

        - per Yang (DOI: 10.1061/(ASCE)AS.1943-5525.0000142) we take that
        matrices J, Q, R are diagonal; this greatly simplifies the controller

        - for the system to be globally stable R must be chosen so
        R = cQ2 or R = c Q2 @ J, where c is const.
    """
    
    A = np.zeros(shape=(6, 6))
    A[3:6, 0:3] = 0.5 * np.identity(3)
    
    def __init__(self, R: np.ndarray, Q: np.ndarray, J: np.ndarray):
        """ expects the satellite's inertia tensor in g/m^2 """
        B = np.vstack([np.linalg.inv(J), np.zeros(shape=(3, 3))])
        Q1 = Q[0:3, 0:3]
        Q2 = Q[3:6, 3:6]

        F12 = J @ sqrtm(R) @ sqrtm(Q2)

        F11 = J @ sqrtm(R) @ sqrtm(Q1 + 0.5 * \
            (J @ sqrtm(R) @ sqrtm (Q2) + sqrtm(Q2) @ sqrtm(R) @ J))

        F22 = 2 * sqrtm(Q2) @ sqrtm(Q1 + J @ sqrtm(R) @ sqrtm(Q2))
        
        F = np.block([[F11, F12], [F12.T, F22]])

        R_inv = np.linalg.inv(R)

        self.G = -R_inv @ B.T @ F
        print(f"B: {B.shape}, F: {F.shape}, G: {self.G.shape}")

    def __lyapunov_algebraic_eq(self, F, A, B, R, Q):
        """ defines the F matrix """
        R_inv = np.linalg.inv(R)
        return -F @ A - A.T @ F + F @ B @ R_inv @ B.T @ F - Q
    
    def get_control_torque(self, satellite: "Satellite", dt: float) -> np.ndarray:
        q = satellite.q_body_to_eci_error
        x = np.concatenate((satellite.omega, q.vector))

        return self.G @ x
