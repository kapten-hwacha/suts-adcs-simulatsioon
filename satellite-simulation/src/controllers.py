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
    def __init__(self, kp: float, ki: float, kd: float, integral_max: float = -1):
        self.kp = kp  # proportional gain
        self.ki = ki  # integral gain
        self.kd = kd  # derivative gain
        self.integral = np.zeros(3)
        self.integral_max = integral_max

    def get_control_torque(self, satellite: "Satellite", dt: float) -> np.ndarray:
        """ calculates the control torque using the PID control law """

        q = satellite.q_body_to_eci_error
        
        # accumulate error and clip if threshold is set
        self.integral += q.vector * dt
        if self.integral_max > 0:
            np.clip(self.integral, -self.integral_max, self.integral_max, out=self.integral)

        # PID control law (using omega as derivative: cuts noise)
        torque = -self.kp * q.vector - self.ki * self.integral - self.kd * (satellite.omega - satellite.omega_target)

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



class LQR(Controller):
    """
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
    
    def __calculate_q_dot_vec(self, satellite):
        """ calculate the vector part of q-dot """
        w, x, y, z = satellite.attitude_q.q
        O = np.array((
            [w, -z, y],
            [z, w, -x],
            [-y, x, w]
        ))
        
        return 0.5 * O @ satellite.omega
    
    def __calculate_omega_dot(self, satellite):
        return np.linalg.inv(satellite.inertia_tensor) @ satellite.omega
    
    def get_control_torque(self, satellite: "Satellite", dt: float) -> np.ndarray:
        q = satellite.q_body_to_eci_error
        x = np.concat((satellite.omega, q.vector))

        return self.G @ x
