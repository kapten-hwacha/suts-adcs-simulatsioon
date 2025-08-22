import numpy as np
from quaternions import Quaternion
from abc import ABC, abstractmethod


class Controller(ABC):
    """ abstract base class for all controller implementations """
    
    @abstractmethod
    # returns control torque vector (3D)
    def get_control(self, satellite: "Satellite", dt: float) -> np.ndarray:
        """ calculate control torque based on on information in satellite class object """
        pass


class PID(Controller):
    def __init__(self, kp: float, ki: float, kd: float, target_q: Quaternion, integral_max: float = -1):
        self.kp = kp  # proportional gain
        self.ki = ki  # integral gain
        self.kd = kd  # derivative gain
        self.integral = np.zeros(3)
        self.integral_max = integral_max
        
        self.target_q = target_q

    def get_control(self, satellite: "Satellite", dt: float) -> np.ndarray:
        """ calculates the control torque using the PID control law """

        # Calculate the error quaternion: q_error = q_target * q_current_conjugate
        error_q = self.target_q * satellite.attidude_q.get_conjugate()
        
        # accumulate error and clip if threshold is set
        self.integral += error_q.vector * dt
        if self.integral_max > 0:
            np.clip(self.integral, -self.integral_max, self.integral_max, out=self.integral)

        # PID control law (using omega as derivative: cuts noise)
        torque = -self.kp * error_q.vector - self.ki * self.integral - self.kd * satellite.omega.flatten()

        return torque  # return 1d vector (x, y, z)


class BDot(Controller):
    def __init__(self, gain: float, B_initial: np.ndarray):
        self.B_prev = B_initial
        self.gain = gain

    def get_control(self, satellite: "Satellite", dt: float) -> np.ndarray:
        """ calculates control torque using b-dot control law """
        
        # would using a better numerical differentiation method be beneficial?

        if dt == 0:
            return np.zeros(3)
        
        B_dot = (satellite.B_field_gauss - self.B_prev) / dt
        # apply bdot control law
        torque = np.cross(-self.gain * B_dot, satellite.B_field_gauss)
        
        self.B_prev = satellite.B_field_gauss  # update for next iteration
        
        return torque.flatten()  # return 1d vector (x, y, z)
