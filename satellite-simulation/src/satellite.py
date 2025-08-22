import numpy as np
from scipy.integrate import solve_ivp
from quaternions import Quaternion
from controllers import Controller
from frames import *
from igrf_wrapper import *


# for numerical solvers
def calculate_alpha(torque: np.ndarray, omega: np.ndarray, inertia_tensor: np.ndarray) -> np.ndarray:
        """ calculate angular acceleration from applied torque """
        inertia_tensor_inv = np.linalg.inv(inertia_tensor)
        return inertia_tensor_inv @ (torque - np.cross(omega, inertia_tensor @ omega))


# for numerical solvers
def calculate_q_dot(omega: np.ndarray, attitude: np.ndarray) -> np.ndarray:
        """ Calculate the derivative of the attitude quaternion: dq/dt = 0.5 * q * omega_q """
        # Convert angular velocity vector to pure quaternion (0, x, y, z)
        omega_q = Quaternion(0, *omega)
        attitude_q = Quaternion(*attitude)

        # quaternion kinematic equation
        q_dot = 0.5 * (attitude_q * omega_q)
        return q_dot.q


class Satellite:
    def __init__(self,
                attidude_quaternion: Quaternion,
                angular_velocity: np.ndarray,
                controller: Controller,
                inertia_tensor: np.ndarray,
                magnetic_field: np.ndarray = np.zeros(3)):

        assert angular_velocity.shape == (3,), "Angular velocity vector must be a 3 element vector!"
        assert inertia_tensor.shape == (3, 3), "Inertia tensor must be a 3x3 matrix!"

        self.attitude_q = attidude_quaternion  # Quaternion representing the satellite's attidude (in vector form)
        self.omega = angular_velocity  # Angular velocity vector (x, y, z)
        self.controller = controller
        self.inertia_tensor = inertia_tensor
        self.B_field_gauss = magnetic_field
        self.torque = np.zeros(3)

    def update(self, dt) -> np.ndarray:
        """ main iteration loop """
        # DYNAMICS
        # get control torque from controller
        control_torque = self.controller.get_control(self, dt)
        # model external torque, ie disturbances somehow
        external_torque = 0
        # handle rotational dynamics
        self.torque = control_torque + external_torque

        sol = solve_ivp(
            fun=get_derivatives,
            t_span=[t_start, t_end],
            y0=y0,
            args=(self,),
            method='RK45',
        )

        # Normalize the quaternion to ensure it remains a unit quaternion
        self.attitude_q.normalize()
        return control_torque

    def get_attitude(self) -> np.ndarray:
        """ returns satellite orientation as Euler angles """
        return self.attitude_q.vector
