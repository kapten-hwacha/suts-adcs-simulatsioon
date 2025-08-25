import numpy as np
from scipy.integrate import solve_ivp
from typing import TypedDict
from quaternions import Quaternion
from controllers import Controller
from frames import *
from igrf_wrapper import *


class ControllerMap(TypedDict):
    DETUMBLE: Controller
    POINT: Controller


class Satellite:
    # all values are in satellite's body frame
    def __init__(self,
                attidude_quaternion: Quaternion,
                angular_velocity: np.ndarray,
                inertia_tensor: np.ndarray,
                controllers: ControllerMap,
                magnetic_field: np.ndarray = np.zeros(3),
                state: str = "DETUMBLE"
                ):

        assert angular_velocity.shape == (3,), "Angular velocity vector must be a 3 element vector!"
        assert inertia_tensor.shape == (3, 3), "Inertia tensor must be a 3x3 matrix!"

        self.attitude_q = attidude_quaternion  # Quaternion representing the satellite's attidude (in vector form)
        self.omega = angular_velocity  # Angular velocity vector (x, y, z)
        self.controllers = controllers
        self.inertia_tensor = inertia_tensor
        self.inertia_tensor_inv = np.linalg.inv(self.inertia_tensor)
        self.B_field_gauss = magnetic_field
        self.torque = np.zeros(3)
        self.state = state

    # for numerical solvers
    def __calculate_alpha(self, torque: np.ndarray, omega: np.ndarray) -> np.ndarray:
            """ calculate angular acceleration from applied torque """
            return self.inertia_tensor_inv @ (torque - np.cross(omega, self.inertia_tensor @ omega))

    # for numerical solvers
    def __calculate_q_dot(self, omega: np.ndarray, attitude: np.ndarray) -> np.ndarray:
            """ Calculate the derivative of the attitude quaternion: dq/dt = 0.5 * q * omega_q """
            # Convert angular velocity vector to pure quaternion (0, x, y, z)
            omega_q = Quaternion(0, *omega)
            attitude_q = Quaternion(*attitude)

            # quaternion kinematic equation
            q_dot = 0.5 * (attitude_q * omega_q)
            return q_dot.q

    def __get_derivatives(self, t, y: np.ndarray):
        omega = y[0:3]
        attitude = y[3:7]
        
        alpha = self.__calculate_alpha(self.torque, omega)
        q_dot = self.__calculate_q_dot(omega, attitude)

        return np.concatenate((alpha, q_dot))

    def __is_detumbled(self, angular_speed_orbit: float):
        THRESHOLD_RATIO = 3
        angular_speed = np.linalg.norm(self.omega)
        if (angular_speed / angular_speed_orbit) < THRESHOLD_RATIO:
            return True
        else:
            return False

    def update(self, dt: float, angular_speed_orbit: float):
        """ 
        main iteration loop
            @todo: implement disturbances
        """
        # DYNAMICS

        if self.state not in self.controllers:
            raise ValueError(f"invalid state: {self.state}")
        
        controller = self.controllers[self.state]
        control_torque = controller.get_control_torque(self, dt)
        external_torque = 0  # model external torque, ie disturbances somehow
        self.torque = control_torque + external_torque

        # KINEMATICS
        
        # using advanced numerical solver for stability
        y0 = np.concatenate((self.omega, self.attitude_q.q))
        sol = solve_ivp(
            fun=self.__get_derivatives,
            t_span=[0, dt],
            y0=y0,
            method='RK45',
        )

        # unpack solver results
        self.omega = sol.y[0:3, -1]
        self.attitude_q = Quaternion(*sol.y[3:7, -1])

        # normalize the quaternion to ensure it remains a unit quaternion
        self.attitude_q.normalize()
        
        # condition for changing the state
        detumbled = False
        if self.state == "DETUMBLE" and self.__is_detumbled(angular_speed_orbit):
            self.state = "POINT"
            detumbled = True

        return control_torque, detumbled  # return for debugging

    def get_attitude(self) -> np.ndarray:
        """ returns satellite orientation as Euler angles """
        return self.attitude_q.vector
