import numpy as np
from scipy.integrate import solve_ivp
from typing import TypedDict
from quaternions import Quaternion
from controllers import Controller
from frames import *
from igrf_wrapper import *


class ControllerMap(TypedDict):
    DETUMBLE: Controller
    COARSE_POINT_NADIR: Controller
    FINE_POINT_NADIR: Controller


class Satellite:
    """
    @todo implement max jerk for actuators
    """
    # all values are in satellite's body frame
    def __init__(self,
                q_body_eci: Quaternion,
                angular_velocity: np.ndarray,
                inertia_tensor: np.ndarray,
                controllers: ControllerMap,
                detumbled_omega: float,
                magnetic_field: np.ndarray = np.zeros(3),
                state: str = "DETUMBLE",
                ):

        assert angular_velocity.shape == (3,), "Angular velocity vector must be a 3 element vector!"
        assert inertia_tensor.shape == (3, 3), "Inertia tensor must be a 3x3 matrix!"

        self.q_body_to_eci_error = Quaternion()  # valid initialization
        self.q_body_to_eci = q_body_eci  # Quaternion representing the satellite's attidude (in vector form)
        self.omega = angular_velocity  # Angular velocity vector (x, y, z)
        self.omega_target = np.zeros(3)
        self.controllers = controllers
        self.inertia_tensor = inertia_tensor
        self.inertia_tensor_inv = np.linalg.inv(self.inertia_tensor)
        self.B_field_gauss = magnetic_field
        self.torque = np.zeros(3)
        self.state = state

        # these values ought to be realistic (per axis)
        # can be changed to match with the actual specs
        self.max_torque = 0.001  # Nm
        self.max_actuator_jerk = 0.005  # Nm / s
        
        self.fine_point_threshold = np.sqrt(0.02**2 * 3)
        self.revert_to_coarse_point_threshold = np.sqrt(0.1**2 * 3)

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

    def update(self, dt: float, angular_speed_orbit: float, 
                q_body_to_eci_target: Quaternion, 
                omega_body_target: np.ndarray = np.zeros(3)):
        """ 
        main iteration loop
            @todo: implement disturbances
        """
        # set new error quaternion (body to eci rotation)
        
        # trial and error :)
        q_body_to_eci_error = q_body_to_eci_target.get_conjugate() * self.q_body_to_eci  # option 1, works with pid

        q_body_to_eci_error.normalize()  # there might be fp numerical errors

        # ensure the scalar part of the error quaternion is positive to avoid jumps
        # this is exhibits mysterious behaviour atm
        # @todo understand why this works as it does atm

        # q_body_to_eci_error.unflip()
        
        # approach 0
        self.q_body_to_eci_error = q_body_to_eci_error

        # print(f'attitude error is {self.q_body_to_eci_error}')

        self.omega_target = omega_body_target
        
        # DYNAMICS
        if self.state not in self.controllers:
            raise ValueError(f"invalid state: {self.state}")

        controller = self.controllers[self.state]
        commanded_torque = controller.get_control_torque(self, dt)
        applied_torque = np.clip(commanded_torque, -self.max_torque, self.max_torque)

        # @todo model external torque, ie disturbances somehow
        external_torque = 0

        torque = applied_torque + external_torque
        max_delta_torque = self.max_actuator_jerk * dt
        self.torque = np.clip(torque, self.torque - max_delta_torque , self.torque + max_delta_torque)

        # KINEMATICS
        
        # using advanced numerical solver for stability
        y0 = np.concatenate((self.omega, self.q_body_to_eci.q))
        sol = solve_ivp(
            fun=self.__get_derivatives,
            t_span=[0, dt],
            y0=y0,
            method='RK45',
        )

        # unpack solver results
        self.omega = sol.y[0:3, -1]
        self.q_body_to_eci = Quaternion(*sol.y[3:7, -1])

        # normalize the quaternion to ensure it remains a unit quaternion
        self.q_body_to_eci.normalize()

        # state transistion logic
        match self.state:

            case "DETUMBLE":
                if self.__is_detumbled(angular_speed_orbit):
                    self.state = "COARSE_POINT_NADIR"

            case "COARSE_POINT_NADIR":
                if False and self.q_body_to_eci_error.norm < self.fine_point_threshold:
                    self.state = "FINE_POINT_NADIR"
                    print(f'transistioned to {self.state}')

            case "FINE_POINT_NADIR":
                if False and self.q_body_to_eci_error.norm > self.revert_to_coarse_point_threshold:
                    self.state = "COARSE_POINT_NADIR"
                    print(f'transistioned to {self.state}')

            case _:
                raise NotImplementedError(f"state {self.state} has no implemented transistion logic!")

        return commanded_torque, applied_torque  # return both for debugging

    def get_attitude(self) -> np.ndarray:
        """ returns satellite orientation as Euler angles """
        return self.q_body_to_eci.to_euler()
