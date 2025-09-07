import numpy as np
from quaternions import Quaternion
from abc import ABC, abstractmethod


# defines reference frame transformations for vectors

def ned_to_body(vector_ned: np.ndarray, attitude_q: Quaternion) -> np.ndarray:
    """
    Rotates a vector from the NED frame to the body frame using the given attitude quaternion.
    """
    assert vector_ned.shape == (3,), "Vector in NED frame must be a 3 element vector!"

    # Convert the quaternion to a rotation matrix
    rotation_matrix = attitude_q.to_rotation_matrix()

    # Rotate the vector
    vector_body = rotation_matrix @ vector_ned

    return vector_body


def body_to_ned(vector_body: np.ndarray, attitude_q: Quaternion) -> np.ndarray:
    """
    Rotates a vector from the body frame to the NED frame using the given attitude quaternion.
    """
    assert vector_body.shape == (3,), "Vector in body frame must be a 3 element vector!"

    # Get the inverse rotation matrix (transpose of the original - special property)
    inv_rotation_matrix = attitude_q.to_rotation_matrix().T
    vector_ned = inv_rotation_matrix @ vector_body

    return vector_ned


def eci_to_lvlh(r_eci: np.ndarray, v_eci: np.ndarray) -> tuple:
    """ constructs lvlh frame basis vectors in eci frame """
    z_lvlh = -r_eci 
    z_lvlh /= np.linalg.norm(z_lvlh)

    y_lvlh = np.cross(r_eci, v_eci)
    y_lvlh /= np.linalg.norm(y_lvlh)
    
    x_lvlh = np.cross(y_lvlh, z_lvlh)

    return x_lvlh, y_lvlh, z_lvlh


def get_rotation_matrix(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """ 
    expects basis vectors [x, y, z] of target frame in source frame coordinates,

    returns rotation matrix that transforms a vector's coordinates from source to target frame
        - the inverse rotation can be performed using the transpose of the matrix!
    """
    assert x.shape == (3,) and y.shape == (3,) and z.shape == (3,), \
                        "basis vectors must be 3 element vectors!"

    R = np.array(([x, y, z])).T
    return R


class Frame(ABC):
    @abstractmethod
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    @property
    def vector(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


class ECI(Frame):
    pass


class ECEF(Frame):
    pass


class LVLH(Frame):
    pass


class Body(Frame):
    pass
