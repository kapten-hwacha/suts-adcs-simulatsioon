import numpy as np
from quaternions import Quaternion

# defines reference frame transformations for vectors

def body_to_ned(vector_ned: np.ndarray, attitude_q: Quaternion) -> np.ndarray:
    """
    Rotates a vector from the NED frame to the body frame using the given attitude quaternion.
    """
    assert vector_ned.shape == (3,), "Vector in NED frame must be a 3 element vector!"

    # Convert the quaternion to a rotation matrix
    rotation_matrix = attitude_q.to_rotation_matrix()

    # Rotate the vector
    vector_body = rotation_matrix @ vector_ned

    return vector_body


def ned_to_body(vector_body: np.ndarray, attitude_q: Quaternion) -> np.ndarray:
    """
    Rotates a vector from the body frame to the NED frame using the given attitude quaternion.
    """
    assert vector_body.shape == (3,), "Vector in body frame must be a 3 element vector!"

    # Get the inverse rotation matrix (transpose of the original - special property)
    inv_rotation_matrix = attitude_q.to_rotation_matrix().T
    vector_ned = inv_rotation_matrix @ vector_body

    return vector_ned
