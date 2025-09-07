import numpy as np
from typing import Union

class Quaternion:
    def __init__(self, w=1, x=0, y=0, z=0):
        self.q = np.array([w, x, y, z])
    
    # methods for cleaner access to quaternion elements outside the class
    @property
    def w(self):
        return self.q[0]

    @property
    def x(self):
        return self.q[1]

    @property
    def y(self):
        return self.q[2]

    @property
    def z(self):
        return self.q[3]
    
    @property
    def vector(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    # overwritten arithmetic operations
    def __add__(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(*(self.q + other.q))

    def __sub__(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(*(self.q - other.q))

    def __mul__(self, other: Union["Quaternion", float, int]) -> "Quaternion":
        if isinstance(other, Quaternion):
            w1, x1, y1, z1 = self.q
            w2, x2, y2, z2 = other.q
            return Quaternion(
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2
            )
        elif isinstance(other, (float, int)):
            return Quaternion(*(self.q * other))
        else:
            raise TypeError("Quternion can only be multiplied by other quaternion (or number)")
    
    def __rmul__(self, other: Union[float, int]) -> "Quaternion":
        return self.__mul__(other)

    # normalization for unit constraint
    def normalize(self) -> None:
        norm = np.linalg.norm(self.q)
        if norm == 0:
            print("cannot normalize zero-quaternion, returning identity")
            self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.q /= norm
    
    def get_conjugate(self) -> "Quaternion":
        """ return conjugate quaternion """
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def to_euler(self) -> np.ndarray:
        """ return equivalent rotation in Euler angle notation as a vector of [roll, pitch, yaw] """
        w, x, y, z = self.q
        # Roll (x-axis rotation)
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
        # Pitch (y-axis rotation)
        pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))  # avoid gimbal lock
        # Yaw (z-axis rotation)
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
        return np.array([roll, pitch, yaw])
    
    def to_rotation_matrix(self) -> np.ndarray:
        """ return equivalent rotation in rotation matrix notation (3x3 matrix) """
        w, x, y, z = self.q
        
        return np.array([
            [1 - 2*(y**2 + z**2),   2*(x*y - w*z),          2*(x*z + w*y)],
            [2*(x*y + w*z),         1 - 2*(x**2 + z**2),    2*(y*z - w*x)],
            [2*(x*z - w*y),         2*(y*z + w*x),          1 - 2*(x**2 + y**2)]
        ])


def rotation_matrix_to_quaternion(R: np.ndarray) -> Quaternion:
    assert R.shape == (3,3), "Rotation matrix must be a 3x3 matrix!"
    trace = np.trace(R)
    if trace > 0:
        w = np.sqrt(1 + trace) / 2
        x = (R[2, 1] - R[1, 2]) / (4 * w)
        y = (R[0, 2] - R[2, 0]) / (4 * w)
        z = (R[1, 0] - R[0, 1]) / (4 * w)
    else:
        # Find the largest diagonal element to determine the primary axis
        i = np.argmax([R[0, 0], R[1, 1], R[2, 2]])
        if i == 0: # R[0,0] is largest
            x = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) / 2
            w = (R[2, 1] - R[1, 2]) / (4 * x)
            y = (R[0, 1] + R[1, 0]) / (4 * x)
            z = (R[0, 2] + R[2, 0]) / (4 * x)
        elif i == 1: # R[1,1] is largest
            y = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) / 2
            w = (R[0, 2] - R[2, 0]) / (4 * y)
            x = (R[0, 1] + R[1, 0]) / (4 * y)
            z = (R[1, 2] + R[2, 1]) / (4 * y)
        else: # R[2,2] is largest
            z = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) / 2
            w = (R[1, 0] - R[0, 1]) / (4 * z)
            x = (R[0, 2] + R[2, 0]) / (4 * z)
            y = (R[1, 2] + R[2, 1]) / (4 * z)
    return Quaternion(w, x, y, z)


def get_random_unit_quaternion() -> Quaternion:
    q = Quaternion(*np.random.uniform(-1, 1, 4))
    q.normalize()
    return q
