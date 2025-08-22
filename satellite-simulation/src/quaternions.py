import numpy as np
from typing import Union

class Quaternion:
    def __init__(self, w, x, y, z):
        self.q = np.array([w, x, y, z])
    
    # methods for clener access to quaternion elements outside the class
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

    # overwritten arithmetic operations for cleanliness
    def __add__(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(*(self.q + other.q))

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

    # conversion to euler angles (as 3 element vector)
    def to_euler(self) -> np.ndarray:
        w, x, y, z = self.q
        # Roll (x-axis rotation)
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
        # Pitch (y-axis rotation)
        pitch = np.arcsin(2*(w*y - z*x))
        # Yaw (z-axis rotation)
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
        return np.array([roll, pitch, yaw])
    
    def to_rotation_matrix(self) -> np.ndarray:
        w, x, y, z = self.q
        
        return np.array([
            [1 - 2*(y**2 + z**2),   2*(x*y - w*z),          2*(x*z + w*y)],
            [2*(x*y + w*z),         1 - 2*(x**2 + z**2),    2*(y*z - w*x)],
            [2*(x*z - w*y),         2*(y*z + w*x),          1 - 2*(x**2 + y**2)]
        ])


def get_random_unit_quaternion() -> "Quaternion":
    q = Quaternion(*np.random.uniform(-1, 1, 4))
    q.normalize()
    return q