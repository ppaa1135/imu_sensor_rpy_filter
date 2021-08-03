import numpy as np
import math
import numbers
from math import sin, cos, tan, atan2

class Quaternion:
    def __init__(self, w_q, x=None, y=None, z=None):
        self._q = np.array([1, 0, 0, 0])

        if x != None and y != None and z != None:
            w = w_q
            q = np.array([w, x, y, z])

        elif isinstance(w_q, Quaternion):
            q = np.array(w_q.q)

        else:
            q = np.array(w_q)
            if len(q) != 4:
                raise ValueError("Expecting a 4-element array or w x y z as param")

        self.set_q(q)

    def set_q(self, q):
        self._q = q

    def get_q(self):
        return self._q

    def conj(self):
        return Quaternion([self._q[0], -self._q[1], -self._q[2], -self._q[3]])

    def __mul__(self, other):
        """
        multiply the given quaternion with another quaternion or a scalar
        :param other: a Quaternion object or a number
        :return:
        """
        if isinstance(other, Quaternion):
            w = self._q[0] * other._q[0] - self._q[1] * other._q[1] - self._q[2] * other._q[2] - self._q[3] * other._q[
                3]
            x = self._q[0] * other._q[1] + self._q[1] * other._q[0] + self._q[2] * other._q[3] - self._q[3] * other._q[
                2]
            y = self._q[0] * other._q[2] - self._q[1] * other._q[3] + self._q[2] * other._q[0] + self._q[3] * other._q[
                1]
            z = self._q[0] * other._q[3] + self._q[1] * other._q[2] - self._q[2] * other._q[1] + self._q[3] * other._q[
                0]

            return Quaternion(w, x, y, z)
        elif isinstance(other, numbers.Number):
            q = self._q * other
            return Quaternion(q)

    def __add__(self, other):
        """
        add two quaternions element-wise or add a scalar to each element of the quaternion
        :param other:
        :return:
        """
        if not isinstance(other, Quaternion):
            if len(other) != 4:
                raise TypeError("Quaternions must be added to other quaternions or a 4-element array")
            q = self._q + other
        else:
            q = self._q + other._q

        return Quaternion(q)

    def __getitem__(self, item):
        return self._q[item]

    def __array__(self):
        return self._q

    def to_euler123(self):
        # roll = np.arctan2(-2 * (self[2] * self[3] - self[0] * self[1]),
        #                   self[0] ** 2 - self[1] ** 2 - self[2] ** 2 + self[3] ** 2)
        # pitch = np.arcsin(2 * (self[1] * self[3] + self[0] * self[1]))
        # yaw = np.arctan2(-2 * (self[1] * self[2] - self[0] * self[3]),
        #                  self[0] ** 2 + self[1] ** 2 - self[2] ** 2 - self[3] ** 2)

        roll = np.arctan2(2*(self[0]*self[1] + self[2]*self[3]),
                          1 - 2*(self[1]**2 + self[2]**2))
        pitch = np.arcsin(2 * (self[0] * self[2] - self[3] * self[1]))
        yaw = np.arctan2(2*(self[0] * self[3] + self[1] * self[2]),
                         1 - 2*(self[2] ** 2 + self[3] ** 2))

        # to avoid gimbal lock #
        # when pitch +90 -90 #
        if self[0] * self[2] - self[3] * self[1] > 0.495:
            roll = 0
            yaw = 2 * np.arctan2(self[1], self[0])
            print("pitch +90")

        if self[0] * self[2] - self[3] * self[1] < -0.495:
            roll = 0
            yaw = -2 * np.arctan2(self[1], self[0])
            print("pitch -90")
        return roll, pitch, yaw


def euler_rotation_mat(roll, pitch, yaw):
    """
    euler angle to rotation matrix
    ##### be careful #####
    # not fixed cordinate #
    :param roll:
    :param pitch:
    :param yaw:
    :return:
    """
    cos_phi = math.cos(roll)
    sin_phi = math.sin(roll)

    cos_theta = math.cos(pitch)
    sin_theta = math.sin(pitch)

    cos_psi = math.cos(yaw)
    sin_psi = math.sin(yaw)

    roll_m = np.array([[1, 0, 0], \
                       [0, cos_phi, -sin_phi], \
                       [0, sin_phi, cos_phi]])

    pitch_m = np.array([[cos_theta, 0, sin_theta], \
                        [0, 1, 0], \
                        [-sin_theta, 0, cos_theta]])

    yaw_m = np.array([[cos_psi, -sin_psi, 0], \
                      [sin_psi, cos_psi, 0], \
                      [0, 0, 1]])

    R_mat = yaw_m @ pitch_m @ roll_m
    # R_mat = roll_m @ pitch_m @ yaw_m
    return R_mat

def quaternion_rotation_mat(q):
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]

    # R_mat = np.array([[1 - 2 * (y ** 2) - 2 * (z ** 2), 2 * x * y + 2 * z * w, 2 * x * z - 2 * y * w], \
    #                 [2 * x * y - 2 * z * w, 1 - 2 * (x ** 2) - 2 * (z ** 2), 2 * y * z + 2 * x * w], \
    #                 [2 * x * z + 2 * y * w, 2 * y * z - 2 * x * w, 1 - 2 * (x ** 2) - 2 * (y ** 2)]])

    R_mat = np.array([[1-2*(y**2 + z**2), 2*(x*y - w*z), 2*(w*y + x*z)],
                      [2*(x*y + w*z), 1-2*(x**2 + z**2), 2*(y*z - w*x)],
                      [2*(x*z - w*y), 2*(w*x + y*z), 1-2*(x**2 + y**2)]])

    return R_mat


if __name__ == '__main__':
    pass