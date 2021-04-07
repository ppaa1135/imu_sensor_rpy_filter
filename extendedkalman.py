import numpy as np
from numpy.linalg import norm, inv
import matplotlib.pyplot as plt
from math import sin, cos, tan, atan2
import math

from mathlib import Quaternion, euler_rotation_mat, quaternion_rotation_mat
from excel_read import excel_file_read
from plot3d import plot_3d


class ExtendedKalman():
    def __init__(self, dt):
        self.H = np.array([[1, 0, 0],
                           [0, 1, 0]])

        self.Q = np.array([[0.00001, 0, 0],
                           [0, 0.00001, 0],
                           [0, 0, 0.00001]])

        self.R = 100 * np.identity(2)

        self.x = np.zeros(3)
        self.P = 100 * np.identity(3)

        self.dt = dt

    def cal_A(self, p, q, r):
        A = np.zeros((3, 3))

        phi = self.x[0]
        theta = self.x[1]

        A[0, 0] = q * cos(phi) * tan(theta) - r * sin(phi) * tan(theta)
        A[0, 1] = q * sin(phi) / (cos(theta) ** 2) + r * cos(phi) / (cos(theta) ** 2)
        A[0, 2] = 0

        A[1, 0] = -q * sin(phi) - r * cos(phi)
        A[1, 1] = 0
        A[1, 2] = 0

        A[2, 0] = q * cos(phi) / cos(theta) - r * sin(phi) / cos(theta)
        A[2, 1] = q * sin(phi) / cos(theta) * tan(theta) + r * cos(phi) / cos(theta) * tan(theta)
        A[2, 2] = 0

        A = np.identity(3) + A * self.dt

        return A

    def cal_acc(self, ax, ay, az):

        phi = atan2(ay, az)
        theta = atan2(ax, math.sqrt(ay**2 + az**2))

        return phi, theta

    def cal_x_pred(self, p, q, r):
        phi = self.x[0]
        theta = self.x[1]

        xdot = np.zeros(3)
        xdot[0] = p + q*sin(phi)*tan(theta) + r*cos(phi)*tan(theta)
        xdot[1] = q*cos(phi) - r*sin(phi)
        xdot[2] = q*sin(phi)/cos(theta) + r*cos(phi)/cos(theta)

        x_pred = self.x + xdot*self.dt

        return x_pred

    def update(self, imu):
        self.A = self.cal_A(imu[3], imu[4], imu[5])
        phi_a, theta_a = self.cal_acc(imu[0], imu[1], imu[2])

        self.z = np.array([phi_a, theta_a])

        self.x_pred = self.cal_x_pred(imu[3], imu[4], imu[5])
        self.p_pred = self.A @ self.P @ self.A.T + self.Q

        self.K = self.p_pred @ self.H.T @ inv(self.H @ self.p_pred @ self.H.T + self.R)

        self.x = self.x_pred + self.K @ (self.z - self.H @ self.x_pred)
        self.P = self.p_pred - self.K @ self.H @ self.p_pred

        return self.x

if __name__ == '__main__':
    imu = excel_file_read("./imu_roll_pitch_yaw_200hz.xlsx")
    # imu = calibration_imu(imu)

    ekf = ExtendedKalman(0.005)

    rpy = []
    imu_plot = []
    for one_imu in imu:
        x = ekf.update(one_imu)
        rpy.append([x[0], x[1], x[2]])

        r_mat = euler_rotation_mat(x[0], x[1], x[2])

        plot_xyz = r_mat @ np.identity(3)

        imu_plot.append(plot_xyz.T)
    rpy = np.array(rpy) * 180 / math.pi

    plt.plot(rpy[:, 0], 'r-')
    plt.plot(rpy[:, 1], 'b-')
    plt.plot(rpy[:, 2], 'g-')
    plt.grid()
    plt.show()

    imu_plot = np.array(imu_plot)
    plot_3d(imu_plot)