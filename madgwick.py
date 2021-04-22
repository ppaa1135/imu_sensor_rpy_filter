import numpy as np
from numpy.linalg import norm, inv
import matplotlib.pyplot as plt

from lib.mathlib_v2 import *
from lib.excel_read import excel_file_read
from lib.plot3d import plot_3d

class Madgwick:
	def __init__(self):
		self.dt = 0.005
		self.quaternion = quaternion(1,0,0,0)

		######## if beta is high, faster but not smooth
		self.beta = 0.5

	def update_6imu(self, imu):
		accelerometer = imu[0:3].copy()
		gyroscope = imu[3:6].copy()

		q = self.quaternion

		accelerometer = norm_quaternion(accelerometer)

		f = np.array([
			2*(q[1]*q[3] - q[0]*q[2]) - accelerometer[0],
			2*(q[0]*q[1] + q[2]*q[3]) - accelerometer[1],
			2*(0.5 - q[1]**2 - q[2]**2) - accelerometer[2]
		])
		j = np.array([
			[-2*q[2], 2*q[3], -2*q[0], 2*q[1]],
			[2*q[1], 2*q[0], 2*q[3], 2*q[2]],
			[0, -4*q[1], -4*q[2], 0]
		])
		step = j.T.dot(f)
		step = norm_quaternion(step)  # normalise step magnitude

		# Compute rate of change of quaternion
		qdot = mul_quaternion(q,[0, gyroscope[0], gyroscope[1], gyroscope[2]]) * 0.5 - self.beta * step.T

		# Integrate to yield quaternion
		q += qdot * self.dt
		self.quaternion = norm_quaternion(q)  # normalise quaternion

		return self.quaternion

	# 9 axis
	def update_9imu(self, imu):
		accelerometer = imu[0:3].copy()
		gyroscope = imu[3:6].copy()
		magnetometer = imu[6:9].copy()

		q = self.quaternion

		accelerometer = norm_quaternion(accelerometer)
		magnetometer = norm_quaternion(magnetometer)

		h1 = mul_quaternion(q, [0, magnetometer[0], magnetometer[1], magnetometer[2]])
		h = mul_quaternion(h1, conj_quaternion(q))
		b = np.array([0, norm(h[1:3]), 0, h[3]])

		f = np.array([
			2*(q[1]*q[3] - q[0]*q[2]) - accelerometer[0],
			2*(q[0]*q[1] + q[2]*q[3]) - accelerometer[1],
			2*(0.5 - q[1]**2 - q[2]**2) - accelerometer[2],
			2*b[1]*(0.5 - q[2]**2 - q[3]**2) + 2*b[3]*(q[1]*q[3] - q[0]*q[2]) - magnetometer[0],
			2*b[1]*(q[1]*q[2] - q[0]*q[3]) + 2*b[3]*(q[0]*q[1] + q[2]*q[3]) - magnetometer[1],
			2*b[1]*(q[0]*q[2] + q[1]*q[3]) + 2*b[3]*(0.5 - q[1]**2 - q[2]**2) - magnetometer[2]
		])
		j = np.array([
			[-2*q[2],                  2*q[3],                  -2*q[0],                  2*q[1]],
			[2*q[1],                   2*q[0],                  2*q[3],                   2*q[2]],
			[0,                        -4*q[1],                 -4*q[2],                  0],
			[-2*b[3]*q[2],             2*b[3]*q[3],             -4*b[1]*q[2]-2*b[3]*q[0], -4*b[1]*q[3]+2*b[3]*q[1]],
			[-2*b[1]*q[3]+2*b[3]*q[1], 2*b[1]*q[2]+2*b[3]*q[0], 2*b[1]*q[1]+2*b[3]*q[3],  -2*b[1]*q[0]+2*b[3]*q[2]],
			[2*b[1]*q[2],              2*b[1]*q[3]-4*b[3]*q[1], 2*b[1]*q[0]-4*b[3]*q[2],  2*b[1]*q[1]]
		])
		step = j.T.dot(f)
		step = norm_quaternion(step)  # normalise step magnitude

		# Compute rate of change of quaternion
		qdot = mul_quaternion(q,[0, gyroscope[0], gyroscope[1], gyroscope[2]]) * 0.5 - self.beta * step.T

		# Integrate to yield quaternion
		q += qdot * self.dt
		self.quaternion = norm_quaternion(q)  # normalise quaternion

		return self.quaternion



if __name__ == '__main__':
	imu = excel_file_read("./data/imu_roll_pitch_yaw_200hz.xlsx")

	madgwick = Madgwick()

	rpy_list = []
	imu_plot = []

	figure_cnt = 0
	figure_freq = 10
	for one_imu in imu:
		q = madgwick.update_9imu(one_imu)
		rpy = quaternion2euler(q)
		rpy_list.append(rpy)

		r_mat = euler_rotation_mat(rpy)
		r_mat_q = quaternion_rotation_mat(q)

		plot_xyz_q = r_mat_q @ np.identity(3)

		figure_cnt += 1
		if figure_cnt % figure_freq == 0:
			imu_plot.append(plot_xyz_q.T)

	rpy_list = np.array(rpy_list) * 180 / 3.1416
	plt.plot(rpy_list[:, 0], 'r-')
	plt.plot(rpy_list[:, 1], 'b-')
	plt.plot(rpy_list[:, 2], 'g-')
	plt.grid()
	plt.show()

	imu_plot = np.array(imu_plot)
	plot_3d(imu_plot)