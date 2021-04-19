import numpy as np
from numpy.linalg import norm, inv
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks

from mathlib_v2 import *
from excel_read import excel_file_read
from plot3d import plot_3d
from madgwick import Madgwick


def excel_read(file_name):
	imu_dataframe = pd.read_excel(file_name, sheet_name="Sheet1")
	imu_dataframe = imu_dataframe.drop(columns=imu_dataframe.columns[0])
	
	return imu_dataframe


def main():	
	imu_dataframe[['acc_x','acc_y','acc_z']] *= 9.80665
	imu_dataframe[['gyr_x','gyr_y','gyr_z']] /= 57.295779513
	madgwick = Madgwick()

	rpy_list = []
	imu_plot = []

	figure_cnt = 0
	figure_freq = 10
	for idx, temp_imu in imu_dataframe.iterrows():
		# imu_6dof = temp_imu[1:7]
		imu_9dof = temp_imu[1:]

		# q = madgwick.update_6imu(imu_6dof)
		q = madgwick.update_9imu(imu_9dof)

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