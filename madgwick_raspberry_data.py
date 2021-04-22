import numpy as np
from numpy.linalg import norm, inv
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks

from mathlib_v2 import *
from excel_read import excel_file_read
from plot3d import plot_3d
from madgwick import Madgwick

dt = 0.005

def excel_read(file_name):
	imu_dataframe = pd.read_excel(file_name, sheet_name="Sheet1")
	imu_dataframe = imu_dataframe.drop(columns=imu_dataframe.columns[0])
	
	return imu_dataframe


def main():
	imu_dataframe = excel_read("./data/2021-04-18 23_23_36.xlsx")

	### step detect ###
	acc_norm = imu_dataframe['acc_x']**2 + imu_dataframe['acc_y']**2 + imu_dataframe['acc_z']**2
	peaks, _ = find_peaks(acc_norm, height=1.5, distance=20)

	print(peaks)

	### walking frequency cal ###
	# peak diff #
	diff_peaks = np.zeros(peaks.shape[0] - 1)
	for idx in range(diff_peaks.shape[0]):
		diff_peaks[idx] = peaks[idx + 1] - peaks[idx]

	# peak diff avg #
	diff_peaks_avg = np.average(diff_peaks)

	# big diff delete #
	delete_idx = []
	for idx, one_diff_peak in enumerate(diff_peaks):
		if one_diff_peak > diff_peaks_avg * 2:
			delete_idx.append(idx)
	diff_peaks = np.delete(diff_peaks, delete_idx)

	freq_walk = np.average(diff_peaks)
	half_freq_walk = int(freq_walk / 2)

	### stationary ###
	stationary = np.ones(imu_dataframe.shape[0])
	for idx in range(len(delete_idx)+1):
		if idx == 0:
			stationary[peaks[0]-half_freq_walk:peaks[delete_idx[0]]+half_freq_walk] = 0

		elif idx == len(delete_idx):
			stationary[peaks[delete_idx[idx-1]+1]-half_freq_walk:peaks[-1]+half_freq_walk] = 0
		
		else:
			stationary[peaks[delete_idx[idx-1]+1]-half_freq_walk:peaks[delete_idx[idx]]+half_freq_walk] = 0

	mad = Madgwick()

	linear_acc = []
	ori = []

	imu_dataframe[['acc_x','acc_y','acc_z']] *= 9.80665
	imu_dataframe[['gyr_x','gyr_y','gyr_z']] /= 57.295779513
	for idx, temp_imu in imu_dataframe.iterrows():
	
		imu_6dof = temp_imu[1:7]
		imu_9dof = temp_imu[1:]

		quat = mad.update_6imu(imu_6dof)
		# quat = mad.update_9imu(imu_9dof)
		
		### rotation mat ###
		con_quat = conj_quaternion(quat)
		rot_mat = quaternion_rotation_mat(con_quat)
		
		### acc -> linear acc ###
		one_linear_acc = rot_mat @ temp_imu[:3]
		linear_acc.append(one_linear_acc)

		### roll, pitch, yaw ###
		rpy = quaternion2euler(quat)
		# print(rpy)
		ori.append(rpy)

	ori = np.array(ori)
	linear_acc = np.array(linear_acc)
	linear_acc[:,2] -= 9.80665


	### velocity cal ###
	vel = np.zeros((imu_dataframe.shape[0], 3))
	move = False
	move_cnt = 0
	for idx, one_linear_acc in enumerate(linear_acc):
		if idx == 0:
			continue

		## start move
		if stationary[idx] == 0 and stationary[idx-1] == 1:
			move = True
			move_cnt = 0

		## end move
		elif (stationary[idx] == 1 and stationary[idx-1] == 0)\
			or (idx == linear_acc.shape[0]-1 and move == True):
			move = False

			if move_cnt == 0:
				continue

			vel_drift = vel[idx-1] - vel[idx - move_cnt]
			vel_drift_rate = vel_drift / move_cnt
			vel[idx-move_cnt:idx, :] -= np.arange(1,move_cnt+1,1).reshape(move_cnt, 1) @ vel_drift_rate.reshape(1, 3)
			move_cnt = 0


		if move:
			# vel[idx,:] = vel[idx-1,:] + (linear_acc[idx] + linear_acc[idx-1]) * 0.02 / 2
			vel[idx,:] = vel[idx-1,:] + one_linear_acc * dt
			move_cnt += 1

	### position cal ###
	pos = np.zeros((imu_dataframe.shape[0], 3))
	for idx, one_vel in enumerate(vel):
		if idx == 0:
			pos[idx, :] = vel[idx, :] * dt
			continue

		# pos[idx, :] = pos[idx-1, :] + vel[idx, :] * 0.02
		pos[idx, :] = pos[idx-1, :] + (vel[idx, :] + vel[idx-1, :]) * dt / 2

	plt.subplot(311)
	plt.plot(imu_dataframe['acc_x'], 'r-')
	plt.plot(imu_dataframe['acc_y'], 'b-')
	plt.plot(imu_dataframe['acc_z'], 'g-')
	plt.grid()
	plt.subplot(312)
	plt.plot(vel[:, 0], 'r-')
	plt.plot(vel[:, 1], 'b-')
	plt.plot(vel[:, 2], 'g-')
	plt.grid()
	plt.subplot(313)
	plt.plot(pos[:, 0], 'r-')
	plt.plot(pos[:, 1], 'b-')
	plt.plot(pos[:, 2], 'g-')
	plt.grid()

	plt.figure()
	plt.plot(acc_norm, 'b-')
	plt.plot(peaks, acc_norm[peaks], "r^")
	plt.plot(stationary, 'g-')

	### pos 2D plot ###
	plt.figure()
	plt.plot(pos[:, 0], pos[:, 1], "k.")		
	

	# rpy_list = np.array(rpy_list) * 180 / 3.1416
	# plt.plot(rpy_list[:, 0], 'r-')
	# plt.plot(rpy_list[:, 1], 'b-')
	# plt.plot(rpy_list[:, 2], 'g-')
	# plt.grid()

	plt.show()

if __name__ == '__main__':
	main()