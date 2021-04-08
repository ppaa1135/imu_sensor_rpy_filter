import numpy as np
import math
import numbers
from math import sin, cos, tan, atan2
from numpy.linalg import norm, inv

RAD2DEG = 180.0 / math.pi
DEG2RAD = math.pi / 180.0

def quaternion(w, x, y, z):
	return np.array([w, x, y, z], dtype="float32")

def mul_quaternion(q1, q2):
	w = q1[0]*q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
	x = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
	y = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
	z = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]

	return np.array([w, x, y, z])

def norm_quaternion(quaternion):
	norm_q = norm(quaternion)
	if norm_q == 1 or norm_q == 0:
		return quaternion

	else:
		return quaternion / norm_q

def conj_quaternion(quaternion):
	w = quaternion[0]
	x = -quaternion[1]
	y = -quaternion[2]
	z = -quaternion[3]

	return np.array([w,x,y,z])

def quaternion2euler(quaternion):
	w = quaternion[0]
	x = quaternion[1]
	y = quaternion[2]
	z = quaternion[3]

	# if t1 > 1 or < -1, can't cal arcsin
	t1 = 2*(w * y - z * x)
	if t1 > 1.0:
		t1 = 1
	if t1 < -1.0:
		t1 = -1

	roll = np.arctan2(2*(w*x + y*z),
					  1-2*(x**2 + y**2))
	pitch = np.arcsin(t1)
	yaw = np.arctan2(2*(w*z + x*y),
					 1-2*(y**2 + z**2))

	# to avoid gimbal lock #
	# when pitch +90 -90 #
	################################
	if t1 > 0.98:
		roll = 0
		yaw = 2 * np.arctan2(x, w)

	if t1 < -0.98:
		roll = 0
		yaw = -2 * np.arctan2(x, w)
	################################

	return np.array([roll, pitch, yaw])

def euler2quaternion(euler):
	roll = euler[0]
	pitch = euler[1]
	yaw = euler[2]

	w = cos(roll/2)*cos(pitch/2)*cos(yaw/2) + sin(roll/2)*sin(pitch/2)*sin(yaw/2)
	x = sin(roll/2)*cos(pitch/2)*cos(yaw/2) - cos(roll/2)*sin(pitch/2)*sin(yaw/2)
	y = cos(roll/2)*sin(pitch/2)*cos(yaw/2) + sin(roll/2)*cos(pitch/2)*sin(yaw/2)
	z = cos(roll/2)*cos(pitch/2)*sin(yaw/2) - sin(roll/2)*sin(pitch/2)*cos(yaw/2)

	return np.array([w, x, y, z])

def euler_rotation_mat(euler):
	"""
	euler angle to rotation matrix
	##### be careful #####
	# not fixed cordinate #
	:param roll:
	:param pitch:
	:param yaw:
	:return:
	"""
	roll = euler[0]
	pitch = euler[1]
	yaw = euler[2]

	cos_phi = cos(roll)
	sin_phi = sin(roll)

	cos_theta = cos(pitch)
	sin_theta = sin(pitch)

	cos_psi = cos(yaw)
	sin_psi = sin(yaw)

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

	return R_mat

def quaternion_rotation_mat(quaternion):
	w = quaternion[0]
	x = quaternion[1]
	y = quaternion[2]
	z = quaternion[3]

	R_mat = np.array([[1-2*(y**2 + z**2), 2*(x*y - w*z), 2*(w*y + x*z)],
					  [2*(x*y + w*z), 1-2*(x**2 + z**2), 2*(y*z - w*x)],
					  [2*(x*z - w*y), 2*(w*x + y*z), 1-2*(x**2 + y**2)]])

	return R_mat

def deg2rad(angle):
	return angle*DEG2RAD

def rad2deg(angle):
	return angle*RAD2DEG

if __name__ == '__main__':

	### gimbal lock test ###
	roll = deg2rad(0)
	pitch = deg2rad(90)
	yaw = deg2rad(0)

	e = np.array([roll, pitch, yaw])
	print("e :", e)
	q1 = euler2quaternion(e)
	print("q1 :", q1)
	e1 = quaternion2euler(q1)
	print("e1 :", e1)
	q2 = euler2quaternion(e1)
	print("q2 :",q2)
