import numpy as np
import math

# 테스트: 발사각 45도, 방위각 90도
launch_angle_deg = 45
azimuth_deg = 90

gamma = math.radians(launch_angle_deg)
psi = math.radians(azimuth_deg)

# 3DOF 방향 벡터 (관성 좌표계)
dir_east = math.cos(gamma) * math.sin(psi)
dir_north = math.cos(gamma) * math.cos(psi)
dir_up = math.sin(gamma)

print(f"3DOF Direction Vector:")
print(f"  East (X): {dir_east:.4f}")
print(f"  North (Y): {dir_north:.4f}")
print(f"  Up (Z): {dir_up:.4f}")

# NEW DCM 생성 방법 (vstack)
x_body_i = np.array([dir_east, dir_north, dir_up])
x_body_i = x_body_i / np.linalg.norm(x_body_i)
print(f"\nBody X-axis (flight direction): {x_body_i}")

up_vec = np.array([0, 0, 1])
y_body_i = np.cross(up_vec, x_body_i)
if np.linalg.norm(y_body_i) < 1e-6:
    y_body_i = np.array([0, 1, 0])
y_body_i = y_body_i / np.linalg.norm(y_body_i)
print(f"Body Y-axis: {y_body_i}")

z_body_i = np.cross(x_body_i, y_body_i)
z_body_i = z_body_i / np.linalg.norm(z_body_i)
print(f"Body Z-axis: {z_body_i}")

# DCM (동체 -> 관성): 각 행이 동체 축을 관성 좌표계로 표현
dcm_b_to_i = np.vstack([x_body_i, y_body_i, z_body_i])
print(f"\nDCM (body -> inertial):\n{dcm_b_to_i}")

# DCM (관성 -> 동체)
dcm_i_to_b = dcm_b_to_i.T
print(f"\nDCM (inertial -> body):\n{dcm_i_to_b}")

# 초기 속도 (동체 X축 방향 1 m/s)
vel_b = np.array([1.0, 0.0, 0.0])
print(f"\nBody frame velocity: {vel_b}")

# 관성 좌표계 속도
vel_i = dcm_b_to_i @ vel_b
print(f"Inertial frame velocity: {vel_i}")
print(f"  East (X): {vel_i[0]:.4f}")
print(f"  North (Y): {vel_i[1]:.4f}")
print(f"  Up (Z): {vel_i[2]:.4f}")

print(f"\nExpected:")
print(f"  East (X): {dir_east:.4f}")
print(f"  North (Y): {dir_north:.4f}")
print(f"  Up (Z): {dir_up:.4f}")

if np.allclose(vel_i, x_body_i):
    print("\nOK: Body X-axis = flight direction")
else:
    print("\nERROR: Body X-axis != flight direction")
