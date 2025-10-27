import numpy as np
import math

# 테스트: 발사각 45도, 방위각 90도
launch_angle_deg = 45
azimuth_deg = 90

pitch = math.radians(launch_angle_deg)
yaw = math.radians(azimuth_deg)
roll = 0.0

cy = math.cos(yaw * 0.5)
sy = math.sin(yaw * 0.5)
cp = math.cos(pitch * 0.5)
sp = math.sin(pitch * 0.5)
cr = math.cos(roll * 0.5)
sr = math.sin(roll * 0.5)

q0 = cr * cp * cy + sr * sp * sy
q1 = sr * cp * cy - cr * sp * sy
q2 = cr * sp * cy + sr * cp * sy
q3 = cr * cp * sy - sr * sp * cy
att_q = np.array([q0, q1, q2, q3])
att_q = att_q / np.linalg.norm(att_q)

print(f"Quaternion: {att_q}")
print(f"Norm: {np.linalg.norm(att_q)}")

# DCM 계산
q0, q1, q2, q3 = att_q
dcm = np.array([
    [1 - 2 * (q2**2 + q3**2),   2 * (q1*q2 - q0*q3),   2 * (q1*q3 + q0*q2)],
    [2 * (q1*q2 + q0*q3),     1 - 2 * (q1**2 + q3**2),   2 * (q2*q3 - q0*q1)],
    [2 * (q1*q3 - q0*q2),     2 * (q2*q3 + q0*q1),     1 - 2 * (q1**2 + q2**2)]
])

print(f"\nDCM:\n{dcm}")

# 초기 속도 (동체 좌표계): 전진 방향 1 m/s
vel_b = np.array([1.0, 0.0, 0.0])

# 관성 좌표계 속도
vel_i = dcm @ vel_b
print(f"\nvel_b (body): {vel_b}")
print(f"vel_i (inertial): {vel_i}")
print(f"  X (East): {vel_i[0]:.4f}")
print(f"  Y (North): {vel_i[1]:.4f}")
print(f"  Z (Up): {vel_i[2]:.4f}")

# 예상: 45도 위로, 90도 동쪽 방향
# 구형 좌표계: 피치 45도, 요 90도
# X (East) = cos(45) * sin(90) = 0.707
# Y (North) = cos(45) * cos(90) = 0
# Z (Up) = sin(45) = 0.707
print(f"\n예상 (구형 좌표계):")
print(f"  X (East): {math.cos(math.radians(45)) * math.sin(math.radians(90)):.4f}")
print(f"  Y (North): {math.cos(math.radians(45)) * math.cos(math.radians(90)):.4f}")
print(f"  Z (Up): {math.sin(math.radians(45)):.4f}")
