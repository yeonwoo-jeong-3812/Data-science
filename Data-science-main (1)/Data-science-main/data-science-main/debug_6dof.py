"""6DOF 디버깅 스크립트"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Data-science'))

import numpy as np
import matplotlib.pyplot as plt
from main_6dof import MissileSimulation6DoF, quaternion_to_euler

# 6DOF 시뮬레이션 실행
sim = MissileSimulation6DoF(missile_type="SCUD-B")
results = sim.run_simulation(launch_angle_deg=45, sim_time=500)

time = results.t
pos = results.y[0:3]
vel_b = results.y[3:6]
att_q = results.y[6:10]

# 고도
altitude = pos[2]

# 동체 좌표계 속도
vel_b_mag = np.sqrt(vel_b[0]**2 + vel_b[1]**2 + vel_b[2]**2)

# 관성 좌표계 속도 (위치 미분)
vel_i_x = np.gradient(pos[0], time)
vel_i_y = np.gradient(pos[1], time)
vel_i_z = np.gradient(pos[2], time)
vel_i_mag = np.sqrt(vel_i_x**2 + vel_i_y**2 + vel_i_z**2)

# 피치각
pitch_list = []
for i in range(len(time)):
    _, p, _ = quaternion_to_euler(att_q[:, i])
    pitch_list.append(p)
pitch = np.array(pitch_list)

# 특정 시점 출력
print("="*60)
print("6DOF 디버깅 정보")
print("="*60)
for t_check in [10, 30, 65, 100, 200]:
    idx = np.argmin(np.abs(time - t_check))
    print(f"\n시간 = {time[idx]:.1f}초:")
    print(f"  고도: {altitude[idx]:.1f} m")
    print(f"  피치각: {pitch[idx]:.1f}°")
    print(f"  동체 속도: u={vel_b[0][idx]:.1f}, v={vel_b[1][idx]:.1f}, w={vel_b[2][idx]:.1f} m/s")
    print(f"  동체 속도 크기: {vel_b_mag[idx]:.1f} m/s")
    print(f"  관성 속도: vx={vel_i_x[idx]:.1f}, vy={vel_i_y[idx]:.1f}, vz={vel_i_z[idx]:.1f} m/s")
    print(f"  관성 속도 크기: {vel_i_mag[idx]:.1f} m/s")

print("\n" + "="*60)

# 그래프
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# 속도 비교
axes[0].plot(time, vel_b_mag, 'r-', linewidth=2, label='동체 좌표계 속도')
axes[0].plot(time, vel_i_mag, 'b--', linewidth=2, label='관성 좌표계 속도')
axes[0].set_ylabel('Velocity (m/s)')
axes[0].set_title('6DOF 속도 비교')
axes[0].legend()
axes[0].grid(True)

# 고도
axes[1].plot(time, altitude/1000, 'g-', linewidth=2)
axes[1].set_ylabel('Altitude (km)')
axes[1].set_title('고도')
axes[1].grid(True)

# 피치각
axes[2].plot(time, pitch, 'orange', linewidth=2)
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Pitch Angle (deg)')
axes[2].set_title('피치각')
axes[2].grid(True)

plt.tight_layout()
plt.savefig('debug_6dof.png', dpi=300)
print("\n그래프 저장: debug_6dof.png")
plt.show()
