import sys
import os
sys.path.insert(0, 'Data-science')
import numpy as np
from main_6dof import MissileSimulation6DoF

sim = MissileSimulation6DoF("SCUD-B")
results = sim.run_simulation(45, 90, 500)

time = results.t
altitude = results.y[2]

print(f"시뮬레이션 시간: 0 ~ {time[-1]:.2f}초")
print(f"데이터 포인트: {len(time)}개")
print(f"\n고도 변화:")
for i in [0, 10, 30, 50, 65, 100, -1]:
    if i == -1:
        idx = -1
    else:
        idx = np.argmin(np.abs(time - i))
    if idx < len(time):
        print(f"  t={time[idx]:.1f}초: 고도={altitude[idx]:.1f}m")

print(f"\n최소 고도: {np.min(altitude):.1f}m (t={time[np.argmin(altitude)]:.1f}초)")
print(f"최대 고도: {np.max(altitude):.1f}m (t={time[np.argmax(altitude)]:.1f}초)")
