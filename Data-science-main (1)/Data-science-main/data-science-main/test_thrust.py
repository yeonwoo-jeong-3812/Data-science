"""추력 계산 비교 테스트"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'professor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Data-science'))

import professor.config as cfg_3dof
import config as cfg_6dof

# SCUD-B 파라미터
missile_3dof = cfg_3dof.MISSILE_TYPES["SCUD-B"]
missile_6dof = cfg_6dof.get_enhanced_missile_info("SCUD-B")

print("="*60)
print("3DOF vs 6DOF 추력 계산 비교")
print("="*60)

# 공통 파라미터
t = 10.0  # 10초
h = 1000.0  # 1km 고도

# 3DOF 계산
propellant_mass_3dof = missile_3dof["propellant_mass"]
burn_time_3dof = missile_3dof["burn_time"]
isp_sea_3dof = missile_3dof["isp_sea"]
g_3dof = cfg_3dof.G * cfg_3dof.R**2 / (cfg_3dof.R + h)**2
T_3dof = isp_sea_3dof * (propellant_mass_3dof / burn_time_3dof) * g_3dof

print(f"\n[3DOF]")
print(f"  propellant_mass: {propellant_mass_3dof} kg")
print(f"  burn_time: {burn_time_3dof} s")
print(f"  isp_sea: {isp_sea_3dof}")
print(f"  g (at {h}m): {g_3dof:.6f} m/s²")
print(f"  mass_flow_rate: {propellant_mass_3dof / burn_time_3dof:.4f} kg/s")
print(f"  추력: {T_3dof:.2f} N")

# 6DOF 계산
propellant_mass_6dof = missile_6dof["propellant_mass"]
burn_time_6dof = missile_6dof["burn_time"]
isp_sea_6dof = missile_6dof["isp_sea"]
g_6dof = cfg_6dof.PhysicsUtils.gravity_at_altitude(h)
mass_flow_rate_6dof = propellant_mass_6dof / burn_time_6dof
T_6dof = isp_sea_6dof * mass_flow_rate_6dof * g_6dof

print(f"\n[6DOF]")
print(f"  propellant_mass: {propellant_mass_6dof} kg")
print(f"  burn_time: {burn_time_6dof} s")
print(f"  isp_sea: {isp_sea_6dof}")
print(f"  g (at {h}m): {g_6dof:.6f} m/s²")
print(f"  mass_flow_rate: {mass_flow_rate_6dof:.4f} kg/s")
print(f"  추력: {T_6dof:.2f} N")

print(f"\n[비교]")
print(f"  추력 차이: {abs(T_3dof - T_6dof):.2f} N ({abs(T_3dof - T_6dof)/T_3dof*100:.2f}%)")
print(f"  추력 비율 (6DOF/3DOF): {T_6dof/T_3dof:.4f}")
print("="*60)
