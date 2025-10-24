# -------------------- main_6dof_path.py (진짜 최종 완성 버전) --------------------
import numpy as np
from scipy.integrate import solve_ivp
import math
import matplotlib
import config_6 as cfg # config_6 -> config로 수정
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from config import ENHANCED_MISSILE_TYPES, PhysicsUtils

# 헬퍼 함수: 쿼터니언 -> 오일러 각 변환
def quaternion_to_euler(q):
    q0, q1, q2, q3 = q
    roll = math.atan2(2*(q0*q1 + q2*q3), 1 - 2*(q1**2 + q2**2))
    sinp_clipped = np.clip(2*(q0*q2 - q3*q1), -1.0, 1.0)
    pitch = math.asin(sinp_clipped)
    yaw = math.atan2(2*(q0*q3 + q1*q2), 1 - 2*(q2**2 + q3**2))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

class MissileSimulation6DoF_Path:
    def __init__(self, missile_type="SCUD-B"):
        print(f"🚀 Initializing 6DoF Simulation for '{missile_type}'...")
        self.missile_data = cfg.get_enhanced_missile_info(missile_type)
        
        # 물리 데이터 불러오기
        self.m = self.missile_data["launch_weight"]
        self.propellant_mass = self.missile_data["propellant_mass"]
        self.burn_time = self.missile_data["burn_time"]
        self.thrust_profile = self.missile_data["thrust_profile"]
        self.reference_area = self.missile_data["reference_area"]
        self.diameter = self.missile_data["diameter"]
        self.Ix = self.missile_data.get("Ix", 300)
        self.Iy = self.missile_data.get("Iy", 20000)
        self.Iz = self.missile_data.get("Iz", 20000)
        self.vertical_time = self.missile_data["vertical_time"]
        self.pitch_time = self.missile_data["pitch_time"]
        self.target_pitch_end_deg = self.missile_data["pitch_angle_deg"]
        
        # 6DoF 시뮬레이션에 필요한 추가 변수 초기화
        self.J_body = self.missile_data.get("inertia_tensor")
        self.l_cm = self.missile_data.get("center_of_mass")[0]
        self.missile_length = self.missile_data.get("length")

        # 객체 생성 시점에 초기 상태를 자동으로 정의
        self.initialize_simulation()

    def event_ground_6dof(self, t, state):
        """지구 중심 좌표계에서 지면 충돌을 감지하는 이벤트 함수"""
        if t < 1e-6:
            return 1.0
        position_magnitude = np.linalg.norm(state[0:3])
        return position_magnitude - cfg.R_EARTH
    event_ground_6dof.terminal = True
    event_ground_6dof.direction = -1

    def initialize_simulation(self, launch_angle_deg=45, azimuth_deg=90):
        pos_i = np.array([0.0, 0.0, cfg.R_EARTH])
        vel_b = np.array([5.0, 0.0, 0.0])
        el = math.radians(launch_angle_deg)
        self.initial_launch_angle_rad = el
        az = math.radians(azimuth_deg)
        cy = math.cos(az * 0.5)
        sy = math.sin(az * 0.5)
        cp = math.cos(el * 0.5)
        sp = math.sin(el * 0.5)
        cr = math.cos(0 * 0.5)
        sr = math.sin(0 * 0.5)
        q0 = cr * cp * cy + sr * sp * sy
        q1 = sr * cp * cy - cr * sp * sy
        q2 = cr * sp * cy + sr * cp * sy
        q3 = cr * cp * sy - sr * sp * cy
        att_q = np.array([q0, q1, q2, q3])
        ang_vel_b = np.array([0.0, 0.0, 0.0])
        
        self.initial_state = np.concatenate((pos_i, vel_b, att_q, ang_vel_b, [self.m]))
        
        print(f"✅ Initial state vector created (Launch Angle: {launch_angle_deg} deg, Azimuth: {azimuth_deg} deg).")
        return self.initial_state

    def quaternion_to_rotation_matrix(self, q_vec):
        q0, q1, q2, q3 = q_vec
        R_bi = np.array([
            [q0**2 + q1**2 - q2**2 - q3**2, 2*(q1*q2 + q0*q3), 2*(q1*q3 - q0*q2)],
            [2*(q1*q2 - q0*q3), q0**2 - q1**2 + q2**2 - q3**2, 2*(q2*q3 + q0*q1)],
            [2*(q1*q3 + q0*q2), 2*(q2*q3 - q0*q1), q0**2 - q1**2 - q2**2 + q3**2]
        ])
        return R_bi

    def dynamics(self, t, state):
        x, y, z, vx, vy, vz, q0, q1, q2, q3, p, q, r, m = state
        q_vec = np.array([q0, q1, q2, q3])
        V_inertial = np.array([vx, vy, vz])
        omega_body = np.array([p, q, r])
        
        q_norm = np.linalg.norm(q_vec)
        if q_norm < 1e-6: q_vec = np.array([1, 0, 0, 0])
        else: q_vec /= q_norm

        R_bi = self.quaternion_to_rotation_matrix(q_vec)
        R_ib = R_bi.T
        
        r_inertial = np.array([x, y, z])
        Fg_inertial = - (cfg.GM_EARTH * m / np.linalg.norm(r_inertial)**3) * r_inertial
        Fg_body = R_bi @ Fg_inertial
        
        if t <= self.missile_data["burn_time"]:
            thrust_magnitude = self.missile_data["thrust_profile"](t)
        else:
            thrust_magnitude = 0.0
        Ft_body = np.array([thrust_magnitude, 0, 0])
        
        V_rel_inertial = V_inertial
        V_rel_body = R_bi @ V_rel_inertial
        
        if np.linalg.norm(V_rel_body) < 1e-6:
            alpha = 0
            beta = 0
            mach = 0
        else:
            alpha = np.arctan2(V_rel_body[2], V_rel_body[0])
            beta = np.arctan2(V_rel_body[1], V_rel_body[0])
            mach = np.linalg.norm(V_rel_body) / cfg.PhysicsUtils.sound_speed(np.linalg.norm(r_inertial) - cfg.R_EARTH)
        
        rho = cfg.PhysicsUtils.atmospheric_density(np.linalg.norm(r_inertial) - cfg.R_EARTH)
        q_dyn = 0.5 * rho * np.linalg.norm(V_rel_body)**2
        
        Cd, Cl, Cm, Cn, Cl_roll = cfg.PhysicsUtils.get_aerodynamic_coefficients(
            self.missile_data, mach, alpha, beta
        )
        
        F_aero_body = np.array([
            -q_dyn * self.missile_data["reference_area"] * Cd,
            q_dyn * self.missile_data["reference_area"] * Cl * math.sin(beta),
            -q_dyn * self.missile_data["reference_area"] * Cl * math.sin(alpha)
        ])
        
        l_cp = self.missile_data.get("center_of_pressure", 0.6) * self.missile_length
        moment_arm = l_cp - self.l_cm
        
        M_aero_body = np.array([
            q_dyn * self.missile_data["reference_area"] * self.missile_length * Cl_roll,
            q_dyn * self.missile_data["reference_area"] * self.missile_length * Cm,
            q_dyn * self.missile_data["reference_area"] * self.missile_length * Cn
        ])
        
        M_control_body = np.array([0.0, 0.0, 0.0])
        pitch_phase_end_time = self.missile_data["vertical_time"] + self.missile_data["pitch_time"]
        if t > self.missile_data["vertical_time"] and t < pitch_phase_end_time:
            _, _, current_pitch_deg = quaternion_to_euler(q_vec)
            target_pitch_rate = (self.missile_data["pitch_angle_deg"] * cfg.DEG_TO_RAD) / self.missile_data["pitch_time"]
            M_control_body[1] = self.J_body[1, 1] * (target_pitch_rate - q) * 100
        
        M_total_body = M_aero_body + M_control_body
        
        F_total_body = Fg_body + Ft_body + F_aero_body
        F_total_inertial = R_ib @ F_total_body
        
        dxdt, dydt, dzdt = V_inertial
        dvxdt, dvydt, dvzdt = F_total_inertial / m
        
        q_mat = np.array([
            [q0, -q1, -q2, -q3],
            [q1, q0, -q3, q2],
            [q2, q3, q0, -q1],
            [q3, -q2, q1, q0]
        ])
        d_q_vec = 0.5 * q_mat @ np.concatenate(([0], omega_body))
        dq0dt, dq1dt, dq2dt, dq3dt = d_q_vec
        
        J_w = self.J_body @ omega_body
        omega_cross_J_w = np.cross(omega_body, J_w)
        d_omega_body = np.linalg.inv(self.J_body) @ (M_total_body - omega_cross_J_w)
        dpdt, dqdt_ang, drdt = d_omega_body
        
        dmdt = -self.missile_data.get("mass_flow_rate", lambda t: 0)(t) if t <= self.missile_data["burn_time"] else 0.0
        
        return [dxdt, dydt, dzdt, dvxdt, dvydt, dvzdt, dq0dt, dq1dt, dq2dt, dq3dt, dpdt, dqdt_ang, drdt, dmdt]
    
    def run_simulation(self, sim_time=2000):
        print(f"🚀 6DoF 시뮬레이션 시작: {sim_time}초 동안 통합 계산")
        
        sol = solve_ivp(
            fun=self.dynamics,
            t_span=[0, sim_time],
            y0=self.initial_state,
            events=[self.event_ground_6dof],
            dense_output=True,
            method='RK45',
            rtol=1e-6,
            atol=1e-9
        )
        
        if sol.success:
            print(f"✅ 시뮬레이션 성공: 총 비행 시간 {sol.t[-1]:.2f}초")
            
            t_dense = np.arange(0, sol.t[-1], 0.1)
            if len(t_dense) == 0 and sol.t.size > 0:
                t_dense = np.array([sol.t[0]])
            states_dense = sol.sol(t_dense)
            
            positions = states_dense[0:3].T
            velocities = states_dense[3:6].T
            quaternions = states_dense[6:10].T
            angular_velocities = states_dense[10:13].T
            mass = states_dense[13].T
            
            results = {
                'time': t_dense,
                'positions': positions,
                'velocities': velocities,
                'quaternions': quaternions,
                'angular_velocities': angular_velocities,
                'mass': mass
            }
            
            final_pos_vec = results['positions'][-1, :]
            initial_pos_vec = results['positions'][0, :]
            final_range = np.linalg.norm(final_pos_vec[0:2] - initial_pos_vec[0:2])
            altitudes = np.linalg.norm(results['positions'], axis=1) - cfg.R_EARTH
            max_altitude = np.max(altitudes)
            
            print(f"   최대 고도: {max_altitude/1000:.2f} km")
            print(f"   최종 사거리: {final_range/1000:.2f} km")
            
            self.results = results
            return results
        else:
            print(f"❌ 시뮬레이션 실패: {sol.message}")
            return None
    
    def run_simulation_realtime_6dof(self, sim_time=500):
        print("\n--- 1. Running full simulation to get trajectory data ---")
        results = self.run_simulation(sim_time)
        
        if not results or len(results['time']) < 2:
            print("❌ Simulation failed to generate enough data for animation.")
            return

        print("\n--- 2. Starting 3D Realtime Visualization ---")
        plt.ion()
        fig = plt.figure("Realtime 3D Trajectory", figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        time = results['time']
        positions = results['positions']
        velocities = results['velocities']

        for i in range(0, len(time), 5):
            ax.clear()
            
            # 💡 수정: ECI 좌표계에 맞춰 고도 계산
            altitudes = np.linalg.norm(positions[:i+1], axis=1) - cfg.R_EARTH
            
            ax.plot(positions[:i+1, 0], positions[:i+1, 1], altitudes, 'b-')
            ax.plot([positions[i, 0]], [positions[i, 1]], [altitudes[i]], 'ro')
            
            ax.set_xlabel("X Position (m)")
            ax.set_ylabel("Y Position (m)")
            ax.set_zlabel("Altitude (m)")
            
            total_velocity = np.linalg.norm(velocities[i])
            altitude = np.linalg.norm(positions[i]) - cfg.R_EARTH
            title = (f'Time: {time[i]:.1f} s, Vel: {total_velocity:.1f} m/s, '
                     f'Alt: {altitude/1000:.2f} km')
            ax.set_title(title)
            
            ax.autoscale(enable=True, axis='both', tight=True)
            plt.pause(0.001)

        print("\n--- 3. Animation finished ---")
        plt.ioff()
        plt.show(block=True)

def main():
    missile_to_simulate = "SCUD-B"
    print(f"Path-Following 6DoF 미사일 궤적 시뮬레이션을 시작합니다 (Missile: {missile_to_simulate})...")
    sim6dof = MissileSimulation6DoF_Path(missile_type=missile_to_simulate)
    
    print("\n실행 모드를 선택하세요:")
    print("1. 실시간 3D 궤적 시뮬레이션")
    print("2. 상세 결과 그래프 (미구현)")
    mode = input("모드 선택 (1-2, 기본값: 1): ")

    launch_angle = 45  
    sim_time = 1000

    if mode == "2":
        print("\n--- 상세 결과 그래프 모드는 현재 비활성화되어 있습니다. ---")
    else:
        print("\n--- 실시간 3D 궤적 시뮬레이션 모드 실행 ---")
        sim6dof.run_simulation_realtime_6dof(sim_time=sim_time)

    print("\n미사일 궤적 시뮬레이션이 완료되었습니다.")

if __name__ == "__main__":
    main()