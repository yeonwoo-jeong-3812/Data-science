# -------------------- main_6dof.py (최종 수정 버전) --------------------
import numpy as np
from scipy.integrate import solve_ivp
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import config  # config.py 사용

# 헬퍼 함수: 쿼터니언 -> 오일러 각 변환
def quaternion_to_euler(q):
    """쿼터니언을 오일러 각(롤, 피치, 요)으로 변환 (단위: 도)"""
    q0, q1, q2, q3 = q
    
    # 롤 (x-축 회전)
    sinr_cosp = 2 * (q0 * q1 + q2 * q3)
    cosr_cosp = 1 - 2 * (q1**2 + q2**2)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # 피치 (y-축 회전)
    sinp = 2 * (q0 * q2 - q3 * q1)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp) # 90도 근처에서 짐벌락 방지
    else:
        pitch = math.asin(sinp)

    # 요 (z-축 회전)
    siny_cosp = 2 * (q0 * q3 + q1 * q2)
    cosy_cosp = 1 - 2 * (q2**2 + q3**2)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

class MissileSimulation6DoF:
    def __init__(self, missile_type="SCUD-B"):
            """6DoF 시뮬레이션 클래스 생성자"""
            print(f"🚀 6DoF Missile Simulation Initialized for '{missile_type}'")
            # config.py에서 미사일 정보를 가져옵니다.
            self.missile_info = config.get_enhanced_missile_info(missile_type)
            if not self.missile_info:
                raise ValueError(f"'{missile_type}'은(는) 유효한 미사일 타입이 아닙니다.")
                
            # 클래스 속성으로 시뮬레이션 파라미터 저장
            self.m0 = self.missile_info["launch_weight"]
            self.propellant_mass = self.missile_info["propellant_mass"]
            self.burn_time = self.missile_info["burn_time"]
            self.vertical_time = self.missile_info["vertical_time"]
            self.pitch_time = self.missile_info["pitch_time"]
            self.pitch_angle_deg_cmd = self.missile_info["pitch_angle_deg"]
            
            # 관성 텐서 (Ix, Iy, Iz)
            self.I = np.array(self.missile_info.get("inertia_tensor", np.diag([50000, 100000, 100000])))
            
            self.results = None # 시뮬레이션 결과 저장

    def event_ground_impact(self, t, state):
        """지면 충돌 이벤트 함수"""
        # t > 1초 이후에만 이벤트 감지 (발사 직후 종료 방지)
        if t < 1:
            return 1
        return state[2] # z 좌표(고도)가 0이 되는 시점
    event_ground_impact.terminal = True
    event_ground_impact.direction = -1

    def initialize_simulation(self, launch_angle_deg=45, azimuth_deg=90):
        """시뮬레이션 초기 상태 벡터 생성"""
        
        # ✨✨✨ [수정] 초기 고도를 0.1m로 설정하여 즉시 종료 방지 ✨✨✨
        pos_i = np.array([0.0, 0.0, 0.1]) # 초기 위치 (관성좌표계)
        
        vel_b = np.array([1.0, 0.0, 0.0]) # 초기 속도 (동체좌표계)
        
        # 오일러 각 -> 쿼터니언 변환
        el = -math.radians(launch_angle_deg)
        az = math.radians(azimuth_deg)
        
        cy = math.cos(az * 0.5); sy = math.sin(az * 0.5)
        cp = math.cos(el * 0.5); sp = math.sin(el * 0.5)
        cr = 1.0; sr = 0.0 # 롤 각은 0으로 가정
        
        q0 = cr * cp * cy + sr * sp * sy
        q1 = sr * cp * cy - cr * sp * sy
        q2 = cr * sp * cy + sr * cp * sy
        q3 = cr * cp * sy - sr * sp * cy
        att_q = np.array([q0, q1, q2, q3]) # 초기 자세 쿼터니언
        
        ang_vel_b = np.array([0.0, 0.0, 0.0]) # 초기 각속도 (동체좌표계)
        
        initial_state = np.concatenate((pos_i, vel_b, att_q, ang_vel_b))
        print(f"✅ Initial 6DoF state vector created (Launch Angle: {launch_angle_deg} deg, Azimuth: {azimuth_deg} deg).")
        return initial_state

    def quaternion_to_dcm(self, q):
        """쿼터니언을 방향 코사인 행렬(DCM)으로 변환 (동체 -> 관성)"""
        q0, q1, q2, q3 = q
        norm = np.linalg.norm(q)
        if norm > 1e-9:
            q = q / norm
        
        # [수정] 올바른 부호로 회전 행렬 공식 수정
        dcm = np.array([
            [1 - 2 * (q2**2 + q3**2),   2 * (q1*q2 - q0*q3),   2 * (q1*q3 + q0*q2)],
            [2 * (q1*q2 + q0*q3),     1 - 2 * (q1**2 + q3**2),   2 * (q2*q3 - q0*q1)],
            [2 * (q1*q3 - q0*q2),     2 * (q2*q3 + q0*q1),     1 - 2 * (q1**2 + q2**2)]
        ])
        return dcm


    def _get_common_forces_and_moments(self, t, state):
        """모든 비행 단계에 공통적으로 적용되는 물리량 계산"""
        pos_i = state[0:3]
        vel_b = state[3:6]
        att_q = state[6:10]
        
        # 현재 질량 계산
        mass_flow_rate = self.propellant_mass / self.burn_time
        current_mass = self.m0 - mass_flow_rate * t if t < self.burn_time else self.m0 - self.propellant_mass

        # 대기 환경
        altitude = pos_i[2]
        if altitude < 0: altitude = 0 # 고도가 음수가 되지 않도록 방지
        rho = config.PhysicsUtils.atmospheric_density(altitude)
        sound_speed = config.PhysicsUtils.sound_speed(altitude)
        V = np.linalg.norm(vel_b)
        mach = V / sound_speed if sound_speed > 1e-6 else 0

        # ▼▼▼ [수정] 회전 행렬의 역할을 명확히 구분 ▼▼▼
        # dcm_i_to_b: 관성 좌표계 -> 동체 좌표계 변환 행렬
        dcm_i_to_b = self.quaternion_to_dcm(att_q).T 
        
        # 중력 계산 (관성 좌표계에서 계산 후 동체 좌표계로 변환)
        g = config.PhysicsUtils.gravity_at_altitude(altitude)
        Fg_i = np.array([0, 0, -current_mass * g])
        Fg_b = dcm_i_to_b @ Fg_i

        # 추력
        Thrust_b = np.array([self.missile_info["thrust_profile"](t), 0, 0]) if t < self.burn_time else np.array([0, 0, 0])
        
        # 공력 계산
        u, v, w = vel_b
        alpha = math.atan2(w, u) if abs(u) > 1e-6 else 0
        beta = math.asin(v / V) if V > 1e-6 else 0
        
        Cd, Cl, Cm, _, _ = config.PhysicsUtils.get_aerodynamic_coefficients(self.missile_info, mach, alpha, beta)
        q_dynamic = 0.5 * rho * V**2
        S = self.missile_info["reference_area"]
        d = self.missile_info["diameter"]

        Drag = q_dynamic * S * Cd
        Lift = q_dynamic * S * Cl
        Fa_b = np.array([-Drag, 0, -Lift])
        
        pitch_moment = q_dynamic * S * d * Cm
        Ma_b = np.array([0, pitch_moment, 0])

        return Fg_b, Thrust_b, Fa_b, Ma_b, current_mass

    def dynamics_solver(self, t, state, F_total_b, M_total_b, current_mass):
        """상태 미분 방정식을 푸는 공통 솔버"""
        vel_b = state[3:6]
        att_q = state[6:10]
        ang_vel_b = state[10:13]

        # 운동방정식 풀이
        # 1. 병진 운동 (동체 좌표계 기준 가속도)
        vel_dot = (F_total_b / current_mass) - np.cross(ang_vel_b, vel_b)
        
        # 2. 회전 운동 (동체 좌표계 기준 각가속도)
        ang_vel_dot = np.linalg.inv(self.I) @ (M_total_b - np.cross(ang_vel_b, self.I @ ang_vel_b))

        # 기구학적 미분값 계산
        # ▼▼▼ [수정] 올바른 회전 행렬(동체->관성) 사용 ▼▼▼
        # dcm_b_to_i: 동체 좌표계 -> 관성 좌표계 변환 행렬
        dcm_b_to_i = self.quaternion_to_dcm(att_q)
        pos_dot = dcm_b_to_i @ vel_b
        
        # 2. 자세 변화 (쿼터니언)
        p, q, r = ang_vel_b
        omega_matrix = 0.5 * np.array([
            [0, -p, -q, -r],
            [p,  0,  r, -q],
            [q, -r,  0,  p],
            [r,  q, -p,  0]
        ])
        quat_dot = omega_matrix @ att_q
        
        return np.concatenate((pos_dot, vel_dot, quat_dot, ang_vel_dot))

    def dynamics_phased(self, t, state):
        """비행 단계에 따라 다른 제어 로직을 적용하는 통합 동역학 함수"""
        Fg_b, Thrust_b, Fa_b, Ma_b, current_mass = self._get_common_forces_and_moments(t, state)

        # 제어 모멘트 초기화
        Mc_b = np.array([0.0, 0.0, 0.0])

        # 비행 단계별 제어 로직
        if t <= self.vertical_time:
            # 1. 수직 상승 단계: 제어 없음
            pass
        elif t <= self.vertical_time + self.pitch_time:
            # 2. 피치 기동 단계: 목표 피치 각속도에 도달하도록 제어
            target_pitch_rate = math.radians(self.pitch_angle_deg_cmd) / self.pitch_time
            current_pitch_rate = state[11] # q (pitch rate)
            error = target_pitch_rate - current_pitch_rate
            
            # P 제어기 (게인 값은 튜닝 필요)
            Kp = 10000 
            Mc_b[1] = Kp * error
        else:
            # 3. 탄도 비행 단계: 제어 없음
            pass

        # 힘과 모멘트 합산
        F_total_b = Fg_b + Fa_b + Thrust_b
        M_total_b = Ma_b + Mc_b
        
        return self.dynamics_solver(t, state, F_total_b, M_total_b, current_mass)

    def run_simulation(self, launch_angle_deg=45, azimuth_deg=90, sim_time=500):
        """단일 6DoF 시뮬레이션을 실행하고 결과 반환"""
        initial_state = self.initialize_simulation(launch_angle_deg, azimuth_deg)
        sol = solve_ivp(
            self.dynamics_phased, 
            [0, sim_time], 
            initial_state, 
            method='RK45', 
            dense_output=True, 
            events=self.event_ground_impact,
            max_step=0.1
        )
        print("✅ 6DoF simulation finished.")
        self.results = sol
        return sol

    def run_simulation_realtime(self, launch_angle_deg=45, azimuth_deg=90, sim_time=500):
        """흔들림 없는 실시간 3D 시각화와 함께 6DoF 시뮬레이션을 실행"""
        print("\n--- 1. Running full simulation to get trajectory data ---")
        results = self.run_simulation(launch_angle_deg, azimuth_deg, sim_time)
        
        if not results.success or len(results.t) < 2:
            print("❌ Simulation failed to generate enough data for animation.")
            return

        print("\n--- 2. Starting Stable 3D Realtime Visualization ---")
        plt.ion()
        fig = plt.figure("Realtime 3D Trajectory", figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        time = results.t
        pos_e = results.y[1]
        pos_n = results.y[0]
        altitude = results.y[2]
        vel_b_u, vel_b_v, vel_b_w = results.y[3], results.y[4], results.y[5]
        total_velocity = np.sqrt(vel_b_u**2 + vel_b_v**2 + vel_b_w**2)

        # --- ✨ 그래프 흔들림 방지 ✨ ---
        # 전체 궤적을 기반으로 축 범위 고정
        max_range = max(np.max(np.abs(pos_e)), np.max(np.abs(pos_n)), np.max(altitude)) * 1.1
        if max_range < 1: max_range = 1 # 0이 되는 것 방지
        
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(0, max_range)
        # --------------------------------

        for i in range(0, len(time), 5): # 5 프레임씩 건너뛰며 부드럽게
            ax.clear()
            
            # 고정된 축 범위 재설정
            ax.set_xlim(-max_range, max_range)
            ax.set_ylim(-max_range, max_range)
            ax.set_zlim(0, max_range)

            # 궤적 그리기
            ax.plot(pos_e[:i+1], pos_n[:i+1], altitude[:i+1], 'b-')
            ax.plot([pos_e[i]], [pos_n[i]], [altitude[i]], 'ro')

            ax.set_xlabel("East Position (m)")
            ax.set_ylabel("North Position (m)")
            ax.set_zlabel("Altitude (m)")
            title = f'Missile Trajectory 3D Realtime Visualization\n'
            title += f'Time: {time[i]:.1f} s, Velocity: {total_velocity[i]:.1f} m/s, Altitude: {altitude[i]/1000:.2f} km'
            ax.set_title(title)
            
            plt.pause(0.001)

        print("\n--- 3. Animation finished ---")
        plt.ioff()
        plt.show(block=True)

    def plot_detailed_results(self):
        """사용자 요청 기반의 6DoF 시뮬레이션 상세 결과 시각화"""
        results = self.results
        if not results or not results.success or len(results.t) < 2:
            print("❌ 플롯할 시뮬레이션 데이터가 충분하지 않습니다.")
            return

        print("📊 Plotting Detailed 6DoF simulation results...")
        
        time = results.t

        # 데이터 추출 및 변환
        # [수정] 고도 좌표계를 현재 시뮬레이션에 맞게 양수(z)로 변경
        pos_n, pos_e, altitude = results.y[0], results.y[1], results.y[2]
        vel_b_u, vel_b_v, vel_b_w = results.y[3], results.y[4], results.y[5]
        total_velocity = np.sqrt(vel_b_u**2 + vel_b_v**2 + vel_b_w**2)
        
        quaternions = results.y[6:10]
        roll, pitch, yaw = [], [], []
        for i in range(len(time)):
            r, p, y = quaternion_to_euler(quaternions[:, i])
            roll.append(r); pitch.append(p); yaw.append(y)
        
        # 질량 변화 계산
        initial_mass = self.missile_info['launch_weight']
        final_mass = initial_mass - self.missile_info['propellant_mass']
        burn_time = self.missile_info['burn_time']
        mass = np.piecewise(time, [time < burn_time, time >= burn_time], 
                            [lambda t: initial_mass - (initial_mass - final_mass) * t / burn_time, final_mass])
        
        # 공력 데이터 재계산 (그래프용)
        angular_velocities = results.y[10:13]
        alphas, betas, aero_moments_M = [], [], []
        S = self.missile_info["reference_area"]
        d = self.missile_info["diameter"]

        for i in range(len(time)):
            V = total_velocity[i]
            alt = altitude[i]
            
            # 받음각, 옆미끄럼각
            alpha_rad = math.atan2(vel_b_w[i], vel_b_u[i]) if abs(vel_b_u[i]) > 1e-6 else 0
            beta_rad = math.asin(vel_b_v[i] / V) if V > 1e-6 else 0
            alphas.append(math.degrees(alpha_rad))
            betas.append(math.degrees(beta_rad))
            
            # 피칭 모멘트
            rho = config.PhysicsUtils.atmospheric_density(alt)
            mach = V / config.PhysicsUtils.sound_speed(alt)
            q_dynamic = 0.5 * rho * V**2
            _, _, Cm, _, _ = config.PhysicsUtils.get_aerodynamic_coefficients(self.missile_info, mach, alpha_rad, beta_rad)
            pitch_moment = q_dynamic * S * d * Cm
            aero_moments_M.append(pitch_moment)

        # 그래프 생성
        figures = {
            "Figure 1: Velocity & Attitude": [('Velocity (m/s)', time, total_velocity), ('Pitch Angle (deg)', time, pitch), ('Yaw Angle (deg)', time, yaw)],
            "Figure 2: Position & Mass": [('North Position (m)', time, pos_n), ('East Position (m)', time, pos_e), ('Altitude (m)', time, altitude), ('Mass (kg)', time, mass)],
            "Figure 4: 6DoF Core Dynamics": [
                ('Angular Velocity (deg/s)', [time, time, time], [np.degrees(angular_velocities[0, :]), np.degrees(angular_velocities[1, :]), np.degrees(angular_velocities[2, :])], ['p (Roll rate)', 'q (Pitch rate)', 'r (Yaw rate)']),
                ('Flight Angles (deg)', [time, time], [alphas, betas], ['alpha (AoA)', 'beta (Sideslip)']),
                ('Aerodynamic Moment (Nm)', [time], [aero_moments_M], ['M (Pitch Moment)'])
            ]
        }

        for figname, subplots in figures.items():
            num_subplots = len(subplots)
            plt.figure(figname, figsize=(12, 4 * num_subplots))
            plt.suptitle(figname)
            for i, plot_data in enumerate(subplots, 1):
                plt.subplot(num_subplots, 1, i)
                ylabel, xdatas, ydatas, *labels_tuple = plot_data
                
                if isinstance(xdatas, list): # Multi-line plot
                    labels = labels_tuple[0]
                    for j in range(len(xdatas)):
                        plt.plot(xdatas[j], ydatas[j], label=labels[j])
                    plt.legend()
                else: # Single line plot
                    plt.plot(xdatas, ydatas)

                plt.ylabel(ylabel)
                if i == num_subplots: plt.xlabel("Time (s)")
                plt.grid(True)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 3D 궤적
        fig3d = plt.figure("Figure 3: 3D Trajectory", figsize=(10, 8))
        ax3d = fig3d.add_subplot(111, projection='3d')
        ax3d.plot(pos_e, pos_n, altitude)
        ax3d.set_xlabel("East Position (m)"); ax3d.set_ylabel("North Position (m)"); ax3d.set_zlabel("Altitude (m)")
        ax3d.set_title("Missile Trajectory 3D Visualization")
        
        print("✅ All plots generated. Displaying figures...")
        plt.show(block=True)

def main():
    """메인 함수: 사용자에게 실행 모드를 선택받음"""
    print("6DoF 미사일 궤적 시뮬레이션을 시작합니다...")
    
    sim6dof = MissileSimulation6DoF(missile_type="SCUD-B")
    
    print("\n실행 모드를 선택하세요:")
    print("1. 실시간 3D 궤적 시뮬레이션")
    print("2. 상세 결과 그래프")
    
    mode = input("모드 선택 (1-2, 기본값: 1): ")
    if mode not in ["1", "2"]:
        mode = "1"

    launch_angle = 45
    sim_time = 500

    if mode == "2":
        print("\n--- 상세 결과 그래프 모드 실행 ---")
        # 1. 시뮬레이션 실행
        sim6dof.run_simulation(launch_angle_deg=launch_angle, sim_time=sim_time)
        # 2. 클래스 내부의 상세 그래프 메서드 호출
        sim6dof.plot_detailed_results()

    else:
        print("\n--- 실시간 3D 궤적 시뮬레이션 모드 실행 ---")
        sim6dof.run_simulation_realtime(launch_angle_deg=launch_angle, sim_time=sim_time)

    print("\n미사일 궤적 시뮬레이션이 완료되었습니다.")


if __name__ == "__main__":
    main()