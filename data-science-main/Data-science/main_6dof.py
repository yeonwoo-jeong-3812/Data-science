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
            self.pitch_angle_deg_cmd = 35  # 더 공격적인 피치 기동
            
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
        
        # 초기 고도를 0.1m로 설정하여 즉시 종료 방지
        pos_i = np.array([0.0, 0.0, 0.1]) # 초기 위치 (관성좌표계)
        
        # 초기 속도를 10.0 m/s로 설정 (수치 안정성)
        vel_b = np.array([10.0, 0.0, 0.0]) # 초기 속도 (동체좌표계)
        
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
        
        # 현재 질량 계산 - 수정 (연소 종료 후 일정)
        if t < self.burn_time:
            mass_flow_rate = self.propellant_mass / self.burn_time
            current_mass = self.m0 - mass_flow_rate * t
        else:
            # 연소 종료 후 구조 질량만 남음
            mass_flow_rate = 0.0
            current_mass = self.m0 - self.propellant_mass
        
        # 최소 질량 제한 (안전장치)
        min_mass = self.m0 - self.propellant_mass
        current_mass = max(current_mass, min_mass)

        # 대기 환경
        altitude = pos_i[2]
        if altitude < 0: altitude = 0 # 고도가 음수가 되지 않도록 방지
        rho = config.PhysicsUtils.atmospheric_density(altitude)
        sound_speed = config.PhysicsUtils.sound_speed(altitude)
        V = np.linalg.norm(vel_b)
        mach = V / sound_speed if sound_speed > 1e-6 else 0

        # 회전 행렬: 관성 좌표계 -> 동체 좌표계 변환
        dcm_i_to_b = self.quaternion_to_dcm(att_q).T 
        
        # 중력 계산 (관성 좌표계에서 계산 후 동체 좌표계로 변환)
        g = config.PhysicsUtils.gravity_at_altitude(altitude)
        Fg_i = np.array([0, 0, -current_mass * g])
        Fg_b = dcm_i_to_b @ Fg_i

        # ========== ✅ 추력 계산 수정 (교수님 자료 기반) ========== #
        if t < self.burn_time:
            # 고도에 따른 비추력 보간 (해수면 -> 진공)
            isp_sea = self.missile_info["isp_sea"]
            isp_vac = self.missile_info["isp_vac"]
            # 100km 이상에서 진공 비추력 사용
            if altitude >= 100000:
                isp_current = isp_vac
            else:
                # 선형 보간
                isp_current = isp_sea + (isp_vac - isp_sea) * (altitude / 100000)
            
            # 추력 = ISP * 연료소모율 * g (올바른 공식)
            thrust_magnitude = isp_current * mass_flow_rate * g
            Thrust_b = np.array([thrust_magnitude, 0, 0])
        else:
            Thrust_b = np.array([0, 0, 0])
        
        # 공력 계산
        u, v, w = vel_b
        alpha = math.atan2(w, u) if abs(u) > 1e-6 else 0
        beta = math.asin(v / V) if V > 1e-6 else 0
        
        # 각속도 추출 (댐핑 계산용)
        ang_vel_b = state[10:13]
        p, q, r = ang_vel_b
        
        # 현재 속도를 missile_info에 저장 (댐핑 계산에 필요)
        self.missile_info["current_velocity"] = V
        
        # 공력 계수 계산 (댐핑 포함)
        Cd, Cl, Cm, Cn, Cl_roll = config.PhysicsUtils.get_aerodynamic_coefficients(
            self.missile_info, mach, alpha, beta, q_pitch_rate=q, r_yaw_rate=r
        )
        
        # 동압 및 공력
        q_dynamic = 0.5 * rho * V**2
        S = self.missile_info["reference_area"]
        d = self.missile_info["diameter"]

        # 항력과 양력 (동체 좌표계)
        Drag = q_dynamic * S * Cd
        Lift = q_dynamic * S * Cl
        # 항력은 속도 반대 방향, 양력은 수직 방향
        Fa_b = np.array([-Drag, 0, -Lift])
        
        # 공력 모멘트 (댐핑 포함)
        pitch_moment = q_dynamic * S * d * Cm
        yaw_moment = q_dynamic * S * d * Cn
        Ma_b = np.array([0, pitch_moment, yaw_moment])

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
        """비행 단계에 따라 다른 제어 로직을 적용하는 통합 동역학 함수 (4단계)"""
        Fg_b, Thrust_b, Fa_b, Ma_b, current_mass = self._get_common_forces_and_moments(t, state)

        # 제어 모멘트 초기화
        Mc_b = np.array([0.0, 0.0, 0.0])

        # 비행 단계별 제어 로직 (교수님 자료와 동일한 4단계)
        if t <= self.vertical_time:
            # ========== 1. 수직 상승 단계 ========== #
            pass
            
        elif t <= self.vertical_time + self.pitch_time:
            # ========== 2. 피치 기동 단계: PD 제어기 ========== #
            
            # 목표 피치 각속도 계산
            target_pitch_rate = math.radians(self.pitch_angle_deg_cmd) / self.pitch_time
            
            # 현재 피치 각속도
            current_pitch_rate = state[11]  # q (pitch rate)
            
            # 오차 계산
            error = target_pitch_rate - current_pitch_rate
            
            # PD 제어기
            Kp = 800  # 500 → 800으로 증가 (더 빠른 응답)
            Kd = 400  # 300 → 400으로 증가
            
            # 이전 오차 저장 (첫 실행 시 초기화)
            if not hasattr(self, 'prev_pitch_error'):
                self.prev_pitch_error = 0.0
                self.prev_time = t
            
            # 오차 변화율 (미분항)
            dt = t - self.prev_time
            if dt > 1e-6:
                error_rate = (error - self.prev_pitch_error) / dt
            else:
                error_rate = 0.0
            
            # PD 제어 출력
            Mc_b[1] = Kp * error + Kd * error_rate
            
            # 제어 모멘트 제한
            max_control_moment = 40000  # N·m
            Mc_b[1] = np.clip(Mc_b[1], -max_control_moment, max_control_moment)
            
            # 상태 업데이트
            self.prev_pitch_error = error
            self.prev_time = t
            
        elif t <= self.burn_time:
            # ========== 3. 등자세 선회 단계 (새로 추가!) ========== #
            # 피치각을 일정하게 유지하면서 추력으로 가속
            # 이 단계가 없어서 미사일이 제대로 가속되지 않았음!
            
            # 목표 피치각 (피치 기동 완료 후의 각도)
            target_pitch_deg = 90 - self.pitch_angle_deg_cmd  # 90 - 35 = 55도
            target_pitch_rad = math.radians(target_pitch_deg)
            
            # 현재 피치각 (쿼터니언 → 오일러각)
            roll, pitch, yaw = quaternion_to_euler(state[6:10])
            current_pitch_rad = math.radians(pitch)
            
            # 오차 계산
            error = target_pitch_rad - current_pitch_rad
            
            # P 제어 (자세 유지용, 낮은 게인)
            Kp = 300  # 유지만 하면 되므로 낮은 게인
            Mc_b[1] = Kp * error
            
            # 제어 모멘트 제한
            max_control_moment = 20000  # N·m (유지용이므로 작은 값)
            Mc_b[1] = np.clip(Mc_b[1], -max_control_moment, max_control_moment)
            
            # 이전 오차 초기화 (다음 단계를 위해)
            if hasattr(self, 'prev_pitch_error'):
                del self.prev_pitch_error
                del self.prev_time
            
        else:
            # ========== 4. 탄도 비행 단계 ========== #
            # 추력 없음, 제어 없음
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
    
    def plot_results_6dof_clean(self):
        """
        ✨ main_fixed 스타일의 3x4 그리드 레이아웃
        True 6DOF 물리 모델 + 깔끔한 시각화
        """
        if self.results is None:
            print("❌ 시뮬레이션 결과가 없습니다.")
            return
        
        # 결과 데이터 추출
        time = self.results.t
        pos_n, pos_e, altitude = self.results.y[0], self.results.y[1], self.results.y[2]
        vel_b_u, vel_b_v, vel_b_w = self.results.y[3], self.results.y[4], self.results.y[5]
        total_velocity = np.sqrt(vel_b_u**2 + vel_b_v**2 + vel_b_w**2)
        
        # 쿼터니언 -> 오일러각 변환
        att_q = self.results.y[6:10]
        roll_list, pitch_list, yaw_list = [], [], []
        for i in range(len(time)):
            q = att_q[:, i]
            roll, pitch, yaw = quaternion_to_euler(q)
            roll_list.append(roll)
            pitch_list.append(pitch)
            yaw_list.append(yaw)
        
        roll = np.array(roll_list)
        pitch = np.array(pitch_list)
        yaw = np.array(yaw_list)
        
        # 각속도
        ang_vel = self.results.y[10:13]
        p_rate = np.rad2deg(ang_vel[0])  # 롤 각속도
        q_rate = np.rad2deg(ang_vel[1])  # 피치 각속도
        r_rate = np.rad2deg(ang_vel[2])  # 요 각속도
        
        # 비행거리 계산
        range_km = np.sqrt(pos_n**2 + pos_e**2) / 1000
        
        # ========== 3x4 그리드 플롯 생성 ========== #
        fig = plt.figure(figsize=(24, 15))
        plt.rcParams.update({'font.size': 9})
        
        # 1. 3D 궤적
        ax1 = fig.add_subplot(3, 4, 1, projection='3d')
        ax1.plot(pos_n/1000, pos_e/1000, altitude/1000, 'b-', linewidth=2)
        ax1.scatter([0], [0], [0], c='g', marker='o', s=100, label='Launch')
        ax1.scatter([pos_n[-1]/1000], [pos_e[-1]/1000], [altitude[-1]/1000], 
                    c='r', marker='x', s=100, label='Impact')
        ax1.set_xlabel('North (km)', fontsize=9, labelpad=8)
        ax1.set_ylabel('East (km)', fontsize=9, labelpad=8)
        ax1.set_zlabel('Altitude (km)', fontsize=9, labelpad=8)
        ax1.set_title('3D Trajectory', fontsize=10, pad=10, fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. 속도
        ax2 = fig.add_subplot(3, 4, 2)
        ax2.plot(time, total_velocity, 'b-', linewidth=2)
        ax2.set_xlabel('Time (s)', fontsize=9)
        ax2.set_ylabel('Velocity (m/s)', fontsize=9)
        ax2.set_title('Total Velocity', fontsize=10, pad=10, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=8)
        
        # 3. 고도
        ax3 = fig.add_subplot(3, 4, 3)
        ax3.plot(time, altitude/1000, 'b-', linewidth=2)
        ax3.set_xlabel('Time (s)', fontsize=9)
        ax3.set_ylabel('Altitude (km)', fontsize=9)
        ax3.set_title('Altitude', fontsize=10, pad=10, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(labelsize=8)
        
        # 4. 비행거리 vs 고도
        ax4 = fig.add_subplot(3, 4, 4)
        ax4.plot(range_km, altitude/1000, 'b-', linewidth=2)
        ax4.set_xlabel('Range (km)', fontsize=9)
        ax4.set_ylabel('Altitude (km)', fontsize=9)
        ax4.set_title('Range vs Altitude', fontsize=10, pad=10, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(labelsize=8)
        
        # ========== 오일러 각도 (Roll, Pitch, Yaw) ========== #
        
        # 5. 롤각 (Roll)
        ax5 = fig.add_subplot(3, 4, 5)
        ax5.plot(time, roll, 'r-', linewidth=2)
        ax5.set_xlabel('Time (s)', fontsize=9)
        ax5.set_ylabel('Roll Angle (deg)', fontsize=9)
        ax5.set_title('Roll Angle (φ)', fontsize=10, pad=10, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.tick_params(labelsize=8)
        
        # 6. 피치각 (Pitch)
        ax6 = fig.add_subplot(3, 4, 6)
        ax6.plot(time, pitch, 'g-', linewidth=2)
        ax6.axhline(y=45, color='k', linestyle='--', alpha=0.5, label='Target (45°)')
        ax6.set_xlabel('Time (s)', fontsize=9)
        ax6.set_ylabel('Pitch Angle (deg)', fontsize=9)
        ax6.set_title('Pitch Angle (θ)', fontsize=10, pad=10, fontweight='bold')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)
        ax6.tick_params(labelsize=8)
        
        # 7. 요각 (Yaw)
        ax7 = fig.add_subplot(3, 4, 7)
        ax7.plot(time, yaw, 'b-', linewidth=2)
        ax7.set_xlabel('Time (s)', fontsize=9)
        ax7.set_ylabel('Yaw Angle (deg)', fontsize=9)
        ax7.set_title('Yaw Angle (ψ)', fontsize=10, pad=10, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        ax7.tick_params(labelsize=8)
        
        # ========== 각속도 (Angular Rates) ========== #
        
        # 8. 롤 각속도 (p)
        ax8 = fig.add_subplot(3, 4, 8)
        ax8.plot(time, p_rate, 'r-', linewidth=2)
        ax8.set_xlabel('Time (s)', fontsize=9)
        ax8.set_ylabel('Roll Rate (deg/s)', fontsize=9)
        ax8.set_title('Roll Rate (p)', fontsize=10, pad=10, fontweight='bold')
        ax8.grid(True, alpha=0.3)
        ax8.tick_params(labelsize=8)
        
        # 9. 피치 각속도 (q)
        ax9 = fig.add_subplot(3, 4, 9)
        ax9.plot(time, q_rate, 'g-', linewidth=2)
        ax9.set_xlabel('Time (s)', fontsize=9)
        ax9.set_ylabel('Pitch Rate (deg/s)', fontsize=9)
        ax9.set_title('Pitch Rate (q)', fontsize=10, pad=10, fontweight='bold')
        ax9.grid(True, alpha=0.3)
        ax9.tick_params(labelsize=8)
        
        # 10. 요 각속도 (r)
        ax10 = fig.add_subplot(3, 4, 10)
        ax10.plot(time, r_rate, 'b-', linewidth=2)
        ax10.set_xlabel('Time (s)', fontsize=9)
        ax10.set_ylabel('Yaw Rate (deg/s)', fontsize=9)
        ax10.set_title('Yaw Rate (r)', fontsize=10, pad=10, fontweight='bold')
        ax10.grid(True, alpha=0.3)
        ax10.tick_params(labelsize=8)
        
        # ========== 추가 정보 ========== #
        
        # 11. 속도 성분 (동체 좌표계)
        ax11 = fig.add_subplot(3, 4, 11)
        ax11.plot(time, vel_b_u, 'r-', linewidth=1.5, label='u (forward)')
        ax11.plot(time, vel_b_v, 'g-', linewidth=1.5, label='v (side)')
        ax11.plot(time, vel_b_w, 'b-', linewidth=1.5, label='w (up)')
        ax11.set_xlabel('Time (s)', fontsize=9)
        ax11.set_ylabel('Velocity (m/s)', fontsize=9)
        ax11.set_title('Velocity Components (Body Frame)', fontsize=10, pad=10, fontweight='bold')
        ax11.legend(fontsize=8)
        ax11.grid(True, alpha=0.3)
        ax11.tick_params(labelsize=8)
        
        # 12. 위치 (평면도)
        ax12 = fig.add_subplot(3, 4, 12)
        ax12.plot(pos_e/1000, pos_n/1000, 'b-', linewidth=2)
        ax12.scatter([0], [0], c='g', marker='o', s=100, label='Launch')
        ax12.scatter([pos_e[-1]/1000], [pos_n[-1]/1000], 
                     c='r', marker='x', s=100, label='Impact')
        ax12.set_xlabel('East (km)', fontsize=9)
        ax12.set_ylabel('North (km)', fontsize=9)
        ax12.set_title('Ground Track', fontsize=10, pad=10, fontweight='bold')
        ax12.legend(fontsize=8)
        ax12.grid(True, alpha=0.3)
        ax12.tick_params(labelsize=8)
        ax12.axis('equal')
        
        plt.tight_layout(pad=4.0, h_pad=3.5, w_pad=3.5)
        
        # ========== 결과 요약 출력 ========== #
        final_range = range_km[-1]
        max_altitude = np.max(altitude) / 1000
        flight_time = time[-1]
        final_velocity = total_velocity[-1]
        
        print("\n" + "="*60)
        print("6DOF 시뮬레이션 결과 요약")
        print("="*60)
        print(f"최종 사거리: {final_range:.2f} km")
        print(f"최대 고도: {max_altitude:.2f} km")
        print(f"비행 시간: {flight_time:.2f} s")
        print(f"최종 속도: {final_velocity:.2f} m/s")
        print(f"최종 롤각: {roll[-1]:.2f}°")
        print(f"최종 피치각: {pitch[-1]:.2f}°")
        print(f"최종 요각: {yaw[-1]:.2f}°")
        print("="*60)
        
        # 저장
        import os, datetime
        os.makedirs("results_6dof", exist_ok=True)
        now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f"results_6dof/6dof_clean_results_{now_str}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ 그래프 저장: {save_path}")
        
        plt.show()
        return save_path

    def run_simulation_realtime(self, launch_angle_deg=45, azimuth_deg=90, sim_time=500):
        """여러 서브플롯으로 나눠진 실시간 시각화 (모드 2와 동일한 궤도)"""
        print("\n--- 1. Running full simulation to get trajectory data ---")
        results = self.run_simulation(launch_angle_deg, azimuth_deg, sim_time)
        
        if not results.success or len(results.t) < 2:
            print("❌ Simulation failed to generate enough data for animation.")
            return

        print("\n--- 2. Starting Multi-Panel Realtime Visualization ---")
        
        # 데이터 추출 (모드 2와 동일하게)
        time = results.t
        pos_n, pos_e, altitude = results.y[0], results.y[1], results.y[2]
        vel_b_u, vel_b_v, vel_b_w = results.y[3], results.y[4], results.y[5]
        total_velocity = np.sqrt(vel_b_u**2 + vel_b_v**2 + vel_b_w**2)
        
        # 자세 각도 계산
        quaternions = results.y[6:10]
        pitch_list = []
        for i in range(len(time)):
            _, p, _ = quaternion_to_euler(quaternions[:, i])
            pitch_list.append(p)
        pitch = np.array(pitch_list)

        # 좌표를 시작점을 (0,0,0)으로 이동 (시각성 향상)
        pos_e_rel = pos_e - pos_e[0]
        pos_n_rel = pos_n - pos_n[0]
        altitude_rel = altitude - altitude[0]

        # 축 범위 설정: 각 축을 [0, max]로 설정하여 한 구석에서 출발하는 뷰
        max_e = max(np.max(pos_e_rel), 1) * 1.1
        max_n = max(np.max(pos_n_rel), 1) * 1.1
        max_alt = max(np.max(altitude_rel), 1) * 1.1

        # Figure 생성 (2x2 레이아웃)
        plt.ion()
        fig = plt.figure("Realtime Multi-Panel Visualization", figsize=(14, 10))
        
        # 서브플롯 생성
        ax1 = plt.subplot(2, 2, 1, projection='3d')  # 3D 궤적
        ax2 = plt.subplot(2, 2, 2)  # 속도
        ax3 = plt.subplot(2, 2, 3)  # 고도
        ax4 = plt.subplot(2, 2, 4)  # 피치각
        
        # 애니메이션 루프
        for i in range(0, len(time), 5):
            # 모든 서브플롯 클리어
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()
            
            # --- 1. 3D 궤적 (모드 2의 Figure 3와 동일) ---
            ax1.plot(pos_e_rel[:i+1], pos_n_rel[:i+1], altitude_rel[:i+1], 'b-', linewidth=2)
            ax1.plot([pos_e_rel[i]], [pos_n_rel[i]], [altitude_rel[i]], 'ro', markersize=8)
            ax1.set_xlim(0, max_e)
            ax1.set_ylim(0, max_n)
            ax1.set_zlim(0, max_alt)
            ax1.set_xlabel("East Position (m)")
            ax1.set_ylabel("North Position (m)")
            ax1.set_zlabel("Altitude (m)")
            ax1.set_title("3D Trajectory")
            
            # --- 2. 속도 그래프 ---
            ax2.plot(time[:i+1], total_velocity[:i+1], 'g-', linewidth=2)
            ax2.plot([time[i]], [total_velocity[i]], 'ro', markersize=8)
            ax2.set_xlim(0, time[-1])
            ax2.set_ylim(0, max(total_velocity) * 1.1)
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Velocity (m/s)")
            ax2.set_title(f"Velocity: {total_velocity[i]:.1f} m/s")
            ax2.grid(True)
            
            # --- 3. 고도 그래프 ---
            ax3.plot(time[:i+1], altitude_rel[:i+1]/1000, 'b-', linewidth=2)
            ax3.plot([time[i]], [altitude_rel[i]/1000], 'ro', markersize=8)
            ax3.set_xlim(0, time[-1])
            ax3.set_ylim(0, max(altitude_rel)/1000 * 1.1)
            ax3.set_xlabel("Time (s)")
            ax3.set_ylabel("Altitude (km)")
            ax3.set_title(f"Altitude: {altitude_rel[i]/1000:.2f} km")
            ax3.grid(True)
            
            # --- 4. 피치각 그래프 ---
            ax4.plot(time[:i+1], pitch[:i+1], 'r-', linewidth=2)
            ax4.plot([time[i]], [pitch[i]], 'ro', markersize=8)
            ax4.set_xlim(0, time[-1])
            ax4.set_ylim(min(pitch) * 1.1, max(pitch) * 1.1)
            ax4.set_xlabel("Time (s)")
            ax4.set_ylabel("Pitch Angle (deg)")
            ax4.set_title(f"Pitch: {pitch[i]:.1f}°")
            ax4.grid(True)
            
            # 전체 타이틀
            fig.suptitle(f'Missile 6DoF Simulation - Time: {time[i]:.1f} s', fontsize=14, fontweight='bold')
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
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
        # 3. 클래스 내부의 깔끔한 결과 그래프 메서드 호출
        sim6dof.plot_results_6dof_clean()

    else:
        print("\n--- 실시간 3D 궤적 시뮬레이션 모드 실행 ---")
        sim6dof.run_simulation_realtime(launch_angle_deg=launch_angle, sim_time=sim_time)

    print("\n미사일 궤적 시뮬레이션이 완료되었습니다.")


if __name__ == "__main__":
    main()