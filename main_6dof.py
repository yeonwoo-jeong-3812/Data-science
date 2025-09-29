# -------------------- main_6dof.py (들여쓰기 수정 최종 버전) --------------------
import numpy as np
from scipy.integrate import solve_ivp
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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
        print("🚀 6DoF Missile Simulation Initialized")
        self.m = 5860
        self.propellant_mass = 4875
        self.burn_time = 65
        self.Ix = 300
        self.Iy = 20000
        self.Iz = 20000
        self.pitch_time = 15
        self.pitch_angle_deg_cmd = 20

    def event_ground_6dof(self, t, state):
        """6DoF 지면 충돌 이벤트 함수"""
        return state[2]
    event_ground_6dof.terminal = True
    event_ground_6dof.direction = 1

    def initialize_simulation(self, launch_angle_deg=45, azimuth_deg=90):
        """시뮬레이션 초기 상태 벡터 생성"""
        pos_i = np.array([0.0, 0.0, 0.0])
        vel_b = np.array([5.0, 0.0, 0.0])
        
        el = math.radians(launch_angle_deg)
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
        
        self.initial_state = np.concatenate((pos_i, vel_b, att_q, ang_vel_b))
        print(f"✅ Initial 6DoF state vector created (Launch Angle: {launch_angle_deg} deg, Azimuth: {azimuth_deg} deg).")
        return self.initial_state

    def quaternion_to_dcm(self, q):
        """쿼터니언을 방향 코사인 행렬(DCM)으로 변환"""
        q0, q1, q2, q3 = q
        dcm = np.array([
            [q0**2 + q1**2 - q2**2 - q3**2, 2 * (q1*q2 + q0*q3), 2 * (q1*q3 - q0*q2)],
            [2 * (q1*q2 - q0*q3), q0**2 - q1**2 + q2**2 - q3**2, 2 * (q2*q3 + q0*q1)],
            [2 * (q1*q3 + q0*q2), 2 * (q2*q3 - q0*q1), q0**2 - q1**2 - q2**2 + q3**2]
        ])
        return dcm

    # main_6dof.py의 dynamics_6dof 함수를 이걸로 교체하세요.

    def dynamics_6dof(self, t, state):
        """6DoF 동역학 계산 함수 (쿼터니언 정규화 추가)"""
        # 1. 상태 변수 할당
        pos_i = state[0:3]
        vel_b = state[3:6]
        att_q = state[6:10]
        
        # --- 🚨 FIX: 쿼터니언 정규화(Normalization) ---
        # 수치 오차 누적으로 인해 쿼터니언의 크기가 1에서 벗어나는 것을 방지합니다.
        q_norm = np.linalg.norm(att_q)
        if q_norm > 1e-6:
            att_q = att_q / q_norm
        # -----------------------------------------
        
        ang_vel_b = state[10:13]
        
        u, v, w = vel_b
        p, q, r = ang_vel_b
        V = np.linalg.norm(vel_b)
        
        current_mass = self.m - (self.propellant_mass * min(t, self.burn_time) / self.burn_time)
        
        if V < 1e-6:
            alpha = 0.0
            beta = 0.0
        else:
            alpha = math.atan2(w, u)
            beta = math.asin(v / V)
        
        # 2. 힘과 모멘트 계산
        g = 9.80665
        Fg_i = np.array([0, 0, current_mass * g])
        dcm_i_to_b = self.quaternion_to_dcm(att_q)
        Fg_b = dcm_i_to_b @ Fg_i
        
        if t < self.burn_time:
            Thrust_b = np.array([130000, 0, 0])
        else:
            Thrust_b = np.array([0, 0, 0])
            
        rho = 1.225 * np.exp(pos_i[2] / 8500)
        q_dynamic = 0.5 * rho * V**2
        S = 0.6
        d = 0.88
        CD_0 = 0.2
        CL_alpha = 2.5
        
        # 3. PD 제어기 및 비행 프로그램
        if t < self.pitch_time:
            alpha_cmd = math.radians(self.pitch_angle_deg_cmd) * (t / self.pitch_time)
        else:
            alpha_cmd = 0.0
        
        alpha_error = alpha - alpha_cmd
        
        Cm_alpha = -1.5
        Cm_q = -15.0
        
        Lift = q_dynamic * S * (CL_alpha * alpha)
        Drag = q_dynamic * S * CD_0
        Fa_b = np.array([-Drag, 0, -Lift])
        
        pitch_moment = q_dynamic * S * d * ((Cm_alpha * alpha_error) + (Cm_q * (q * d / (2 * V + 1e-6))))
        Ma_b = np.array([0, pitch_moment, 0])
        
        # 4. 힘과 모멘트 합산
        F_total_b = Fg_b + Fa_b + Thrust_b
        M_total_b = Ma_b
        
        # 5. 운동방정식 풀이
        u_dot = (F_total_b[0] / current_mass) - q*w + r*v
        v_dot = (F_total_b[1] / current_mass) - r*u + p*w
        w_dot = (F_total_b[2] / current_mass) - p*v + q*u
        
        p_dot = (M_total_b[0] - (self.Iz - self.Iy) * q * r) / self.Ix
        q_dot = (M_total_b[1] - (self.Ix - self.Iz) * r * p) / self.Iy
        r_dot = (M_total_b[2] - (self.Iy - self.Ix) * p * q) / self.Iz
        
        # 6. 기구학적 미분값 계산
        q0, q1, q2, q3 = att_q
        q_dot_matrix = 0.5 * np.array([
            [-q1, -q2, -q3],
            [q0, -q3, q2],
            [q3, q0, -q1],
            [-q2, q1, q0]
        ])
        quat_dot = q_dot_matrix @ ang_vel_b
        
        dcm_b_to_i = dcm_i_to_b.T
        vel_i = dcm_b_to_i @ vel_b
        pos_dot = vel_i
        
        # 7. 미분값 반환
        derivatives = np.concatenate((pos_dot, [u_dot, v_dot, w_dot], quat_dot, [p_dot, q_dot, r_dot]))
        return derivatives

    def run_simulation_6dof(self, launch_angle_deg=45, azimuth_deg=90, sim_time=500):
        """단일 6DoF 시뮬레이션을 실행하고 결과를 반환"""
        initial_state = self.initialize_simulation(launch_angle_deg, azimuth_deg)
        sol = solve_ivp(
            self.dynamics_6dof, 
            [0, sim_time], 
            initial_state, 
            method='RK45', 
            dense_output=True, 
            events=self.event_ground_6dof,
            max_step=0.1
        )
        print("✅ 6DoF simulation finished.")
        return sol

    def run_simulation_realtime_6dof(self, launch_angle_deg=45, azimuth_deg=90, sim_time=500):
        """실시간 3D 시각화와 함께 6DoF 시뮬레이션을 실행"""
        print("\n--- 1. Running full simulation to get trajectory data ---")
        results = self.run_simulation_6dof(launch_angle_deg, azimuth_deg, sim_time)
        
        time = results.t
        if len(time) < 2:
            print("❌ Simulation failed to generate enough data for animation.")
            return

        print("\n--- 2. Starting 3D Realtime Visualization ---")
        plt.ion()
        fig = plt.figure("Realtime 3D Trajectory", figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        pos_e = results.y[1]
        pos_n = results.y[0]
        altitude = -results.y[2]
        total_velocity = np.sqrt(results.y[3]**2 + results.y[4]**2 + results.y[5]**2)

        for i in range(0, len(time), 5):
            ax.clear()
            ax.plot(pos_e[:i+1], pos_n[:i+1], altitude[:i+1], 'b-')
            ax.plot([pos_e[i]], [pos_n[i]], [altitude[i]], 'ro')

            view_range = max(1000, np.max(altitude[:i+1]) / 2) if i > 0 else 1000
            ax.set_xlim(pos_e[i] - view_range, pos_e[i] + view_range)
            ax.set_ylim(pos_n[i] - view_range, pos_n[i] + view_range)
            ax.set_zlim(0, max(altitude[i] * 1.5, 1000))

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

def plot_results_6dof(results, sim_params):
    """6DoF 시뮬레이션 결과를 상세 그래프로 시각화"""
    print("📊 Plotting 6DoF simulation results...")
    
    time = results.t
    if len(time) < 2:
        print("❌ 시뮬레이션 데이터가 부족하여 그래프를 그릴 수 없습니다.")
        return

    # 데이터 추출 및 변환
    pos_n, pos_e, altitude = results.y[0], results.y[1], -results.y[2]
    vel_b_u, vel_b_v, vel_b_w = results.y[3], results.y[4], results.y[5]
    total_velocity = np.sqrt(vel_b_u**2 + vel_b_v**2 + vel_b_w**2)
    
    quaternions = results.y[6:10]
    roll, pitch, yaw = [], [], []
    for i in range(len(time)):
        r, p, y = quaternion_to_euler(quaternions[:, i])
        roll.append(r); pitch.append(p); yaw.append(y)
    
    initial_mass, final_mass = sim_params['m'], sim_params['m'] - sim_params['propellant_mass']
    burn_time = sim_params['burn_time']
    mass = np.piecewise(time, [time < burn_time, time >= burn_time], 
                        [lambda t: initial_mass - (initial_mass - final_mass) * t / burn_time, final_mass])
    
    angular_velocities = results.y[10:13]
    alphas, betas, aero_moments_M = [], [], []
    S, d = 0.6, 0.88
    Cm_alpha, Cm_q = -1.5, -15.0

    for i in range(len(time)):
        V = total_velocity[i]
        if V < 1e-6:
            alpha, beta = 0.0, 0.0
        else:
            alpha = math.atan2(vel_b_w[i], vel_b_u[i])
            beta = math.asin(vel_b_v[i] / V)
        alphas.append(math.degrees(alpha))
        betas.append(math.degrees(beta))
        
        rho = 1.225 * np.exp(-altitude[i] / 8500)
        q_dynamic = 0.5 * rho * V**2
        
        if time[i] < sim_params['pitch_time']:
            alpha_cmd = math.radians(sim_params['pitch_angle_deg_cmd']) * (time[i] / sim_params['pitch_time'])
        else:
            alpha_cmd = 0.0
        
        alpha_error = alpha - alpha_cmd
        q_rate = angular_velocities[1, i]
        
        pitch_moment = q_dynamic * S * d * ((Cm_alpha * alpha_error) + (Cm_q * (q_rate * d / (2 * V + 1e-6))))
        aero_moments_M.append(pitch_moment)

    # 그래프 생성
    figures = {
        "Figure 1: Velocity & Attitude": [('Velocity (m/s)', time, total_velocity), ('Pitch Angle (deg)', time, pitch), ('Yaw Angle (deg)', time, yaw)],
        "Figure 2: Position & Mass": [('North Position (m)', time, pos_n), ('East Position (m)', time, pos_e), ('Altitude (m)', time, altitude), ('Mass (kg)', time, mass)],
        "Figure 4: 6DoF Core Dynamics": [('Angular Velocity (deg/s)', [time, time, time], [np.degrees(angular_velocities[0, :]), np.degrees(angular_velocities[1, :]), np.degrees(angular_velocities[2, :])], ['p (Roll rate)', 'q (Pitch rate)', 'r (Yaw rate)']),
                                         ('Flight Angles (deg)', [time, time], [alphas, betas], ['alpha (AoA)', 'beta (Sideslip)']),
                                         ('Aerodynamic Moment (Nm)', [time, time, time], [[0]*len(time), aero_moments_M, [0]*len(time)], ['L (Roll Moment)', 'M (Pitch Moment)', 'N (Yaw Moment)'])]
    }

    for figname, subplots in figures.items():
        num_subplots = len(subplots)
        plt.figure(figname, figsize=(12, 4 * num_subplots))
        plt.suptitle(figname)
        for i, plot_data in enumerate(subplots, 1):
            ax = plt.subplot(num_subplots, 1, i)
            
            if len(plot_data) == 3: # Single line plot
                ylabel, xdata, ydata = plot_data
                plt.plot(xdata, ydata)
            else: # Multi-line plot
                ylabel, xdatas, ydatas, labels = plot_data
                for j in range(len(xdatas)):
                    plt.plot(xdatas[j], ydatas[j], label=labels[j])
                plt.legend()

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
    
    sim6dof = MissileSimulation6DoF()
    
    print("\n실행 모드를 선택하세요:")
    print("1. 실시간 3D 궤적 시뮬레이션")
    print("2. 상세 결과 그래프")
    
    mode = input("모드 선택 (1-2, 기본값: 1): ")

    launch_angle = 45
    sim_time = 500

    if mode == "2":
        print("\n--- 상세 결과 그래프 모드 실행 ---")
        results = sim6dof.run_simulation_6dof(launch_angle_deg=launch_angle, sim_time=sim_time)
        sim_params = {
            'm': sim6dof.m, 'propellant_mass': sim6dof.propellant_mass,
            'burn_time': sim6dof.burn_time, 'pitch_time': sim6dof.pitch_time,
            'pitch_angle_deg_cmd': sim6dof.pitch_angle_deg_cmd
        }
        plot_results_6dof(results, sim_params)
    else:
        print("\n--- 실시간 3D 궤적 시뮬레이션 모드 실행 ---")
        sim6dof.run_simulation_realtime_6dof(launch_angle_deg=launch_angle, sim_time=sim_time)

    print("\n미사일 궤적 시뮬레이션이 완료되었습니다.")

if __name__ == "__main__":
    main()