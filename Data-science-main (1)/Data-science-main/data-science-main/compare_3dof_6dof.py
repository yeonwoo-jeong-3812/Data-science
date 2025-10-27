"""
3DOF vs 6DOF 속도 및 질량 비교 스크립트
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 인코딩 설정
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'professor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Data-science'))

# 3DOF 시뮬레이션 임포트
from professor.main import MissileSimulation as MissileSimulation3DOF

# 6DOF 시뮬레이션 임포트
from main_6dof import MissileSimulation6DoF

def run_comparison():
    """3DOF와 6DOF 시뮬레이션을 실행하고 속도/질량 비교"""
    
    print("="*60)
    print("3DOF vs 6DOF 비교 시뮬레이션 시작")
    print("="*60)
    
    # 공통 파라미터
    launch_angle = 45
    sim_time = 500
    
    # ========== 3DOF 시뮬레이션 ========== #
    print("\n[1/2] 3DOF 시뮬레이션 실행 중...")
    sim_3dof = MissileSimulation3DOF(missile_type="SCUD-B", apply_errors=False)
    sim_3dof.initialize_simulation(launch_angle_deg=launch_angle, sim_time=sim_time)
    results_3dof = sim_3dof.run_simulation()
    
    # 3DOF 데이터 추출 (dict 형태)
    time_3dof = np.array(results_3dof['time'])
    velocity_3dof = np.array(results_3dof['velocity'])
    mass_3dof = np.array(results_3dof['mass'])
    
    print(f"✅ 3DOF 완료: {len(time_3dof)} 데이터 포인트")
    
    # ========== 6DOF 시뮬레이션 ========== #
    print("\n[2/2] 6DOF 시뮬레이션 실행 중...")
    sim_6dof = MissileSimulation6DoF(missile_type="SCUD-B")
    results_6dof = sim_6dof.run_simulation(launch_angle_deg=launch_angle, sim_time=sim_time)
    
    # 6DOF 데이터 추출
    time_6dof = results_6dof.t
    
    # 관성 좌표계 속도 계산 (위치의 시간 미분)
    pos_x = results_6dof.y[0]
    pos_y = results_6dof.y[1]
    pos_z = results_6dof.y[2]
    
    # 수치 미분으로 속도 계산
    vel_i_x = np.gradient(pos_x, time_6dof)
    vel_i_y = np.gradient(pos_y, time_6dof)
    vel_i_z = np.gradient(pos_z, time_6dof)
    velocity_6dof = np.sqrt(vel_i_x**2 + vel_i_y**2 + vel_i_z**2)
    
    # 6DOF 질량 계산 (시간 기반)
    m0 = sim_6dof.m0
    propellant_mass = sim_6dof.propellant_mass
    burn_time = sim_6dof.burn_time
    mass_6dof = np.where(time_6dof < burn_time,
                         m0 - (propellant_mass / burn_time) * time_6dof,
                         m0 - propellant_mass)
    
    print(f"✅ 6DOF 완료: {len(time_6dof)} 데이터 포인트")
    
    # ========== 비교 그래프 생성 ========== #
    print("\n[3/3] 비교 그래프 생성 중...")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 속도 비교
    ax1 = axes[0]
    ax1.plot(time_3dof, velocity_3dof, 'b-', linewidth=2, label='3DOF (Professor)')
    ax1.plot(time_6dof, velocity_6dof, 'r--', linewidth=2, label='6DOF (Your Code)')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Velocity (m/s)', fontsize=12)
    ax1.set_title('Velocity Comparison: 3DOF vs 6DOF', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 질량 비교
    ax2 = axes[1]
    ax2.plot(time_3dof, mass_3dof, 'b-', linewidth=2, label='3DOF (Professor)')
    ax2.plot(time_6dof, mass_6dof, 'r--', linewidth=2, label='6DOF (Your Code)')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Mass (kg)', fontsize=12)
    ax2.set_title('Mass Comparison: 3DOF vs 6DOF', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 저장
    save_path = "comparison_3dof_6dof.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 비교 그래프 저장: {save_path}")
    
    # 통계 출력
    print("\n" + "="*60)
    print("통계 비교")
    print("="*60)
    print(f"{'항목':<20} {'3DOF':<15} {'6DOF':<15} {'차이':<15}")
    print("-"*60)
    
    # 최종 속도
    final_v_3dof = velocity_3dof[-1]
    final_v_6dof = velocity_6dof[-1]
    v_diff = abs(final_v_3dof - final_v_6dof)
    v_diff_pct = (v_diff / final_v_3dof) * 100
    print(f"{'최종 속도 (m/s)':<20} {final_v_3dof:<15.2f} {final_v_6dof:<15.2f} {v_diff:.2f} ({v_diff_pct:.2f}%)")
    
    # 최종 질량
    final_m_3dof = mass_3dof[-1]
    final_m_6dof = mass_6dof[-1]
    m_diff = abs(final_m_3dof - final_m_6dof)
    m_diff_pct = (m_diff / final_m_3dof) * 100
    print(f"{'최종 질량 (kg)':<20} {final_m_3dof:<15.2f} {final_m_6dof:<15.2f} {m_diff:.2f} ({m_diff_pct:.2f}%)")
    
    # 최대 속도
    max_v_3dof = np.max(velocity_3dof)
    max_v_6dof = np.max(velocity_6dof)
    max_v_diff = abs(max_v_3dof - max_v_6dof)
    max_v_diff_pct = (max_v_diff / max_v_3dof) * 100
    print(f"{'최대 속도 (m/s)':<20} {max_v_3dof:<15.2f} {max_v_6dof:<15.2f} {max_v_diff:.2f} ({max_v_diff_pct:.2f}%)")
    
    print("="*60)
    
    plt.show()

if __name__ == "__main__":
    run_comparison()
