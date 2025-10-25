#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Missile Trajectory Simulation - 원본 기반 최소 수정 버전
"""
import os
# Qt Wayland 플러그인 오류 방지를 위한 환경 변수 설정
os.environ["QT_QPA_PLATFORM"] = "xcb"

import numpy as np
import matplotlib
# matplotlib 백엔드 설정
matplotlib.use('TkAgg')  # TkAgg 백엔드 사용 (Qt 의존성 제거)
import matplotlib.pyplot as plt
import datetime
from matplotlib import animation
from scipy.integrate import solve_ivp
import config as cfg
import math
import platform

# matplotlib 음수 기호 표시 설정만 유지
plt.rcParams['axes.unicode_minus'] = False

# 유틸리티 함수들은 동일하므로 생략...
def plot_with_guarantee(fig, save_path, title, show_plot=True):
    """저장 후 새 창에서 그래프를 확실히 표시하는 유틸리티 함수"""
    # 원본 그래프 저장
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"그래프가 저장되었습니다: {save_path}")
    
    # 그래프 표시 처리
    if show_plot:
        # 새 창에서 이미지로 불러와 표시 (확실한 표시 보장)
        fig_show = plt.figure(figsize=(18, 10))
        plt.imshow(plt.imread(save_path))
        plt.axis('off')  # 축 숨기기
        plt.title(title, fontsize=14)
        plt.tight_layout()
        plt.show(block=True)  # 반드시 차단 모드로 표시
        plt.close(fig_show)
    
    # 원본 그래프 창 닫기
    plt.close(fig)
    return True

def get_numeric_input(prompt, default_value=None, min_value=None, max_value=None):
    """숫자 입력을 받고 검증하는 유틸리티 함수"""
    while True:
        user_input = input(prompt)
        if not user_input and default_value is not None:
            return default_value
        
        try:
            value = float(user_input)
            if min_value is not None and value < min_value:
                print(f"입력 값은 {min_value} 이상이어야 합니다.")
                continue
            if max_value is not None and value > max_value:
                print(f"입력 값은 {max_value} 이하여야 합니다.")
                continue
            return value
        except ValueError:
            print("유효한 숫자를 입력해 주세요.")

def get_angle_input(prompt, default_values=None):
    """발사각 입력을 받는 유틸리티 함수"""
    while True:
        angles_input = input(prompt)
        if not angles_input and default_values is not None:
            return default_values
        
        try:
            angles = [float(angle.strip()) for angle in angles_input.split(',')]
            # 각도는 보통 0-90도 사이의 값
            invalid_angles = [angle for angle in angles if angle < 0 or angle > 90]
            if invalid_angles:
                print(f"유효하지 않은 각도: {invalid_angles}")
                print("발사각은 0-90도 사이여야 합니다.")
                continue
            return angles
        except ValueError:
            print("쉼표로 구분된 유효한 각도 값을 입력해 주세요.")

def get_missile_types_input(prompt, available_types, default_value="SCUD-B"):
    """미사일 유형 입력을 받는 유틸리티 함수"""
    while True:
        missile_types_input = input(prompt)
        
        if missile_types_input.lower() == 'all' or not missile_types_input:
            return available_types
        
        types = [m.strip() for m in missile_types_input.split(',')]
        invalid_types = [t for t in types if t not in available_types]
        
        if invalid_types:
            print(f"잘못된 미사일 유형: {', '.join(invalid_types)}")
            print(f"사용 가능한 유형: {', '.join(available_types)}")
        else:
            return types

def prepare_save_path(save_path_input, default_dir, default_filename):
    """저장 경로를 준비하는 유틸리티 함수"""
    # 결과 폴더 생성
    os.makedirs(default_dir, exist_ok=True)
    
    if not save_path_input or save_path_input.strip() == '':
        # 기본 파일명 자동 생성
        now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(default_dir, f"{default_filename}_{now_str}.png")
    else:
        # 확장자 없으면 png로
        if not save_path_input.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf', '.svg')):
            save_path = save_path_input + '.png'
        # 상대경로면 default_dir 폴더로
        if not os.path.isabs(save_path):
            save_path = os.path.join(default_dir, save_path)
    
    # 디렉토리 확인 및 생성
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    
    return save_path

class MissileSimulation:
    """미사일 궤적 시뮬레이션 클래스"""
    
    def __init__(self, missile_type="SCUD-B", apply_errors=True):
        """생성자
        
        Args:
            missile_type: 미사일 유형 (기본값: SCUD-B)
            apply_errors: 오차 모델 적용 여부 (기본값: True)
        """
        # 초기화
        self.results = None  # 시뮬레이션 결과 저장용 딕셔너리
        self.states = []     # 상태 저장용 리스트
        self.t = []          # 시간 저장용 리스트
        self.alpha_list = [] # 받음각 저장용 리스트
        self.CD_list = []    # 항력계수 저장용 리스트
        self.fuel_list = []  # 연료 소모량 저장용 리스트
        self.mach_list = []  # 마하수 저장용 리스트
        self.phase_list = [] # 비행 단계 저장용 리스트
        
        # 대기권 상태 추적을 위한 변수 추가
        self.in_atmosphere = True  # 초기에는 대기권 내에 있다고 가정
        
        # 미사일 유형 설정 및 공기역학 테이블 초기화
        self.missile_type = missile_type
        self.update_missile_type(missile_type)
        
        # 오차 모델 설정
        self.apply_errors = apply_errors
        self.error_seed = np.random.randint(1, 10000)  # 오차 생성을 위한 시드
        
        # 오차 크기 설정 (각 항목별 상대 오차)
        self.error_factors = {
            'thrust': 0.02,       # 추력 오차: ±2%
            'cd': 0.05,           # 항력계수 오차: ±5%
            'cl': 0.05,           # 양력계수 오차: ±5%
            'density': 0.03,      # 대기 밀도 오차: ±3%
            'isp': 0.01,          # 비추력 오차: ±1%
            'wind': [3.0, 3.0],   # 풍속 오차 [x, y] (m/s)
            'mass': 0.01,         # 질량 오차: ±1%
            'gamma': 0.2,         # 피치각 오차: ±0.2도
            'psi': 0.2            # 방위각 오차: ±0.2도
        }
    
    def initialize_simulation(self, launch_angle_deg=45, azimuth_deg=90, sim_time=None):
        """시뮬레이션 초기화
        
        Args:
            launch_angle_deg: 발사각 (도 단위)
            azimuth_deg: 방위각 (도 단위, 0=북쪽, 90=동쪽)
            sim_time: 시뮬레이션 시간 (초, None이면 기본값 사용)
        """
        # 초기화
        self.alpha_list = []  # 받음각 리스트 초기화
        self.CD_list = []     # 항력계수 리스트 초기화
        self.fuel_list = []   # 연료 소모량 리스트 초기화
        self.mach_list = []   # 마하수 리스트 초기화
        self.phase_list = []  # 비행 단계 리스트 초기화
        # 결과 저장용 딕셔너리 초기화
        self.results = {
            'time': [], 'velocity': [], 'gamma': [], 'psi': [],
            'x': [], 'y': [], 'h': [], 'weight': [], 'mass': [],
            'alpha': [], 'CD': [], 'fuel': [], 'mach': [], 'phase': []
        }
        # 시간 설정
        self.sim_time = sim_time if sim_time is not None else cfg.SIM_TIME
        
        # 초기 상태
        self.init_speed = 0.0  # 초기 속도 (m/s)
        self.launch_angle_rad = math.radians(launch_angle_deg)  # 발사각 (라디안)
        self.launch_azimuth_rad = math.radians(azimuth_deg)  # 방위각 (라디안)
        
        # 미사일 유형 정보 가져오기
        missile_info = cfg.MISSILE_TYPES[self.missile_type]
        
        # 초기 상태 벡터 [V, gamma, psi, x, y, h, W, M]
        # V: 속도 (m/s)
        # gamma: 피치각 (rad)
        # psi: 방위각 (rad)
        # x, y: 위치 (m)
        # h: 고도 (m)
        # W: 중량 (N)
        # M: 질량 (kg)
        self.initial_state = np.array([
            self.init_speed,
            self.launch_angle_rad,
            self.launch_azimuth_rad,
            0.0,  # x 위치
            0.0,  # y 위치
            0.0,  # 고도
            missile_info["launch_weight"] * cfg.G,  # 중량 (N)
            missile_info["launch_weight"]  # 질량 (kg)
        ])
        
        # 시뮬레이션 준비 완료
        print(f"초기화 완료: {self.missile_type}, 발사각 {launch_angle_deg}°, 방위각 {azimuth_deg}°")
    
    def update_missile_type(self, missile_type):
        """미사일 유형을 업데이트하고 관련 제원 및 공기역학 테이블 설정"""
        if missile_type in cfg.MISSILE_TYPES:
            self.missile_type = missile_type
            missile_data = cfg.MISSILE_TYPES[missile_type]
            
            # 기본 물리적 특성 설정
            self.diameter = missile_data["diameter"]
            self.length = missile_data["length"]
            self.nozzle_diameter = missile_data["nozzle_diameter"]
            self.propellant_type = missile_data["propellant_type"]
            
            # 추진 관련 특성 설정
            self.missile_mass = missile_data["launch_weight"]
            self.propellant_mass = missile_data["propellant_mass"]
            self.isp_sea = missile_data["isp_sea"] 
            
            # 진공 비추력 설정 - 호환성을 위해 isp_vacuum 및 isp_vac 모두 지원
            try:
                if "isp_vacuum" in missile_data:
                    self.isp_vacuum = missile_data["isp_vacuum"]
                elif "isp_vac" in missile_data:
                    self.isp_vacuum = missile_data["isp_vac"]
                else:
                    # 기본값 설정 (해수면 비추력보다 10% 증가)
                    self.isp_vacuum = self.isp_sea * 1.1
                    print(f"\t주의: '{missile_type}'에 대한 진공 비추력이 정의되지 않았습니다. 해수면 비추력의 110%를 사용합니다.")
            except KeyError as e:
                print(f"\t주의: '{missile_type}'에 대한 비추력 데이터 없음: {e}")
                self.isp_vacuum = self.isp_sea * 1.1
            
            self.burn_time = missile_data["burn_time"]
            
            # 날개 면적 설정 - 호환성을 위해 면적 계산 문제 수정
            try:
                self.wing_area = missile_data["reference_area"]
            except KeyError:
                # 기본값 설정 - 직경 기반으로 계산
                self.wing_area = np.pi * (self.diameter/2)**2
                print(f"\t주의: '{missile_type}'에 대한 reference_area가 정의되지 않았습니다. 미사일 단면적을 사용합니다: {self.wing_area:.2f} m²")
            
            # 비행 단계 관련 시간 설정
            self.vertical_time = missile_data["vertical_time"]
            self.pitch_time = missile_data["pitch_time"]
            self.pitch_angle_deg = missile_data["pitch_angle_deg"]
            
            # 항력 테이블 계산
            self.CD_TABLE = cfg.calculate_cd_table(
                self.diameter, 
                self.length, 
                self.nozzle_diameter, 
                self.propellant_type
            )
            
            # 추진력 프로파일 설정 (기본값은 일정한 추진력)
            self.thrust_profile = missile_data.get("thrust_profile", None)
            
            # 항력 계수 보정값 설정 (기본값은 1.0)
            self.drag_multiplier = missile_data.get("drag_multiplier", 1.0)
            
            print(f"미사일 타입 '{missile_type}' 설정 완료: 연소시간={self.burn_time}초, ISP={self.isp_sea}")
            return True
        else:
            print(f"경고: 미사일 유형 '{missile_type}'을(를) 찾을 수 없습니다. 기본 항력계수 테이블을 사용합니다.")
            self.CD_TABLE = cfg.BASE_CD_TABLE.copy()
            return False
    
    def get_density(self, h):
        """고도에 따른 대기 밀도 계산 (표준 대기 모델)"""
        # 음수 고도 처리
        if h < 0:
            return cfg.STD_DENSITY_SEA_LEVEL
            
        # 84.852km 이하의 고도는 표준 대기 모델 사용
        if h <= 84852:
            # 대기층 찾기
            for layer in cfg.ATMOSPHERIC_LAYERS:
                lower_bound, upper_bound, lapse_rate, base_h, base_T = layer
                
                if lower_bound <= h < upper_bound:
                    # 해당 층에서의 온도 계산
                    T = base_T + lapse_rate * (h - base_h)
                    
                    # 정적 대기층(lapse_rate = 0)인 경우
                    if abs(lapse_rate) < 1e-10:
                        # P = P_b * exp(-g * (h - h_b) / (R * T_b))
                        # 기준 고도에서의 기압 계산 
                        # (재귀적으로 해당 고도의 기압 계산)
                        base_pressure = self.get_pressure_at_altitude(base_h)
                        
                        # 해당 고도에서의 기압 계산
                        pressure = base_pressure * math.exp(
                            -cfg.STD_GRAVITY * cfg.AIR_MOLAR_MASS * (h - base_h) / 
                            (cfg.UNIVERSAL_GAS_CONSTANT * base_T)
                        )
                    else:
                        # 기준 고도에서의 기압 계산
                        base_pressure = self.get_pressure_at_altitude(base_h)
                        
                        # T / T_b = (P / P_b)^(R*a/g)
                        # 여기서 a는 단열 감율
                        exponent = cfg.STD_GRAVITY / (cfg.AIR_GAS_CONSTANT * lapse_rate)
                        pressure = base_pressure * (T / base_T) ** (-exponent)
                    
                    # 이상 기체 방정식으로 밀도 계산: ρ = P / (R * T)
                    density = pressure / (cfg.AIR_GAS_CONSTANT * T)
                    return density
        
        # 84.852km 초과의 고도는 지수함수 모델 사용
        else:
            # 고고도 참조 데이터에서 적절한 참조점 찾기
            for ref in cfg.HIGH_ALTITUDE_REFERENCE:
                ref_altitude, ref_density, scale_height = ref
                
                if h < ref_altitude or ref == cfg.HIGH_ALTITUDE_REFERENCE[-1]:
                    # 지수 감소 모델: ρ = ρ_ref * exp(-(h - h_ref) / H)
                    # H는 척도 고도(scale height)
                    density = ref_density * math.exp(-(h - ref_altitude) / scale_height)
                    return density
                    
        # 기본값 반환 (오류 발생 방지)
        return 1e-15
    
    def get_density_with_error(self, h):
        """오차가 적용된 대기 밀도 계산"""
        base_density = self.get_density(h)
        
        if not self.apply_errors:
            return base_density
            
        # 고도별로 일관된 오차를 위해 고도 기반 시드 설정
        local_seed = int(self.error_seed + h / 1000)
        np.random.seed(local_seed)
        
        # 표준편차가 정해진 비율인 정규분포로 오차 생성
        error_factor = np.random.normal(1.0, self.error_factors['density'])
        
        # 물리적으로 타당한 범위로 제한 (0.9 ~ 1.1)
        error_factor = max(0.9, min(1.1, error_factor))
        
        return base_density * error_factor
    
    def get_pressure_at_altitude(self, h):
        """주어진 고도에서의 기압 계산 (비재귀적 방식)"""
        # 음수 고도 처리
        if h < 0:
            return cfg.STD_PRESSURE_SEA_LEVEL
            
        # 해면 고도인 경우 표준 기압 반환
        if h == 0:
            return cfg.STD_PRESSURE_SEA_LEVEL
        
        # 현재 고도까지의 대기층 찾기
        current_layer_index = 0
        for i, layer in enumerate(cfg.ATMOSPHERIC_LAYERS):
            lower_bound, upper_bound, _, _, _ = layer
            
            if lower_bound <= h < upper_bound:
                current_layer_index = i
                break
        
        # 층별 순차 계산 (해면에서부터 현재 고도까지)
        pressure = cfg.STD_PRESSURE_SEA_LEVEL
        
        for i in range(current_layer_index + 1):
            layer = cfg.ATMOSPHERIC_LAYERS[i]
            lower_bound, upper_bound, lapse_rate, base_h, base_T = layer
            
            # 각 층의 상단 고도 결정 (현재 고도가 이 층에 있으면 현재 고도를 상단으로)
            top_h = min(upper_bound, h) if i == current_layer_index else upper_bound
            
            # 이전 층의 상단 고도를 현재 층의 하단 고도로 설정
            bottom_h = lower_bound
            
            # 하단과 상단의 온도 계산
            bottom_T = base_T + lapse_rate * (bottom_h - base_h)
            top_T = base_T + lapse_rate * (top_h - base_h)
            
            # 정적 대기층(lapse_rate ≈ 0)인 경우
            if abs(lapse_rate) < 1e-10:
                pressure = pressure * math.exp(
                    -cfg.STD_GRAVITY * cfg.AIR_MOLAR_MASS * (top_h - bottom_h) / 
                    (cfg.UNIVERSAL_GAS_CONSTANT * bottom_T)
                )
            else:
                # 동적 대기층
                exponent = cfg.STD_GRAVITY / (cfg.AIR_GAS_CONSTANT * lapse_rate)
                pressure = pressure * (top_T / bottom_T) ** (-exponent)
        
        return pressure

    def get_CD_interpolated(self, mach, alpha_deg=0):
        """마하 수 기반 보간된 항력 계수 반환"""
        # 테이블에서 기본 항력계수 보간
        keys = sorted(self.CD_TABLE.keys())
        
        base_CD = 0.3  # 기본값
        
        if mach <= keys[0]:
            base_CD = self.CD_TABLE[keys[0]]
        elif mach >= keys[-1]:
            base_CD = self.CD_TABLE[keys[-1]]
        else:
            for i in range(len(keys) - 1):
                if keys[i] <= mach < keys[i + 1]:
                    m0, m1 = keys[i], keys[i + 1]
                    cd0, cd1 = self.CD_TABLE[m0], self.CD_TABLE[m1]
                    base_CD = cd0 + (cd1 - cd0) * (mach - m0) / (m1 - m0)
                    break
        
        # 받음각에 따른 추가 항력 (0도에서는 0, 각도가 증가할수록 항력 증가)
        alpha_factor = 0.05 * (alpha_deg / 20.0) if alpha_deg > 0 else 0
        
        # 오차 적용
        if self.apply_errors:
            # 마하수와 받음각에 따라 다른 시드 사용 (범위 제한)
            local_seed = int((self.error_seed + mach * 100 + alpha_deg * 10) % (2**32 - 1))
            np.random.seed(local_seed)
            error_cd = np.random.normal(0, self.error_factors['cd'] * (base_CD + alpha_factor))
            return base_CD + alpha_factor + error_cd
        
        return base_CD + alpha_factor

    def dynamics_vertical(self, t, state):
        """수직상승 단계 미사일 동역학 모델 - 수치적 안정성 개선 버전"""
        # 상태 변수
        V, gamma, psi, x, y, h, W, M_t = state
        
        # 수치적 안정성을 위한 최소값 설정
        V_min = 1.0
        M_min = 100.0
        V_safe = max(abs(V), V_min)
        M_safe = max(M_t, M_min)
        
        # 중력 계산
        g = cfg.G * cfg.R**2 / (cfg.R + h)**2
        
        # 마하수 계산을 위한 음속 계산 (고도에 따른 온도 기반)
        temperature = max(216.65, 288.15 - 0.0065 * h)  # 최소 온도 제한
        if h > 11000:  # 성층권(11km 이상)에서는 온도가 일정
            temperature = 216.65
        sound_speed = 20.05 * np.sqrt(temperature)  # 음속 (m/s)
        mach = V_safe / sound_speed if sound_speed > 0 else 0
        
        # 동압 계산
        rho = self.get_density(h)
        q = 0.5 * rho * V_safe**2
        
        # 받음각 계산 (수직상승단계에서는 0)
        alpha = 0.0  # 받음각 (rad)
        
        # 양력과 항력 계산 - 마하수에 따른 항력계수 변화 추가
        CL = cfg.CL_VERTICAL  # 수직 상승에는 고정값 사용
        
        # 양력계수에 오차 추가
        if self.apply_errors:
            local_seed = int(self.error_seed + t * 10)
            np.random.seed(local_seed)
            cl_error = np.random.normal(0, self.error_factors['cl'] * CL)
            CL += cl_error
            
        # 마하수에 따른 항력계수 보정 (오차 포함)
        CD = self.get_CD_interpolated(mach, alpha_deg=0)
        
        # K 팩터에 의한 추가 항력
        CD += cfg.K * CL**2
        
        # 추진단계에서 연료 소모에 따른 항력계수 보정
        fuel_ratio = (self.missile_mass - M_safe) / self.propellant_mass
        if fuel_ratio <= 1.0:  # 연료 소모 진행 중
            # 연료 소모비율에 따른 항력계수 보정 (소모가 진행됨에 따라 감소)
            CD *= (1.0 - 0.2 * fuel_ratio)
        
        D = q * CD * self.wing_area
        
        # 추력 계산 - 인스턴스 변수 사용
        # 추진력 프로파일이 있으면 사용, 없으면 일정한 추진력
        if hasattr(self, 'thrust_profile') and self.thrust_profile is not None and isinstance(self.thrust_profile, dict):
            # 시간에 따른 추진력 프로파일 적용
            burn_fraction = t / self.burn_time if self.burn_time > 0 else 1.0
            burn_fraction = min(1.0, burn_fraction)  # 0~1 사이 값으로 제한
            
            # 추진력 프로파일에서 가장 가까운 값 찾기
            profile_times = sorted(self.thrust_profile.keys())
            
            # 정확한 시간이 있으면 사용, 없으면 보간
            if burn_fraction in self.thrust_profile:
                thrust_factor = self.thrust_profile[burn_fraction]
            else:
                # 프로파일 시간 값 중에서 burn_fraction보다 작은 값들 찾기
                lower_times = [x for x in profile_times if x <= burn_fraction]
                # 프로파일 시간 값 중에서 burn_fraction보다 큰 값들 찾기
                upper_times = [x for x in profile_times if x > burn_fraction]
                
                if not lower_times:  # 가장 초기 상태
                    thrust_factor = self.thrust_profile[profile_times[0]]
                elif not upper_times:  # 연소 완료 상태
                    thrust_factor = self.thrust_profile[profile_times[-1]]
                else:  # 선형 보간
                    t_lower = max(lower_times)
                    t_upper = min(upper_times)
                    f_lower = self.thrust_profile[t_lower]
                    f_upper = self.thrust_profile[t_upper]
                    # 선형 보간 공식
                    thrust_factor = f_lower + (f_upper - f_lower) * (burn_fraction - t_lower) / (t_upper - t_lower)
        else:
            # 기본값: 일정한 추진력
            thrust_factor = 1.0
        
        # 기본 추진력 계산 - 인스턴스 변수 사용
        if t <= self.burn_time:
            T = self.isp_sea * (self.propellant_mass / self.burn_time) * g * thrust_factor
        else:
            T = 0
        
        if self.apply_errors:
            # 시간에 따라 일관된 추력 오차 적용
            time_segment = int(t / 5)  # 5초마다 새로운 오차
            local_seed = int(self.error_seed + time_segment)
            np.random.seed(local_seed)
            thrust_error = np.random.normal(0, self.error_factors['thrust'] * T)
            T += thrust_error
            
            # ISP에도 오차 적용
            isp_error_factor = 1.0 + np.random.normal(0, self.error_factors['isp'])
            isp_error_factor = max(0.98, min(1.02, isp_error_factor))  # ±2% 범위 제한
            T *= isp_error_factor
        
        # 기타 변수
        phi = 0    # 옆미끄러짐
        
        # 풍속 효과 적용
        wind_x, wind_y = 0, 0
        if self.apply_errors:
            # 고도에 따른 풍속 변화
            altitude_segment = int(h / 1000)  # 1km마다 새로운 풍속
            local_seed = int(self.error_seed + altitude_segment)
            np.random.seed(local_seed)
            
            # 고도가 증가할수록 풍속 증가 (약 10km 고도에서 최대)
            wind_factor = min(1.0, h / 10000)
            wind_x = np.random.normal(0, self.error_factors['wind'][0] * wind_factor)
            wind_y = np.random.normal(0, self.error_factors['wind'][1] * wind_factor)
        
        # 상태 미분값 계산 - 수직상승 단계에서는 수직 방향만
        dV = (T * np.cos(alpha) - D - M_safe * g * np.sin(gamma)) / M_safe
        
        # velocity가 음수가 되지 않도록 안전장치 추가
        if V < 50.0 and dV < 0:
            dV = max(dV, -V/20.0)  # 속도가 낮을 때 점진적 감속
        
        # 속도가 음수가 되지 않도록 제한
        if V + dV * 0.1 < 0:  # 0.1초 후 예상 속도가 음수가 될 경우
            dV = max(dV, -V/10.0)  # 더 완만한 감속
        
        dgamma = 0  # 수직상승 단계에서는 변화 없음
        
        # 0으로 나누기 방지
        if V < 0.1 or abs(np.cos(gamma)) < 1e-6:
            dpsi = 0
        else:
            dpsi = 0  # 수직상승 단계에서는 방위각 변화 없음
            
        # 수직상승 단계에서는 X, Y 변화 없음 (방위각 변화도 없음)
        dx = 0  # 수직상승 단계에서는 수평 이동 없음
        dy = 0  # 수직상승 단계에서는 수평 이동 없음
        dh = V_safe * np.sin(gamma)
        
        # 연료 소모율에 오차 추가
        fuel_consumption_rate = self.propellant_mass / self.burn_time
        if self.apply_errors:
            local_seed = int(self.error_seed + int(t))
            np.random.seed(local_seed)
            fuel_error = np.random.normal(1.0, self.error_factors['mass'])
            fuel_error = max(0.98, min(1.02, fuel_error))  # ±2% 제한
            fuel_consumption_rate *= fuel_error
        
        if t <= self.burn_time:
            dW = -T * cfg.TSFC  # TSFC는 고정값으로 유지
            dM = -fuel_consumption_rate
        else:
            dW = 0
            dM = 0
        
        # 현재 상태 저장
        if not hasattr(self, 'last_t') or t > self.last_t:
            self.last_t = t
            self.alpha_list.append(alpha * cfg.RAD_TO_DEG)  # 도 단위로 저장
            self.CD_list.append(CD)
            self.fuel_list.append(self.missile_mass - M_safe)  # 소모된 연료량
            self.mach_list.append(mach)
            self.phase_list.append("수직상승")
        
        return [dV, dgamma, dpsi, dx, dy, dh, dW, dM]
    
    def dynamics_pitch(self, t, state):
        """피치 프로그램 단계 미사일 동역학 모델 - 수치적 안정성 개선 버전"""
        # 상태 변수
        V, gamma, psi, x, y, h, W, M_t = state
        
        # 수치적 안정성을 위한 최소값 설정
        V_min = 1.0
        M_min = 100.0
        V_safe = max(abs(V), V_min)
        M_safe = max(M_t, M_min)
        
        # 중력 계산
        g = cfg.G * cfg.R**2 / (cfg.R + h)**2
        
        # 마하수 계산을 위한 음속 계산 (고도에 따른 온도 기반)
        temperature = max(216.65, 288.15 - 0.0065 * h)  # 최소 온도 제한
        if h > 11000:  # 성층권(11km 이상)에서는 온도가 일정
            temperature = 216.65
        sound_speed = 20.05 * np.sqrt(temperature)  # 음속 (m/s)
        mach = V_safe / sound_speed if sound_speed > 0 else 0
        
        # 동압 계산
        rho = self.get_density(h)
        q = 0.5 * rho * V_safe**2
        
        # 피치 프로그램 단계에서 받음각 계산
        pitch_rad = self.pitch_angle_deg * cfg.DEG_TO_RAD
        # 피치 시간에 따라 받음각 점진적 증가
        pitch_progress = min(1.0, max(0.0, (t - self.vertical_time) / self.pitch_time))
        alpha = pitch_rad * pitch_progress  # 받음각 (rad)
        
        # 양력과 항력 계산 - 마하수에 따른 항력계수 변화 추가
        CL = cfg.CL_PITCH
        
        # 마하수에 따른 항력계수 보정
        CD = self.get_CD_interpolated(mach, alpha_deg=pitch_rad * cfg.RAD_TO_DEG)
        
        # K 팩터에 의한 추가 항력
        CD += cfg.K * CL**2
        
        # 받음각에 따른 추가 항력
        CD += 0.02 * alpha / cfg.DEG_TO_RAD  # 받음각에 따른 항력 증가
        
        # 추진단계에서 연료 소모에 따른 항력계수 보정
        fuel_ratio = (self.missile_mass - M_safe) / self.propellant_mass
        if fuel_ratio <= 1.0:  # 연료 소모 진행 중
            # 연료 소모비율에 따른 항력계수 보정 (소모가 진행됨에 따라 감소)
            CD *= (1.0 - 0.2 * fuel_ratio)
        
        # 양력 및 항력 계산
        L = q * CL * self.wing_area
        D = q * CD * self.wing_area
        
        # 추력 계산
        if t <= self.burn_time:
            T = self.isp_sea * (self.propellant_mass / self.burn_time) * g
        else:
            T = 0
        
        # 기타 변수
        phi = 0    # 옆미끄러짐
        
        # 상태 미분값 계산 - 수치적 안정성 개선
        dV = (T * np.cos(alpha) - D - M_safe * g * np.sin(gamma)) / M_safe
        
        # 피치 프로그램에서의 dgamma는 시간에 따른 프로그램 각도 변화
        # 부드러운 전환을 위해 사인 함수 적용
        if self.pitch_time > 0:
            # 피치 진행도에 따른 smoothing factor 계산 (0→1 구간에서 점점 증가하다 감소)
            # pitch_progress는 이미 위에서 계산됨
            # 사인 함수 기반 스무딩 - 처음과 끝에서 부드럽게 전환
            
            # 발사각에 따른 적응형 평활화: 낮은 각도에서는 평활화 강도를 줄임
            launch_angle_deg = gamma * cfg.RAD_TO_DEG  # 현재 발사각 (도)
            
            # 발사각이 낮을수록 평활화 강도를 줄임
            angle_factor = min(1.0, max(0.2, launch_angle_deg / 45.0))
            
            # 피치 진행도에 따른 부드러운 전환, 발사각에 따라 강도 조절
            if pitch_progress < 0.5:
                smoothing_factor = np.sin(np.pi * pitch_progress) * angle_factor
            else:
                smoothing_factor = np.sin(np.pi * (1 - pitch_progress)) * angle_factor
            
            # 15도 이하의 낮은 발사각에서는 smoothing을 최소화
            if launch_angle_deg <= 15:
                smoothing_factor = max(0, smoothing_factor)  # 음수 방지
                
            dgamma = -pitch_rad / self.pitch_time * smoothing_factor
        else:
            dgamma = 0
        
        dpsi = 0  # 피치 단계에서는 방위각 변화 없음
        
        # 위치 변화 - 수정된 좌표계 적용
        # X = 동쪽, Y = 북쪽으로 가정
        dx = V_safe * np.cos(gamma) * np.sin(psi)  # 동쪽 성분
        dy = V_safe * np.cos(gamma) * np.cos(psi)  # 북쪽 성분
        dh = V_safe * np.sin(gamma)
        
        # 연료 소모
        if t <= self.burn_time:
            dW = -T * cfg.TSFC  # TSFC는 고정값으로 유지
            dM = -self.propellant_mass / self.burn_time
        else:
            dW = 0
            dM = 0
        
        # 현재 상태 저장
        if not hasattr(self, 'last_t') or t > self.last_t:
            self.last_t = t
            self.alpha_list.append(alpha * cfg.RAD_TO_DEG)  # 도 단위로 저장
            self.CD_list.append(CD)
            self.fuel_list.append(self.missile_mass - M_safe)  # 소모된 연료량
            self.mach_list.append(mach)
            self.phase_list.append("피치프로그램")
        
        return [dV, dgamma, dpsi, dx, dy, dh, dW, dM]
    
    def dynamics_constant(self, t, state):
        """등자세 선회 단계 미사일 동역학 모델 - 수치적 안정성 개선 버전"""
        # 상태 변수
        V, gamma, psi, x, y, h, W, M_t = state
        
        # 수치적 안정성을 위한 최소값 설정
        V_min = 1.0  # 최소 속도 1 m/s
        M_min = 100.0  # 최소 질량 100 kg
        
        # 안전한 값으로 제한
        V_safe = max(abs(V), V_min)
        M_safe = max(M_t, M_min)
        
        # 중력 계산
        g = cfg.G * cfg.R**2 / (cfg.R + h)**2
        
        # 마하수 계산을 위한 음속 계산 (고도에 따른 온도 기반)
        temperature = max(216.65, 288.15 - 0.0065 * h)  # 최소 온도 제한
        if h > 11000:  # 성층권(11km 이상)에서는 온도가 일정
            temperature = 216.65
        sound_speed = 20.05 * np.sqrt(temperature)  # 음속 (m/s)
        mach = V_safe / sound_speed if sound_speed > 0 else 0
        
        # 동압 계산
        rho = self.get_density(h)
        q = 0.5 * rho * V_safe**2
        
        # 양력과 항력 계산
        CL = cfg.CL_CONSTANT
        L = q * CL * self.wing_area
        
        # 마하수에 따른 항력계수 보정
        CD = self.get_CD_interpolated(mach, alpha_deg=0)
        
        # K 팩터에 의한 추가 항력
        CD += cfg.K * CL**2
        
        # 추진단계에서 연료 소모에 따른 항력계수 보정
        fuel_ratio = (self.missile_mass - M_t) / self.propellant_mass
        if fuel_ratio <= 1.0:  # 연료 소모 진행 중
            # 연료 소모비율에 따른 항력계수 보정 (소모가 진행됨에 따라 감소)
            CD *= (1.0 - 0.2 * fuel_ratio)
        
        D = q * CD * self.wing_area
        
        # 추력 계산 - 연소 시간 체크 추가
        if t <= self.burn_time:  # 연소 시간 내에서만 추력 생성
            T = self.isp_sea * (self.propellant_mass / self.burn_time) * g
        else:
            T = 0  # 연소 완료 후에는 추력 없음
        
        # 받음각 계산 - 수치적 안정성 개선
        alpha = 0  # 간단화하여 0으로 설정
        
        # 상태 미분값 계산 - 수치적 안정성 개선
        dV = (T * np.cos(alpha) - D - M_safe * g * np.sin(gamma)) / M_safe
        
        # dgamma 계산 시 분모가 0에 가까워지는 것을 방지
        denominator = M_safe * V_safe
        if abs(denominator) < 1e-6:
            dgamma = 0.0  # 분모가 너무 작으면 0으로 설정
        else:
            dgamma = (T * np.sin(alpha) + L - M_safe * g * np.cos(gamma)) / denominator
        
        # dgamma 변화율 제한 (급격한 변화 방지)
        max_dgamma = 0.1  # 최대 각속도 제한 (rad/s)
        dgamma = np.clip(dgamma, -max_dgamma, max_dgamma)
        
        dpsi = 0  # 등자세 단계에서는 방위각 변화 없음
        
        # 위치 변화 - 수정된 좌표계 적용
        dx = V_safe * np.cos(gamma) * np.sin(psi)  # 동쪽 성분
        dy = V_safe * np.cos(gamma) * np.cos(psi)  # 북쪽 성분
        dh = V_safe * np.sin(gamma)
        
        # 연료 소모 계산
        if t <= self.burn_time:
            dW = -T * cfg.TSFC  # TSFC는 고정값으로 유지
            dM = -self.propellant_mass / self.burn_time
        else:
            dW = 0  # 연소 완료 후에는 질량 변화 없음
            dM = 0
        
        # 미분값 크기 제한 (수치적 폭발 방지)
        max_dV = 100.0      # 최대 가속도 제한
        max_velocity = 5000.0  # 최대 속도 변화율 제한
        
        dV = np.clip(dV, -max_dV, max_dV)
        dx = np.clip(dx, -max_velocity, max_velocity)
        dy = np.clip(dy, -max_velocity, max_velocity)
        dh = np.clip(dh, -max_velocity, max_velocity)
        
        # 현재 상태 저장
        if not hasattr(self, 'last_t') or t > self.last_t:
            self.last_t = t
            self.alpha_list.append(alpha * cfg.RAD_TO_DEG)  # 도 단위로 저장
            self.CD_list.append(CD)
            self.fuel_list.append(self.missile_mass - M_t)  # 소모된 연료량
            self.mach_list.append(mach)
            self.phase_list.append("등자세선회")
        
        return [dV, dgamma, dpsi, dx, dy, dh, dW, dM]
    
    def dynamics_midcourse(self, t, state):
        """중간단계(관성비행) 미사일 동역학 모델 - 수치적 안정성 개선 버전"""
        # 상태 변수
        V, gamma, psi, x, y, h, W, M_t = state
        
        # 수치적 안정성을 위한 최소값 설정
        V_min = 1.0
        M_min = 100.0
        V_safe = max(abs(V), V_min)
        M_safe = max(M_t, M_min)
        
        # 중력 계산
        g = cfg.G * cfg.R**2 / (cfg.R + h)**2
        
        # 마하수 계산을 위한 음속 계산 (고도에 따른 온도 기반)
        temperature = max(216.65, 288.15 - 0.0065 * h)  # 최소 온도 제한
        if h > 11000:  # 성층권(11km 이상)에서는 온도가 일정
            temperature = 216.65
        sound_speed = 20.05 * np.sqrt(temperature)  # 음속 (m/s)
        mach = V_safe / sound_speed if sound_speed > 0 else 0
        
        # 양력과 항력 계산 (관성 비행 단계에서는 거의 무시 가능)
        CL = 0
        
        # 대기 밀도 (고고도에서는 매우 낮음)
        rho = self.get_density(h)
        
        # 동압 계산
        q = 0.5 * rho * V_safe**2
        
        # 보간된 항력계수 사용 (외기권에서는 항력이 거의 없음)
        CD = 0
        if h < 100000:  # 100km 이하에서만 항력 적용
            CD = self.get_CD_interpolated(mach, alpha_deg=0)
            # 관성비행 단계에서는 추가적인 항력계수가 필요 없음
        
        L = q * CL * self.wing_area
        D = q * CD * self.wing_area
        
        # 추력 계산
        T = 0  # 관성 비행 단계에서는 추력이 없음
        
        # 자세 변화 (중력에 의한)
        alpha = 0  # 받음각
        phi = 0    # 경사각
        
        # 상태 미분값 계산 - 수치적 안정성 개선
        dV = (T * np.cos(alpha) - D - M_safe * g * np.sin(gamma)) / M_safe
        
        # dgamma 계산 시 분모 안전성 확보
        denominator = M_safe * V_safe
        if abs(denominator) < 1e-6:
            dgamma = 0.0
        else:
            dgamma = (T * np.sin(alpha) + L - M_safe * g * np.cos(gamma)) / denominator
        
        # 변화율 제한
        max_dgamma = 0.05  # 관성비행에서는 더 작은 제한
        dgamma = np.clip(dgamma, -max_dgamma, max_dgamma)
        
        dpsi = 0  # 관성비행에서는 방위각 변화 없음
        
        # 위치 변화 - 수정된 좌표계 적용
        dx = V_safe * np.cos(gamma) * np.sin(psi)  # 동쪽 성분
        dy = V_safe * np.cos(gamma) * np.cos(psi)  # 북쪽 성분
        dh = V_safe * np.sin(gamma)
        dW = 0  # 관성 비행 단계에서는 연료 소모 없음
        dM = 0
        
        # 현재 상태 저장
        if not hasattr(self, 'last_t') or t > self.last_t:
            self.last_t = t
            self.alpha_list.append(0)  # 관성비행 시 받음각은 0
            self.CD_list.append(CD)
            self.fuel_list.append(self.missile_mass - M_t)  # 소모된 연료량
            self.mach_list.append(mach)
            self.phase_list.append("관성비행단계")
        
        return [dV, dgamma, dpsi, dx, dy, dh, dW, dM]
    
    def dynamics_terminal(self, t, state):
        """종말단계 미사일 동역학 모델 - 수정된 버전"""
        # 상태 변수
        V, gamma, psi, x, y, h, W, M_t = state
        
        # 중력 계산
        g = cfg.G * cfg.R**2 / (cfg.R + h)**2
        
        # 마하수 계산을 위한 음속 계산 (고도에 따른 온도 기반)
        temperature = 288.15 - 0.0065 * h  # 고도에 따른 온도 감소 (표준대기모델)
        if h > 11000:  # 성층권(11km 이상)에서는 온도가 일정
            temperature = 216.65
        sound_speed = 20.05 * np.sqrt(temperature)  # 음속 (m/s)
        mach = V / sound_speed if sound_speed > 0 else 0
        
        # 대기 밀도 계산
        rho = self.get_density(h)
        
        # 동압 계산
        q = 0.5 * rho * V**2
        
        # 양력과 항력 계산 (대기권 재진입)
        CL = cfg.CL_TERMINAL
        
        # 종말 단계에서의 받음각 계산 (속도 벡터와 비행경로 사이의 각도)
        # 재진입시 비행제어를 위한 받음각 설정
        # 고도에 따라 받음각 조절 (낮은 고도에서 큰 받음각)
        alpha_deg = min(20, max(0, (50000 - h) / 5000))  # 0~20도 범위
        alpha = alpha_deg * cfg.DEG_TO_RAD
        
        # 마하수에 따른 항력계수 보정
        base_CD = self.get_CD_interpolated(mach, alpha_deg=alpha_deg)
        
        # 받음각에 따른 항력계수 보정
        CD = base_CD + 0.05 * alpha_deg/20
        
        # 재진입시 열적 효과 고려 (고속 저고도에서 항력 증가)
        if h < 60000 and mach > 3:
            heat_factor = min(2.0, 1.0 + (60000 - h) / 30000 * (mach / 10))
            CD *= heat_factor
        
        # 양력과 항력 계산
        L = q * CL * self.wing_area
        D = q * CD * self.wing_area
        
        # 추력 계산 (종말단계에서는 추력 없음)
        T = 0
        
        # 기타 변수
        phi = 0    # 경사각
        
        # 수치적 안정성을 위한 안전장치
        V_min = 1.0  # 최소 속도 1 m/s
        V_safe = max(abs(V), V_min)
        
        # 상태 미분값 계산 (수치적 안정성 개선)
        dV = (T * np.cos(alpha) - D - M_t * g * np.sin(gamma)) / M_t
        
        # velocity가 음수가 되지 않도록 안전장치 추가 (특히 terminal phase에서 중요)
        if V < 100.0 and dV < 0:
            dV = max(dV, -V/30.0)  # 속도가 낮을 때 점진적 감속
        
        # 속도가 음수가 되지 않도록 제한
        if V + dV * 0.1 < 0:  # 0.1초 후 예상 속도가 음수가 될 경우
            dV = max(dV, -V/15.0)  # 더 완만한 감속
        
        # dgamma: 분모가 0에 가까워지는 것을 방지
        dgamma_numerator = T * np.sin(alpha) + L - M_t * g * np.cos(gamma)
        dgamma_denominator = M_t * V_safe
        dgamma = dgamma_numerator / dgamma_denominator
        
        # dpsi: 종말단계에서는 방위각 변화 없음
        dpsi = 0.0
        
        # 위치 미분값 - 수정된 좌표계 적용
        dx = V * np.cos(gamma) * np.sin(psi)  # 동쪽 성분
        dy = V * np.cos(gamma) * np.cos(psi)  # 북쪽 성분
        dh = V * np.sin(gamma)
        
        # 질량 변화 (Terminal Phase에서는 연료 소모 없음)
        dW = 0
        dM = 0
        
        # 미분값 크기 제한 (수치적 폭발 방지)
        max_dV = 500.0      # 최대 가속도 500 m/s²
        max_dgamma = 0.05   # 최대 비행경로각 변화율 0.05 rad/s
        max_velocity = 8000.0  # 최대 속도 변화율 8000 m/s
        
        dV = np.clip(dV, -max_dV, max_dV)
        dgamma = np.clip(dgamma, -max_dgamma, max_dgamma)
        dx = np.clip(dx, -max_velocity, max_velocity)
        dy = np.clip(dy, -max_velocity, max_velocity)
        dh = np.clip(dh, -max_velocity, max_velocity)
        
        # 현재 상태 저장
        if not hasattr(self, 'last_t') or t > self.last_t:
            self.last_t = t
            self.alpha_list.append(alpha * cfg.RAD_TO_DEG)  # 도 단위로 저장
            self.CD_list.append(CD)
            self.fuel_list.append(cfg.MISSILE_MASS - M_t)  # 소모된 연료량
            self.mach_list.append(mach)
            self.phase_list.append("대기권재진입")
        
        return [dV, dgamma, dpsi, dx, dy, dh, dW, dM]
    
    def event_ground(self, t, y):
        """Ground impact event - 개선된 버전"""
        # y[5] is altitude
        altitude = y[5]
        
        # 고도가 음수가 되거나 매우 낮아지면 지면 충돌로 간주
        return altitude
    
    # Event 속성 설정 - 지면 충돌 시 시뮬레이션 중단
    event_ground.terminal = True  # Stop integration on ground impact
    event_ground.direction = -1   # Detect crossing zero downwards

    def event_exosphere_entry(self, t, y):
        """Exosphere entry event."""
        # y[5] is altitude
        return y[5] - cfg.EXOSPHERE_ALTITUDE
    event_exosphere_entry.terminal = False # Don't stop integration
    event_exosphere_entry.direction = 1    # Detect crossing threshold upwards

    def event_exosphere_exit(self, t, y):
        """Exosphere exit event."""
        # y[5] is altitude
        return y[5] - cfg.EXOSPHERE_ALTITUDE
    event_exosphere_exit.terminal = False # Don't stop integration
    event_exosphere_exit.direction = -1   # Detect crossing threshold downwards

    def _simulate_phase(self, dynamics_func, initial_state, t_start, t_end, phase_name=None, event_func=None):
        """Helper to run a simulation phase with optional event stopping - 수치적 안정성 개선 버전"""
        events = [event_func] if event_func else None
        
        # 단계별로 다른 solver 설정 적용
        if phase_name == "Constant Phase":
            # Constant Phase는 가장 문제가 많으므로 더 robust한 설정
            sol = solve_ivp(
                dynamics_func,
                [t_start, t_end],
                initial_state,
                method='Radau',     # 더 안정적인 implicit solver
                events=events,
                dense_output=True,
                max_step=0.1,       # 더 작은 스텝 사이즈
                rtol=1e-5,          # 다소 완화된 허용 오차
                atol=1e-8,          # 다소 완화된 허용 오차
                first_step=1e-3     # 첫 스텝 크기 명시적 설정
            )
            
            # Radau가 실패하면 RK23으로 재시도
            if not sol.success:
                print(f" Radau solver 실패, RK23으로 재시도: {sol.message}")
                sol = solve_ivp(
                    dynamics_func,
                    [t_start, t_end],
                    initial_state,
                    method='RK23',      # 더 robust한 낮은 차수 방법
                    events=events,
                    dense_output=True,
                    max_step=0.05,      # 더욱 작은 스텝
                    rtol=1e-4,          # 더 완화된 허용 오차
                    atol=1e-7
                )
        else:
            # 다른 단계들은 기존 설정 유지하되 약간 완화
            sol = solve_ivp(
                dynamics_func,
                [t_start, t_end],
                initial_state,
                method='RK45',
                events=events,
                dense_output=True,
                max_step=0.5,
                rtol=1e-6,
                atol=1e-9
            )
        
        # 시뮬레이션 성공 여부 확인
        if not sol.success:
            print(f" {phase_name or 'Unknown Phase'} 시뮬레이션 실패: {sol.message}")
            # 실패 시에도 부분 결과가 있으면 사용
            if len(sol.t) > 1:
                print(f" 부분 결과 사용: {len(sol.t)}개 시점")
                t_end_actual = sol.t[-1]
            else:
                return initial_state
        else:
            # determine actual end time if event triggered
            if events and sol.t_events and sol.t_events[0].size > 0:
                t_end_actual = sol.t_events[0][0]
                print(f" {phase_name or 'Unknown Phase'} 이벤트 감지: {t_end_actual:.2f}초")
            else:
                t_end_actual = sol.t[-1]
        
        # sample dense output at 0.1s intervals
        dt = 0.1
        t_dense = np.linspace(t_start, t_end_actual, int((t_end_actual - t_start)/dt) + 1)
        
        # dense output 처리 (실패 시 대안 제공)
        if sol.sol is not None:
            try:
                states_dense = sol.sol(t_dense).T
            except Exception as e:
                print(f" Dense output 실패: {e}, 원본 데이터 사용")
                states_dense = sol.y.T
                t_dense = sol.t
        else:
            print(f" {phase_name or 'Unknown Phase'} dense output 없음, 원본 데이터 사용")
            states_dense = sol.y.T
            t_dense = sol.t
        
        # 물리적 타당성 검증
        if len(states_dense) > 0:
            altitudes = states_dense[:, 5]  # 고도 (h)
            velocities = states_dense[:, 0]  # 속도 (V)
            
            # 비정상적인 값 검사 (경고만 출력, 시뮬레이션은 계속)
            min_altitude = np.min(altitudes)
            max_altitude = np.max(altitudes)
            min_velocity = np.min(velocities)
            max_velocity = np.max(velocities)
            
            if min_altitude < -100:  # 100m 이상 지하
                print(f" {phase_name or 'Unknown Phase'} 비정상 최소고도: {min_altitude:.1f}m")
            
            if max_altitude > 2000000:  # 2000km 이상
                print(f" {phase_name or 'Unknown Phase'} 비정상 최대고도: {max_altitude:.1f}m")
            
            if min_velocity < -100:  # 음의 속도
                print(f" {phase_name or 'Unknown Phase'} 음의 속도: {min_velocity:.1f}m/s")
            
            if max_velocity > 15000:  # 15km/s 이상
                print(f" {phase_name or 'Unknown Phase'} 비정상 최대속도: {max_velocity:.1f}m/s")
            
            # 데이터 저장
            for ti, state in zip(t_dense.tolist(), states_dense.tolist()):
                self.t.append(ti)
                self.states.append(state)
        else:
            print(f" {phase_name or 'Unknown Phase'} 유효한 데이터 없음")
            return initial_state
        
        return states_dense[-1] if len(states_dense) > 0 else initial_state
    
    def run_simulation(self, sim_time=None):
        """전체 시뮬레이션을 실행 - 개선된 버전"""
        # 초기화
        self.alpha_list = []  # 받음각 리스트 초기화
        self.CD_list = []     # 항력계수 리스트 초기화
        self.fuel_list = []   # 연료 소모량 리스트 초기화
        self.mach_list = []   # 마하수 리스트 초기화
        self.phase_list = []  # 비행 단계 리스트 초기화
        self.last_t = -1      # 마지막 시간 초기화
        self.t = []           # 시간 배열 초기화
        self.states = []      # 상태 배열 초기화
        
        # 미사일 초기화가 되어있는지 확인
        if not hasattr(self, 'initial_state'):
            print(" 시뮬레이션 초기화가 필요합니다. initialize_simulation()을 먼저 호출하세요.")
            self.initialize_simulation()
        
        print(f" 시뮬레이션 시작: {self.missile_type}")
        
        # 각 단계별 시뮬레이션 실행
        
        # 단계 1: 수직 상승 (연직단계)
        print("1️ 수직 상승 단계")
        vertical_phase_end = self._simulate_phase(
            dynamics_func=self.dynamics_vertical,
            initial_state=self.initial_state,
            t_start=0,
            t_end=cfg.VERTICAL_PHASE_TIME,
            phase_name="Vertical Phase"
        )
        
        # 단계 2: 피치 프로그램 (경사단계)
        print("2️ 피치 프로그램 단계")
        pitch_phase_end = self._simulate_phase(
            dynamics_func=self.dynamics_pitch,
            initial_state=vertical_phase_end,
            t_start=cfg.VERTICAL_PHASE_TIME,
            t_end=cfg.VERTICAL_PHASE_TIME + cfg.PITCH_PHASE_TIME,
            phase_name="Pitch Phase"
        )
        
        # 단계 3: 일정 자세 유지 (등자세단계)
        print("3️ 등자세 선회 단계")
        constant_phase_end = self._simulate_phase(
            dynamics_func=self.dynamics_constant,
            initial_state=pitch_phase_end,
            t_start=cfg.VERTICAL_PHASE_TIME + cfg.PITCH_PHASE_TIME,
            t_end=cfg.VERTICAL_PHASE_TIME + cfg.PITCH_PHASE_TIME + cfg.CONSTANT_PHASE_TIME,
            phase_name="Constant Phase"
        )
        
        # 단계 4: 중력 턴 및 관성 비행 (중력회전 및 관성비행단계)
        print("4️ 관성 비행 단계")
        midcourse_phase_end = self._simulate_phase(
            dynamics_func=self.dynamics_midcourse,
            initial_state=constant_phase_end,
            t_start=cfg.VERTICAL_PHASE_TIME + cfg.PITCH_PHASE_TIME + cfg.CONSTANT_PHASE_TIME,
            t_end=cfg.SIMULATION_END_TIME,
            phase_name="Midcourse Phase",
            event_func=self.event_ground
        )
        
        # 결과 저장
        if len(self.t) > 0 and len(self.states) > 0:
            results = {
                'time': np.array(self.t),
                'velocity': np.array([s[0] for s in self.states]),
                'gamma': np.array([s[1] for s in self.states]) * cfg.RAD_TO_DEG,
                'psi': np.array([s[2] for s in self.states]) * cfg.RAD_TO_DEG,
                'x': np.array([s[3] for s in self.states]),
                'y': np.array([s[4] for s in self.states]),
                'h': np.array([s[5] for s in self.states]),
                'weight': np.array([s[6] for s in self.states]),
                'mass': np.array([s[7] for s in self.states]),
                # 추가 분석 결과 저장
                'alpha': np.array(self.alpha_list[:len(self.t)]),
                'CD': np.array(self.CD_list[:len(self.t)]),
                'fuel': np.array(self.fuel_list[:len(self.t)]),
                'mach': np.array(self.mach_list[:len(self.t)]),
                'phase': np.array(self.phase_list[:len(self.t)])
            }
            
            # 성공적인 시뮬레이션 결과 요약
            final_range = np.sqrt(results['x'][-1]**2 + results['y'][-1]**2) / 1000  # km
            max_altitude = np.max(results['h']) / 1000  # km
            flight_time = results['time'][-1]  # seconds
            
            print(f" 시뮬레이션 완료:")
            print(f" 사거리: {final_range:.1f} km")
            print(f" 최대고도: {max_altitude:.1f} km")
            print(f" 비행시간: {flight_time:.1f} 초")
            
            self.results = results
            return results
        else:
            print(" 시뮬레이션 실패: 유효한 데이터가 생성되지 않았습니다.")
            return None

    # 원본 코드의 나머지 메서드들은 그대로 유지...
    def plot_results(self):
        """시뮬레이션 결과 그래프 출력"""
        # 비대화형 모드로 전환하여 정적 플롯 유지
        plt.ioff()
        
        # self.states가 있을 경우 시뮬레이션 이력으로 results 재구성
        if hasattr(self, 'states') and self.states:
            time_arr = np.array(self.t)
            states_arr = np.array(self.states)
            self.results = {
                'time': time_arr,
                'velocity': states_arr[:, 0],
                'gamma': states_arr[:, 1] * cfg.RAD_TO_DEG,
                'psi': states_arr[:, 2] * cfg.RAD_TO_DEG,
                'x': states_arr[:, 3],
                'y': states_arr[:, 4],
                'h': states_arr[:, 5],
                'weight': states_arr[:, 6],
                'mass': states_arr[:, 7],
                'alpha': np.array(self.alpha_list),
                'CD': np.array(self.CD_list),
                'fuel': np.array(self.fuel_list),
                'mach': np.array(self.mach_list),
                'phase': np.array(self.phase_list)
            }
        elif not self.results:
            print("시뮬레이션 결과가 없습니다.")
            return
        
        # 리스트 타입 결과를 numpy array로 변환하여 연산 지원
        for key, val in self.results.items():
            if not isinstance(val, np.ndarray):
                self.results[key] = np.array(val)
        
        # 그래프 1: 속도 및 자세각
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(self.results['time'], self.results['velocity'])
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(self.results['time'], self.results['gamma'])
        plt.xlabel('Time (s)')
        plt.ylabel('Pitch Angle (deg)')
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(self.results['time'], self.results['psi'])
        plt.xlabel('Time (s)')
        plt.ylabel('Azimuth Angle (deg)')
        plt.grid(True)
        plt.tight_layout()
        
        # 그래프 2: 위치 및 질량
        plt.figure(figsize=(12, 10))
        plt.subplot(4, 1, 1)
        plt.plot(self.results['time'], self.results['x'])
        plt.xlabel('Time (s)')
        plt.ylabel('X Position (m)')
        plt.grid(True)
        
        plt.subplot(4, 1, 2)
        plt.plot(self.results['time'], self.results['y'])
        plt.xlabel('Time (s)')
        plt.ylabel('Y Position (m)')
        plt.grid(True)
        
        plt.subplot(4, 1, 3)
        plt.plot(self.results['time'], self.results['h'])
        plt.xlabel('Time (s)')
        plt.ylabel('Altitude (m)')
        plt.grid(True)
        
        plt.subplot(4, 1, 4)
        plt.plot(self.results['time'], self.results['mass'])
        plt.xlabel('Time (s)')
        plt.ylabel('Mass (kg)')
        plt.grid(True)
        plt.tight_layout()
        
        # 그래프 3: 3D 궤적
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.results['x'], self.results['y'], self.results['h'])
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Altitude (m)')
        ax.set_title('Missile Trajectory 3D Visualization')
        
        plt.show(block=True)
    
    def run_simulation_realtime(self):
        """실시간 시각화와 함께 전체 시뮬레이션을 실행"""
        print("실시간 시각화와 함께 시뮬레이션을 시작합니다...")
        # 시뮬레이션 시간 설정
        self.sim_time = cfg.SIM_TIME
        
        # 대화형 모드 활성화
        plt.ion()
        
        # 3D 그래프 설정
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 추적 데이터 초기화
        trajectory_x = []
        trajectory_y = []
        trajectory_z = []
        
        # 상태 표시 텍스트
        status_text = fig.text(0.02, 0.02, "", transform=fig.transFigure)
        
        # 초기 상태 정의
        self.initialize_simulation()
        
        # 1. 수직 상승 단계
        print("1단계: 수직 상승")
        t_vertical_end = self.vertical_time
        sol_vertical = solve_ivp(
            self.dynamics_vertical,
            [0, t_vertical_end],
            self.initial_state,
            method='RK45',
            dense_output=True
        )
        
        # 결과 보간 및 저장
        t_dense = np.linspace(0, t_vertical_end, int(t_vertical_end/0.1)+1)
        if hasattr(sol_vertical, 'sol') and sol_vertical.sol is not None:
            states_dense = sol_vertical.sol(t_dense).T
            
            self.t = t_dense.tolist()
            self.states = states_dense.tolist()
        
        for i in range(len(t_dense)):
            trajectory_x.append(states_dense[i, 3])
            trajectory_y.append(states_dense[i, 4])
            trajectory_z.append(states_dense[i, 5])
            
            # 10개 데이터마다 그래프 업데이트 (성능 최적화)
            if i % 10 == 0 or i == len(t_dense) - 1:
                ax.clear()
                ax.plot(trajectory_x, trajectory_y, trajectory_z, 'b-', alpha=0.7, linewidth=2)
                ax.plot([trajectory_x[-1]], [trajectory_y[-1]], [trajectory_z[-1]], 'ro', markersize=8)
                
                # 축 범위 자동 조정
                ax.set_xlim([min(min(trajectory_x), -10), max(max(trajectory_x)*1.1, 10)])
                ax.set_ylim([min(min(trajectory_y), -10), max(max(trajectory_y)*1.1, 10)])
                ax.set_zlim([0, max(max(trajectory_z)*1.1, 10)])
                
                # 축 레이블 설정
                ax.set_xlabel('X Position (m)')
                ax.set_ylabel('Y Position (m)')
                ax.set_zlabel('Altitude (m)')
                
                # 정보 업데이트
                velocity = states_dense[i, 0]
                altitude = states_dense[i, 5]
                plt.title(f'Missile Trajectory 3D Realtime Visualization\nTime: {t_dense[i]:.1f} s, Velocity: {velocity:.1f} m/s, Altitude: {altitude/1000:.2f} km')
                
                # 화면 갱신
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.01)  # 짧은 대기 시간
        
        # 2. 피치 프로그램 단계 (조건부 실행)
        last_state_after_vertical = sol_vertical.y[:, -1] # 수직 상승 후 최종 상태 저장
        last_state_before_constant = last_state_after_vertical # 기본값: 수직 상승 후 상태
        t_constant_start = t_vertical_end # 기본값: 수직 상승 종료 시간
        
        if self.pitch_time > 1e-6: # PITCH_TIME이 0보다 큰 경우에만 실행 (부동소수점 오차 고려)
            print("2단계: 피치 프로그램")
            t_pitch_start = t_vertical_end
            t_pitch_end = t_pitch_start + self.pitch_time
            
            sol_pitch = solve_ivp(
                self.dynamics_pitch,
                [t_pitch_start, t_pitch_end],
                last_state_after_vertical,
                method='RK45',
                dense_output=True
            )
            # 결과 보간 및 저장
            t_dense = np.linspace(t_pitch_start, t_pitch_end, int((t_pitch_end-t_pitch_start)/0.1)+1)
            if hasattr(sol_pitch, 'sol') and sol_pitch.sol is not None:
                states_dense = sol_pitch.sol(t_dense).T
                
                self.t.extend(t_dense.tolist())
                self.states.extend(states_dense.tolist())
                
                # 피치 단계 실시간 시각화
                for i in range(len(t_dense)):
                    trajectory_x.append(states_dense[i, 3])
                    trajectory_y.append(states_dense[i, 4])
                    trajectory_z.append(states_dense[i, 5])
                    
                    if i % 10 == 0 or i == len(t_dense) - 1:
                        ax.clear()
                        ax.plot(trajectory_x, trajectory_y, trajectory_z, 'b-', alpha=0.7, linewidth=2)
                        ax.plot([trajectory_x[-1]], [trajectory_y[-1]], [trajectory_z[-1]], 'ro', markersize=8)
                        
                        ax.set_xlim([min(min(trajectory_x), -10), max(max(trajectory_x)*1.1, 10)])
                        ax.set_ylim([min(min(trajectory_y), -10), max(max(trajectory_y)*1.1, 10)])
                        ax.set_zlim([0, max(max(trajectory_z)*1.1, 10)])
                        
                        ax.set_xlabel('X Position (m)')
                        ax.set_ylabel('Y Position (m)')
                        ax.set_zlabel('Altitude (m)')
                        
                        velocity = states_dense[i, 0]
                        altitude = states_dense[i, 5]
                        plt.title(f'Missile Trajectory 3D Realtime Visualization\nTime: {t_dense[i]:.1f} s, Velocity: {velocity:.1f} m/s, Altitude: {altitude/1000:.2f} km')
                        
                        fig.canvas.draw_idle()
                        fig.canvas.flush_events()
                        plt.pause(0.01)
                        
            last_state_before_constant = sol_pitch.y[:, -1] # 피치 프로그램 후 상태 업데이트
            t_constant_start = t_pitch_end # 다음 단계 시작 시간 업데이트
        else:
            print("2단계: 피치 프로그램 (PITCH_TIME이 0이므로 건너뜁니다.)")
        
        # 3. 등자세 선회 단계
        print("3단계: 등자세 선회")
        t_constant_end = self.burn_time
        constant_initial = last_state_before_constant
        
        if t_constant_start < t_constant_end:
            sol_constant = solve_ivp(
                self.dynamics_constant,
                [t_constant_start, t_constant_end],
                constant_initial,
                method='RK45',
                dense_output=True
            )
            # 결과 보간 및 저장
            t_dense = np.linspace(t_constant_start, t_constant_end, int((t_constant_end-t_constant_start)/0.1)+1)
            if hasattr(sol_constant, 'sol') and sol_constant.sol is not None:
                states_dense = sol_constant.sol(t_dense).T
                
                self.t.extend(t_dense.tolist())
                self.states.extend(states_dense.tolist())
                
                # 등자세 선회 단계 실시간 시각화
                for i in range(len(t_dense)):
                    trajectory_x.append(states_dense[i, 3])
                    trajectory_y.append(states_dense[i, 4])
                    trajectory_z.append(states_dense[i, 5])
                    
                    if i % 10 == 0 or i == len(t_dense) - 1:
                        ax.clear()
                        ax.plot(trajectory_x, trajectory_y, trajectory_z, 'b-', alpha=0.7, linewidth=2)
                        ax.plot([trajectory_x[-1]], [trajectory_y[-1]], [trajectory_z[-1]], 'ro', markersize=8)
                        
                        ax.set_xlim([min(min(trajectory_x), -10), max(max(trajectory_x)*1.1, 10)])
                        ax.set_ylim([min(min(trajectory_y), -10), max(max(trajectory_y)*1.1, 10)])
                        ax.set_zlim([0, max(max(trajectory_z)*1.1, 10)])
                        
                        ax.set_xlabel('X Position (m)')
                        ax.set_ylabel('Y Position (m)')
                        ax.set_zlabel('Altitude (m)')
                        
                        velocity = states_dense[i, 0]
                        altitude = states_dense[i, 5]
                        plt.title(f'Missile Trajectory 3D Realtime Visualization\nTime: {t_dense[i]:.1f} s, Velocity: {velocity:.1f} m/s, Altitude: {altitude/1000:.2f} km')
                        
                        fig.canvas.draw_idle()
                        fig.canvas.flush_events()
                        plt.pause(0.01)
                        
            last_state_after_constant = sol_constant.y[:, -1]
        else:
            last_state_after_constant = constant_initial
        
        # 4. 중간단계(관성비행)
        print("4단계: 중간단계 비행")
        t_mid_start = t_constant_end
        t_mid_end = self.sim_time
        
        # 중간 단계 통합 (지면 충돌 이벤트 감지 포함)
        sol_mid = solve_ivp(
            self.dynamics_midcourse,
            [t_mid_start, t_mid_end],
            last_state_after_constant,
            method='RK45',
            events=[self.event_ground],
            dense_output=True
        )
        
        # 지면 충돌 발생 시 충돌 시점까지 데이터 보간 및 시각화 후 종료
        collision_times = sol_mid.t_events[0] if sol_mid.t_events else []
        if collision_times:
            t_ground = collision_times[0]
            print(f"지면 충돌 감지: {t_ground:.2f}초")
            t_dense = np.linspace(t_mid_start, t_ground, int((t_ground - t_mid_start) / 0.1) + 1)
            if sol_mid.sol is not None:
                states_dense = sol_mid.sol(t_dense).T
                self.t.extend(t_dense.tolist())
                self.states.extend(states_dense.tolist())
                
                for i in range(len(t_dense)):
                    trajectory_x.append(states_dense[i, 3])
                    trajectory_y.append(states_dense[i, 4])
                    trajectory_z.append(states_dense[i, 5])
                    
                    if i % 10 == 0 or i == len(t_dense) - 1:
                        ax.clear()
                        ax.plot(trajectory_x, trajectory_y, trajectory_z, 'b-', alpha=0.7, linewidth=2)
                        ax.plot([trajectory_x[-1]], [trajectory_y[-1]], [trajectory_z[-1]], 'ro', markersize=8)
                        
                        ax.set_xlim([min(min(trajectory_x), -10), max(max(trajectory_x)*1.1, 10)])
                        ax.set_ylim([min(min(trajectory_y), -10), max(max(trajectory_y)*1.1, 10)])
                        ax.set_zlim([0, max(max(trajectory_z)*1.1, 10)])
                        
                        ax.set_xlabel('X Position (m)')
                        ax.set_ylabel('Y Position (m)')
                        ax.set_zlabel('Altitude (m)')
                        
                        velocity = states_dense[i, 0]
                        altitude = states_dense[i, 5]
                        plt.title(f'Missile Trajectory 3D Realtime Visualization\nTime: {t_dense[i]:.1f} s, Velocity: {velocity:.1f} m/s, Altitude: {altitude/1000:.2f} km')
                        
                        fig.canvas.draw_idle()
                        fig.canvas.flush_events()
                        plt.pause(0.01)
                        
            print("시뮬레이션 종료: 지면 충돌")
            
            # 최종 결과 저장
            self.results = {
                'time': np.array(self.t),
                'velocity': np.array([s[0] for s in self.states]),
                'gamma': np.array([s[1] for s in self.states]),
                'psi': np.array([s[2] for s in self.states]),
                'x': np.array([s[3] for s in self.states]),
                'y': np.array([s[4] for s in self.states]),
                'h': np.array([s[5] for s in self.states]),
                'weight': np.array([s[6] for s in self.states]),
                'mass': np.array([s[7] for s in self.states]),
                'alpha': np.array(self.alpha_list),
                'CD': np.array(self.CD_list),
                'fuel': np.array(self.fuel_list),
                'mach': np.array(self.mach_list),
                'phase': np.array(self.phase_list)
            }
            
            # 대화형 모드 비활성화
            plt.ioff()
            plt.show(block=True)
            return self.results
        
        # 충돌이 없는 경우 전체 구간 시각화
        t_dense = np.linspace(t_mid_start, t_mid_end, int((t_mid_end - t_mid_start) / 0.1) + 1)
        if sol_mid.sol is not None:
            states_dense = sol_mid.sol(t_dense).T
            self.t.extend(t_dense.tolist())
            self.states.extend(states_dense.tolist())
            
            for i in range(len(t_dense)):
                trajectory_x.append(states_dense[i, 3])
                trajectory_y.append(states_dense[i, 4])
                trajectory_z.append(states_dense[i, 5])
                
                if i % 10 == 0 or i == len(t_dense) - 1:
                    ax.clear()
                    ax.plot(trajectory_x, trajectory_y, trajectory_z, 'b-', alpha=0.7, linewidth=2)
                    ax.plot([trajectory_x[-1]], [trajectory_y[-1]], [trajectory_z[-1]], 'ro', markersize=8)
                    
                    ax.set_xlim([min(min(trajectory_x), -10), max(max(trajectory_x)*1.1, 10)])
                    ax.set_ylim([min(min(trajectory_y), -10), max(max(trajectory_y)*1.1, 10)])
                    ax.set_zlim([0, max(max(trajectory_z)*1.1, 10)])
                    
                    ax.set_xlabel('X Position (m)')
                    ax.set_ylabel('Y Position (m)')
                    ax.set_zlabel('Altitude (m)')
                    
                    velocity = states_dense[i, 0]
                    altitude = states_dense[i, 5]
                    plt.title(f'Missile Trajectory 3D Realtime Visualization\nTime: {t_dense[i]:.1f} s, Velocity: {velocity:.1f} m/s, Altitude: {altitude/1000:.2f} km')
                    
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
                    plt.pause(0.01)
        
        print(f"시뮬레이션 계산 완료! 전체 비행 시간: {self.t[-1]:.2f}초")
        
        # 최종 결과 저장
        self.results = {
            'time': np.array(self.t),
            'velocity': np.array([s[0] for s in self.states]),
            'gamma': np.array([s[1] for s in self.states]),
            'psi': np.array([s[2] for s in self.states]),
            'x': np.array([s[3] for s in self.states]),
            'y': np.array([s[4] for s in self.states]),
            'h': np.array([s[5] for s in self.states]),
            'weight': np.array([s[6] for s in self.states]),
            'mass': np.array([s[7] for s in self.states]),
            'alpha': np.array(self.alpha_list),
            'CD': np.array(self.CD_list),
            'fuel': np.array(self.fuel_list),
            'mach': np.array(self.mach_list),
            'phase': np.array(self.phase_list)
        }
        
        # 대화형 모드 비활성화
        plt.ioff()
        plt.show(block=True)
        return self.results

def main():
    """메인 함수"""
    print("미사일 궤적 시뮬레이션을 시작합니다...")
    
    # 미사일 시뮬레이션 객체 생성
    simulation = MissileSimulation()
    
    # 실행 모드 선택
    print("실행 모드를 선택하세요:")
    print("1. 실시간 3D 궤적 시뮬레이션 - 미사일의 전체 비행 궤적을 3D 애니메이션으로 시각화합니다.")
    print("2. 상세 결과 그래프 - 시뮬레이션 결과 데이터를 바탕으로 다양한 물리량의 변화를 분석 그래프로 제공합니다.")
    
    mode = input("모드 선택 (1-2, 기본값: 1): ")
    
    if mode == "2":
        # 시뮬레이션 실행 후 상세 그래프 출력
        simulation.initialize_simulation()
        simulation.run_simulation()
        simulation.plot_results()
    else:
        # 실시간 3D 시뮬레이션 (기본 모드)
        simulation.run_simulation_realtime()
    
    print("미사일 궤적 시뮬레이션이 완료되었습니다.")

if __name__ == "__main__":
    main()
