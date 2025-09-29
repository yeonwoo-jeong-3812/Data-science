# 개선된 config.py - Physics-Informed Neural ODE와 호환성 강화
import numpy as np
import torch

# 단위 환산
DEG_TO_RAD = np.pi / 180
RAD_TO_DEG = 180 / np.pi

# 물리 상수 (더 정확한 값)
G = 9.80665  # 표준 중력가속도 (m/s²) - WGS84 기준
R_EARTH = 6371000  # 지구 평균 반지름 (m) - WGS84 기준
R = R_EARTH  # main.py 호환성을 위해 추가
GM_EARTH = 3.986004418e14  # 지구 중력 매개변수 (m³/s²)

# 대기 상수
AIR_GAS_CONSTANT = 287.0531  # 건조 공기 기체상수 (J/(kg·K))
STD_TEMPERATURE_SEA_LEVEL = 288.15  # 해면 표준온도 (K)
STD_PRESSURE_SEA_LEVEL = 101325.0  # 해면 표준기압 (Pa)
STD_DENSITY_SEA_LEVEL = 1.225  # 해면 표준밀도 (kg/m³)
EXOSPHERE_HEIGHT = 600000  # 외기권 시작 고도 (m)

# Neural ODE와 호환되는 물리 함수들
class PhysicsUtils:
    """Physics-Informed Neural ODE와 호환되는 물리 유틸리티"""
    
    @staticmethod
    def gravity_at_altitude(h):
        """고도에 따른 중력 계산 (더 정확한 모델)"""
        return G * (R_EARTH / (R_EARTH + h))**2
    
    @staticmethod
    def atmospheric_density(h):
        """표준 대기 모델 기반 밀도 계산"""
        if h < 0:
            return STD_DENSITY_SEA_LEVEL
        
        # 11km 이하 (대류권)
        if h <= 11000:
            T = STD_TEMPERATURE_SEA_LEVEL - 0.0065 * h
            P = STD_PRESSURE_SEA_LEVEL * (T / STD_TEMPERATURE_SEA_LEVEL)**5.2561
        # 11-20km (성층권 하부)
        elif h <= 20000:
            T = 216.65  # 일정 온도
            P = 22632.1 * np.exp(-0.00015768 * (h - 11000))
        # 20km 이상 (근사)
        else:
            T = 216.65 + 0.001 * (h - 20000)
            P = 5474.89 * (T / 216.65)**(-34.163)
        
        return P / (AIR_GAS_CONSTANT * T)
    
    @staticmethod
    def sound_speed(h):
        """고도에 따른 음속 계산"""
        if h <= 11000:
            T = STD_TEMPERATURE_SEA_LEVEL - 0.0065 * h
        elif h <= 20000:
            T = 216.65
        else:
            T = 216.65 + 0.001 * (h - 20000)
        
        gamma = 1.4  # 비열비
        R = AIR_GAS_CONSTANT
        return np.sqrt(gamma * R * T)
    
    @staticmethod
    def mach_number(velocity, altitude):
        """마하수 계산"""
        sound_speed = PhysicsUtils.sound_speed(altitude)
        return velocity / sound_speed
    
    @staticmethod
    def drag_coefficient_model(mach, alpha_deg=0):
        """Neural ODE 호환 항력계수 모델"""
        # 기본 마하수별 항력계수 (연속 함수로 모델링)
        cd_base = 0.2 + 0.3 * np.exp(-((mach - 1.2) / 0.8)**2)
        
        # 받음각 효과
        alpha_rad = alpha_deg * DEG_TO_RAD
        cd_alpha = 0.1 * np.sin(2 * alpha_rad)**2
        
        return cd_base + cd_alpha

# Neural ODE 상태 벡터 정의
class StateVector:
    """Neural ODE 상태 벡터 관리 클래스"""
    
    # 상태 인덱스 정의
    VELOCITY = 0      # 속도 (m/s)
    GAMMA = 1         # 피치각 (rad)
    PSI = 2           # 방위각 (rad)
    X = 3             # X 위치 (m)
    Y = 4             # Y 위치 (m)
    H = 5             # 고도 (m)
    MASS = 6          # 질량 (kg)
    FUEL_CONSUMED = 7 # 연료소모 (kg)
    MACH = 8          # 마하수
    RHO = 9           # 대기밀도 (kg/m³)
    
    STATE_DIM = 10
    
    STATE_NAMES = [
        'velocity', 'gamma_rad', 'psi_rad', 'x', 'y', 'h',
        'mass', 'fuel_consumed', 'mach', 'atmospheric_density'
    ]
    
    STATE_UNITS = [
        'm/s', 'rad', 'rad', 'm', 'm', 'm',
        'kg', 'kg', '-', 'kg/m³'
    ]
    
    @staticmethod
    def create_initial_state(missile_info, launch_angle_deg, launch_azimuth_deg=90):
        """초기 상태 벡터 생성"""
        initial_state = np.zeros(StateVector.STATE_DIM)
        
        initial_state[StateVector.VELOCITY] = 0.0  # 초기 속도
        initial_state[StateVector.GAMMA] = np.deg2rad(launch_angle_deg)
        initial_state[StateVector.PSI] = np.deg2rad(launch_azimuth_deg)
        initial_state[StateVector.X] = 0.0
        initial_state[StateVector.Y] = 0.0
        initial_state[StateVector.H] = 0.0
        initial_state[StateVector.MASS] = missile_info["launch_weight"]
        initial_state[StateVector.FUEL_CONSUMED] = 0.0
        initial_state[StateVector.MACH] = 0.0
        initial_state[StateVector.RHO] = PhysicsUtils.atmospheric_density(0.0)
        
        return initial_state
    
    @staticmethod
    def validate_state(state):
        """상태 벡터 유효성 검증"""
        if len(state) != StateVector.STATE_DIM:
            raise ValueError(f"상태 벡터 차원 오류: {len(state)} != {StateVector.STATE_DIM}")
        
        # 물리적 제약 검증
        velocity = state[StateVector.VELOCITY]
        altitude = state[StateVector.H]
        mass = state[StateVector.MASS]
        
        if velocity < 0:
            raise ValueError("속도는 음수가 될 수 없습니다")
        if altitude < -100:  # 해수면 아래 100m까지 허용
            raise ValueError("고도가 너무 낮습니다")
        if mass <= 0:
            raise ValueError("질량은 양수여야 합니다")
        
        return True

# 개선된 미사일 정보 (Neural ODE 호환)
ENHANCED_MISSILE_TYPES = {
    "SCUD-B": {
        # 기본 정보
        "name": "SCUD-B",
        "diameter": 0.88,
        "length": 10.94,
        "nozzle_diameter": 0.6,  # 노즐 지름 추가
        "launch_weight": 5860,
        "payload": 985,
        "propellant_mass": 4875,
        "range_km": 300,
        
        # 추진 정보
        "propellant_type": "LIQUID",
        "isp_sea": 230,
        "isp_vac": 258,
        "burn_time": 65,
        
        # 비행 프로그램
        "vertical_time": 10,
        "pitch_time": 15,
        "pitch_angle_deg": 20,
        
        # 공기역학
        "reference_area": np.pi * (0.88/2)**2,
        "cd_base": 0.25,
        
        # Neural ODE 호환 함수들
        "thrust_profile": lambda t: 5860 * 9.81 * 230 / 65 if t < 65 else 0,
        "mass_flow_rate": lambda t: 4875 / 65 if t < 65 else 0,
        "drag_model": lambda mach, alpha: PhysicsUtils.drag_coefficient_model(mach, alpha),
    },
    
    "NODONG": {
        "name": "노동 1호",
        "diameter": 1.36,
        "length": 16.4,
        "nozzle_diameter": 0.8,
        "launch_weight": 16500,
        "payload": 1200,
        "propellant_mass": 15300,
        "range_km": 1500,
        
        "propellant_type": "UDMH/RFNA",
        "isp_sea": 255,  # 해수면 비추력
        "isp_vacuum": 280,  # 진공 비추력 (향상)
        
        "reference_area": np.pi * (1.36/2)**2,  # 미사일 단면적 = 1.45 m²
        
        "vertical_time": 10,
        "pitch_time": 20,  # 피치 시간 조정
        "pitch_angle_deg": 15,  # 피치각 최적화
        "burn_time": 70,  # 연소 시간 증가
        
        # 추력 프로파일 최적화 - 초기 추력 증가
        "thrust_profile": lambda t: 16500 * 9.81 * 280 / 70 * (1.2 if t < 15 else 1.0) if t < 70 else 0,
        "mass_flow_rate": lambda t: 15300 / 70 if t < 70 else 0,
        "drag_model": lambda mach, alpha: PhysicsUtils.drag_coefficient_model(mach, alpha) * 0.9,  # 항력계수 최적화
    },
    
    "KN-23": {
        "name": "KN-23",
        "diameter": 0.95,
        "length": 7.5,
        "nozzle_diameter": 0.5,
        "launch_weight": 3415,
        "payload": 500,
        "propellant_mass": 2915,
        "range_km": 690,
        
        "propellant_type": "SOLID",
        "isp_sea": 260,
        "isp_vac": 265,
        "burn_time": 40,
        
        "vertical_time": 6,
        "pitch_time": 10,
        "pitch_angle_deg": 25,
        
        "reference_area": np.pi * (0.95/2)**2,
        "cd_base": 0.28,
        
        "thrust_profile": lambda t: 3415 * 9.81 * 260 / 40 if t < 40 else 0,
        "mass_flow_rate": lambda t: 2915 / 40 if t < 40 else 0,
        "drag_model": lambda mach, alpha: PhysicsUtils.drag_coefficient_model(mach, alpha),
    }
}

# Neural ODE 데이터셋 구성
NEURAL_ODE_DATASET_CONFIG = {
    "training_missiles": ["SCUD-B", "NODONG"],
    "validation_missiles": ["KN-23"],
    "launch_angles": [25, 30, 35, 40, 45, 50, 55, 60, 65],
    "simulation_time": 2000,
    "time_step": 0.1,
    "segment_length": 30,
    "overlap_ratio": 0.5,
    "physics_weight": 0.1,
    "data_validation": True,
    "save_metadata": True
}

# PyTorch 호환 함수들
def get_physics_constants():
    """Neural ODE에서 사용할 물리 상수 반환"""
    return {
        'g': G,
        'earth_radius': R_EARTH,
        'air_density_sea_level': STD_DENSITY_SEA_LEVEL,
        'sound_speed_sea_level': PhysicsUtils.sound_speed(0)
    }

def create_missile_tensor(missile_type, device='cpu'):
    """미사일 정보를 PyTorch 텐서로 변환"""
    if missile_type not in ENHANCED_MISSILE_TYPES:
        raise ValueError(f"Unknown missile type: {missile_type}")
    
    info = ENHANCED_MISSILE_TYPES[missile_type]
    
    # 미사일 매개변수 텐서
    params = torch.tensor([
        info["launch_weight"],
        info["propellant_mass"],
        info["isp_sea"],
        info["burn_time"],
        info["reference_area"],
        info["cd_base"]
    ], dtype=torch.float32, device=device)
    
    return params

def get_enhanced_missile_info(missile_type):
    """개선된 미사일 정보 반환"""
    if missile_type not in ENHANCED_MISSILE_TYPES:
        raise ValueError(f"Unknown missile type: {missile_type}")
    return ENHANCED_MISSILE_TYPES[missile_type]

def get_cd_table_for_missile(missile_name):
    """미사일별 항력계수 테이블 반환"""
    # 기본 항력계수 테이블 (마하수별)
    default_cd_table = {
        'mach': np.array([0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0]),
        'cd': np.array([0.3, 0.35, 0.45, 0.8, 0.6, 0.4, 0.35, 0.3, 0.28, 0.25])
    }
    
    # 미사일별 특화된 항력계수 테이블
    missile_cd_tables = {
        'SCUD-B': {
            'mach': np.array([0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0]),
            'cd': np.array([0.32, 0.36, 0.47, 0.82, 0.62, 0.42, 0.37, 0.32, 0.30, 0.27])
        },
        'NODONG': {
            'mach': np.array([0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0]),
            'cd': np.array([0.28, 0.33, 0.43, 0.78, 0.58, 0.38, 0.33, 0.28, 0.26, 0.23])
        },
        'KN-23': {
            'mach': np.array([0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0]),
            'cd': np.array([0.35, 0.38, 0.48, 0.85, 0.65, 0.45, 0.40, 0.35, 0.33, 0.30])
        }
    }
    
    return missile_cd_tables.get(missile_name, default_cd_table)

# 역호환성을 위한 함수들
def set_missile_type(missile_type_key):
    """기존 main.py와의 호환성을 위한 함수"""
    global MISSILE_MASS, MISSILE_WEIGHT, PROPELLANT_MASS, ISP, WING_AREA
    global VERTICAL_TIME, PITCH_TIME, PITCH_ANGLE_DEG, BURN_TIME
    
    if missile_type_key not in ENHANCED_MISSILE_TYPES:
        print(f"오류: 미사일 유형 '{missile_type_key}'을(를) 찾을 수 없습니다.")
        return False
    
    missile_info = ENHANCED_MISSILE_TYPES[missile_type_key]
    
    # 전역 변수 업데이트 (기존 코드 호환성)
    global VERTICAL_PHASE_TIME, PITCH_PHASE_TIME, CONSTANT_PHASE_TIME
    
    MISSILE_MASS = missile_info["launch_weight"]
    MISSILE_WEIGHT = MISSILE_MASS * G
    PROPELLANT_MASS = missile_info["propellant_mass"]
    ISP = missile_info["isp_sea"]
    WING_AREA = missile_info["reference_area"]
    
    # 기본 시간 변수 설정
    VERTICAL_TIME = missile_info["vertical_time"]
    PITCH_TIME = missile_info["pitch_time"]
    PITCH_ANGLE_DEG = missile_info["pitch_angle_deg"]
    BURN_TIME = missile_info["burn_time"]
    
    # main.py에서 사용하는 변수들에도 동일하게 적용
    VERTICAL_PHASE_TIME = VERTICAL_TIME
    PITCH_PHASE_TIME = PITCH_TIME
    # CONSTANT_PHASE_TIME 계산 (연소시간 - 수직시간 - 피치시간)
    CONSTANT_PHASE_TIME = max(0, BURN_TIME - VERTICAL_TIME - PITCH_TIME)
    
    print(f"미사일 유형이 '{missile_info['name']}'(으)로 설정되었습니다.")
    return True

def get_available_missile_types():
    """사용 가능한 미사일 유형 목록 반환"""
    return list(ENHANCED_MISSILE_TYPES.keys())

# 기존 변수들 (역호환성) - 기본값은 SCUD-B에서# 기존 변수들 (역호환성)
# 기본값으로 SCUD-B 미사일 파라미터를 사용
# 이 값들은 set_missile_type() 함수를 통해 동적으로 변경됨

# 기본 값을 SCUD-B로 초기화
default_missile = ENHANCED_MISSILE_TYPES["SCUD-B"]
MISSILE_MASS = default_missile["launch_weight"] 
MISSILE_WEIGHT = MISSILE_MASS * G
PROPELLANT_MASS = default_missile["propellant_mass"]
ISP = default_missile["isp_sea"]
WING_AREA = default_missile["reference_area"]

VERTICAL_TIME = default_missile["vertical_time"]
PITCH_TIME = default_missile["pitch_time"]
PITCH_ANGLE_DEG = default_missile["pitch_angle_deg"]
BURN_TIME = default_missile["burn_time"]

# main.py와 호환성을 위한 변수들
VERTICAL_PHASE_TIME = VERTICAL_TIME
PITCH_PHASE_TIME = PITCH_TIME
CONSTANT_PHASE_TIME = max(0, BURN_TIME - VERTICAL_TIME - PITCH_TIME)

# 시뮬레이션 설정
INTERVAL = 0.1
SIM_TIME = 2000
SIMULATION_END_TIME = SIM_TIME

# 기존 미사일 타입 호환성
MISSILE_TYPES = ENHANCED_MISSILE_TYPES

# main.py에서 필요한 변수들 추가
STD_DENSITY_SEA_LEVEL = 1.225  # kg/m³
STD_PRESSURE_SEA_LEVEL = 101325  # Pa
STD_GRAVITY = 9.80665  # m/s²
AIR_MOLAR_MASS = 0.0289644  # kg/mol
UNIVERSAL_GAS_CONSTANT = 8.314462  # J/(mol·K)
AIR_GAS_CONSTANT = 287.058  # J/(kg·K)

# 대기층 정의 (고도, 온도 기울기 등)
ATMOSPHERIC_LAYERS = [
    (0, 11000, -0.0065, 0, 288.15),
    (11000, 20000, 0, 11000, 216.65),
    (20000, 32000, 0.001, 20000, 216.65),
    (32000, 47000, 0.0028, 32000, 228.65),
    (47000, 51000, 0, 47000, 270.65),
    (51000, 71000, -0.0028, 51000, 270.65),
    (71000, 84852, -0.002, 71000, 214.65)
]

# 고고도 참조 데이터 (고도, 밀도, scale_height)
HIGH_ALTITUDE_REFERENCE = [
    (100000, 0.0000552, 8500),
    (120000, 0.0000024, 9000),
    (150000, 0.0000002, 10000)
]

# 공력 계수들
CL_VERTICAL = 0.0
CL_PITCH = 0.5
CL_CONSTANT = 0.3
CL_TERMINAL = 0.1
K = 0.05
WING_AREA = 0.2  # m²

# 추진 관련
ISP = 250  # 비추력 (s)
TSFC = 0.0004  # 추력비연료소모율 (kg/N/s)

# 위상 시간 정의
PITCH_PHASE_TIME = PITCH_TIME
CONSTANT_PHASE_TIME = BURN_TIME - VERTICAL_TIME - PITCH_TIME

# 고도 상수
EXOSPHERE_ALTITUDE = 600000  # m

# 각도 변환
DEG_TO_RAD = np.pi / 180
RAD_TO_DEG = 180 / np.pi

# 기본 항력계수 테이블
BASE_CD_TABLE = {0: 0.3, 1: 0.8, 2: 1.2, 3: 1.0, 4: 0.9, 5: 0.8}

# main.py에서 필요한 함수
def calculate_cd_table(diameter, length, nozzle_diameter, propellant_type):
    """항력계수 테이블 계산 (main.py 호환성)"""
    # 간단한 항력계수 테이블 반환 (dictionary 형태)
    mach_numbers = np.linspace(0, 5, 51)
    cd_values = np.array([PhysicsUtils.drag_coefficient_model(m) for m in mach_numbers])
    return dict(zip(mach_numbers, cd_values))

def get_available_missile_types():
    """사용 가능한 미사일 타입 목록 반환"""
    return list(ENHANCED_MISSILE_TYPES.keys())

def set_missile_type(missile_type):
    """미사일 타입 설정 (호환성 함수)"""
    if missile_type in ENHANCED_MISSILE_TYPES:
        return True
    return False