# 개선된 config.py - Physics-Informed Neural ODE와 6DoF 호환성 강화
import numpy as np
import torch
import math

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

# Neural ODE와 6DoF와 호환되는 물리 함수들
class PhysicsUtils:
    """Physics-Informed Neural ODE와 6DoF와 호환되는 물리 유틸리티"""
    
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
        R_const = AIR_GAS_CONSTANT
        return np.sqrt(gamma * R_const * T)
    
    @staticmethod
    def mach_number(velocity, altitude):
        """마하수 계산"""
        sound_speed = PhysicsUtils.sound_speed(altitude)
        return velocity / sound_speed
    
    @staticmethod
    def drag_coefficient_model(mach, alpha_deg=0):
        """Neural ODE 호환 항력계수 모델"""
        cd_base = 0.2 + 0.3 * np.exp(-((mach - 1.2) / 0.8)**2)
        alpha_rad = alpha_deg * DEG_TO_RAD
        cd_alpha = 0.1 * np.sin(2 * alpha_rad)**2
        return cd_base + cd_alpha

    @staticmethod
    def get_aerodynamic_coefficients(missile_data, mach, alpha_rad, beta_rad):
        """
        마하 수, 받음각, 옆미끄럼각에 따른 공력 계수 반환 (6DoF용)
        """
        
        def interp_table(table, mach):
            mach_points = sorted(table.keys())
            return np.interp(mach, mach_points, [table[m] for m in mach_points])
        
        # 받음각과 마하수에 따른 Cd, Cl 룩업
        cd_base = interp_table(missile_data.get("cd_table", {}), mach)
        cl_base = interp_table(missile_data.get("cl_table", {}), mach)
        cm_base = interp_table(missile_data.get("cm_table", {}), mach)
        
        # 받음각에 따른 항력 및 양력 계수 보정
        Cd = cd_base + 0.1 * np.sin(alpha_rad)**2
        Cl = cl_base * np.sin(alpha_rad)
        
        # 모멘트 계수 (피치, 요, 롤)
        # Cm: 피치 모멘트, Cn: 요 모멘트, Cl_roll: 롤 모멘트
        Cm = cm_base * np.sin(alpha_rad)
        Cn = 0.0  # 옆미끄럼각에 따른 요 모멘트 (단순화)
        Cl_roll = 0.0 # 롤 모멘트 (단순화)

        return Cd, Cl, Cm, Cn, Cl_roll

# Neural ODE 상태 벡터 정의 (기존 3DoF 호환성 유지)
class StateVector:
    # ... (기존 코드와 동일) ...
    pass

# 6DoF 시뮬레이션용 미사일 정보
ENHANCED_MISSILE_TYPES = {
    "SCUD-B": {
        "name": "SCUD-B",
        "diameter": 0.88,
        "length": 10.94,
        "nozzle_diameter": 0.6,
        "launch_weight": 5860,
        "payload": 985,
        "propellant_mass": 4875,
        "range_km": 300,
        "propellant_type": "LIQUID",
        "isp_sea": 230,
        "isp_vac": 258,
        "burn_time": 65,
        "vertical_time": 10,
        "pitch_time": 15,
        "pitch_angle_deg": 20,
        "reference_area": np.pi * (0.88/2)**2,
        "cd_base": 0.25,
        "thrust_profile": lambda t: 5860 * 9.81 * 230 / 65 if t < 65 else 0,
        "mass_flow_rate": lambda t: 4875 / 65 if t < 65 else 0,
        "drag_model": lambda mach, alpha: PhysicsUtils.drag_coefficient_model(mach, alpha),

        # 6DoF 모델을 위한 물리적 특성 추가
        "center_of_mass": np.array([5.0, 0, 0]),  # 무게중심 (m), 미사일 축 기준
        "inertia_tensor": np.array([  # 관성 모멘트 텐서 (kg·m²)
            [50000, 0, 0],
            [0, 100000, 0],
            [0, 0, 100000]
        ]),
        "center_of_pressure": 0.6,      # 공력 중심 (길이 대비 비율)
        "cd_table": {
            0.0: 0.32, 0.5: 0.36, 0.8: 0.47, 1.0: 0.82,
            1.2: 0.62, 1.5: 0.42, 2.0: 0.37, 3.0: 0.32,
            4.0: 0.30, 5.0: 0.27
        },
        "cl_table": {
            0.0: 0.0, 0.5: 0.1, 0.8: 0.2, 1.0: 0.3,
            1.2: 0.4, 1.5: 0.35, 2.0: 0.25, 3.0: 0.2,
            4.0: 0.15, 5.0: 0.1
        },
        "cm_table": {
            0.0: 0.0, 1.0: -0.1, 2.0: -0.2, 5.0: -0.1
        },
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
        "isp_sea": 255,
        "isp_vacuum": 280,
        "reference_area": np.pi * (1.36/2)**2,
        "vertical_time": 10,
        "pitch_time": 20,
        "pitch_angle_deg": 15,
        "burn_time": 70,
        "thrust_profile": lambda t: 16500 * 9.81 * 280 / 70 * (1.2 if t < 15 else 1.0) if t < 70 else 0,
        "mass_flow_rate": lambda t: 15300 / 70 if t < 70 else 0,
        "drag_model": lambda mach, alpha: PhysicsUtils.drag_coefficient_model(mach, alpha) * 0.9,
        
        "center_of_mass": np.array([8.0, 0, 0]),
        "inertia_tensor": np.array([
            [150000, 0, 0],
            [0, 300000, 0],
            [0, 0, 300000]
        ]),
        "center_of_pressure": 0.65,
        "cd_table": {
            0.0: 0.28, 0.5: 0.33, 0.8: 0.43, 1.0: 0.78,
            1.2: 0.58, 1.5: 0.38, 2.0: 0.33, 3.0: 0.28,
            4.0: 0.26, 5.0: 0.23
        },
        "cl_table": {
            0.0: 0.0, 0.5: 0.1, 0.8: 0.2, 1.0: 0.3,
            1.2: 0.35, 1.5: 0.3, 2.0: 0.2, 3.0: 0.15,
            4.0: 0.1, 5.0: 0.05
        },
        "cm_table": {
            0.0: 0.0, 1.0: -0.08, 2.0: -0.15, 5.0: -0.07
        },
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

        "center_of_mass": np.array([4.0, 0, 0]),
        "inertia_tensor": np.array([
            [10000, 0, 0],
            [0, 20000, 0],
            [0, 0, 20000]
        ]),
        "center_of_pressure": 0.55,
        "cd_table": {
            0.0: 0.35, 0.5: 0.38, 0.8: 0.48, 1.0: 0.85,
            1.2: 0.65, 1.5: 0.45, 2.0: 0.40, 3.0: 0.35,
            4.0: 0.33, 5.0: 0.30
        },
        "cl_table": {
            0.0: 0.0, 0.5: 0.1, 0.8: 0.2, 1.0: 0.25,
            1.2: 0.3, 1.5: 0.28, 2.0: 0.22, 3.0: 0.18,
            4.0: 0.15, 5.0: 0.1
        },
        "cm_table": {
            0.0: 0.0, 1.0: -0.12, 2.0: -0.2, 5.0: -0.1
        },
    }
}

# 기존 3DoF 호환성을 위한 코드 (새로운 6DoF 모델에는 불필요하지만, 기존 코드와의 호환성을 위해 유지)
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
def get_physics_constants():
    return {
        'g': G,
        'earth_radius': R_EARTH,
        'air_density_sea_level': STD_DENSITY_SEA_LEVEL,
        'sound_speed_sea_level': PhysicsUtils.sound_speed(0)
    }
def create_missile_tensor(missile_type, device='cpu'):
    if missile_type not in ENHANCED_MISSILE_TYPES:
        raise ValueError(f"Unknown missile type: {missile_type}")
    info = ENHANCED_MISSILE_TYPES[missile_type]
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
    if missile_type not in ENHANCED_MISSILE_TYPES:
        raise ValueError(f"Unknown missile type: {missile_type}")
    return ENHANCED_MISSILE_TYPES[missile_type]
def get_cd_table_for_missile(missile_name):
    default_cd_table = {
        'mach': np.array([0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0]),
        'cd': np.array([0.3, 0.35, 0.45, 0.8, 0.6, 0.4, 0.35, 0.3, 0.28, 0.25])
    }
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
def set_missile_type(missile_type_key):
    global MISSILE_MASS, MISSILE_WEIGHT, PROPELLANT_MASS, ISP, WING_AREA
    global VERTICAL_TIME, PITCH_TIME, PITCH_ANGLE_DEG, BURN_TIME
    if missile_type_key not in ENHANCED_MISSILE_TYPES:
        print(f"오류: 미사일 유형 '{missile_type_key}'을(를) 찾을 수 없습니다.")
        return False
    missile_info = ENHANCED_MISSILE_TYPES[missile_type_key]
    global VERTICAL_PHASE_TIME, PITCH_PHASE_TIME, CONSTANT_PHASE_TIME
    MISSILE_MASS = missile_info["launch_weight"]
    MISSILE_WEIGHT = MISSILE_MASS * G
    PROPELLANT_MASS = missile_info["propellant_mass"]
    ISP = missile_info["isp_sea"]
    WING_AREA = missile_info["reference_area"]
    VERTICAL_TIME = missile_info["vertical_time"]
    PITCH_TIME = missile_info["pitch_time"]
    PITCH_ANGLE_DEG = missile_info["pitch_angle_deg"]
    BURN_TIME = missile_info["burn_time"]
    VERTICAL_PHASE_TIME = VERTICAL_TIME
    PITCH_PHASE_TIME = PITCH_TIME
    CONSTANT_PHASE_TIME = max(0, BURN_TIME - VERTICAL_TIME - PITCH_TIME)
    print(f"미사일 유형이 '{missile_info['name']}'(으)로 설정되었습니다.")
    return True
def get_available_missile_types():
    return list(ENHANCED_MISSILE_TYPES.keys())
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
VERTICAL_PHASE_TIME = VERTICAL_TIME
PITCH_PHASE_TIME = PITCH_TIME
CONSTANT_PHASE_TIME = max(0, BURN_TIME - VERTICAL_TIME - PITCH_TIME)
INTERVAL = 0.1
SIM_TIME = 2000
SIMULATION_END_TIME = SIM_TIME
MISSILE_TYPES = ENHANCED_MISSILE_TYPES
STD_DENSITY_SEA_LEVEL = 1.225
STD_PRESSURE_SEA_LEVEL = 101325
STD_GRAVITY = 9.80665
AIR_MOLAR_MASS = 0.0289644
UNIVERSAL_GAS_CONSTANT = 8.314462
AIR_GAS_CONSTANT = 287.058
ATMOSPHERIC_LAYERS = [
    (0, 11000, -0.0065, 0, 288.15),
    (11000, 20000, 0, 11000, 216.65),
    (20000, 32000, 0.001, 20000, 216.65),
    (32000, 47000, 0.0028, 32000, 228.65),
    (47000, 51000, 0, 47000, 270.65),
    (51000, 71000, -0.0028, 51000, 270.65),
    (71000, 84852, -0.002, 71000, 214.65)
]
HIGH_ALTITUDE_REFERENCE = [
    (100000, 0.0000552, 8500),
    (120000, 0.0000024, 9000),
    (150000, 0.0000002, 10000)
]
CL_VERTICAL = 0.0
CL_PITCH = 0.5
CL_CONSTANT = 0.3
CL_TERMINAL = 0.1
K = 0.05
WING_AREA = 0.2
ISP = 250
TSFC = 0.0004
PITCH_PHASE_TIME = PITCH_TIME
CONSTANT_PHASE_TIME = BURN_TIME - VERTICAL_TIME - PITCH_TIME
EXOSPHERE_ALTITUDE = 600000
DEG_TO_RAD = np.pi / 180
RAD_TO_DEG = 180 / np.pi
BASE_CD_TABLE = {0: 0.3, 1: 0.8, 2: 1.2, 3: 1.0, 4: 0.9, 5: 0.8}
def calculate_cd_table(diameter, length, nozzle_diameter, propellant_type):
    mach_numbers = np.linspace(0, 5, 51)
    cd_values = np.array([PhysicsUtils.drag_coefficient_model(m) for m in mach_numbers])
    return dict(zip(mach_numbers, cd_values))
def get_available_missile_types():
    return list(ENHANCED_MISSILE_TYPES.keys())
def set_missile_type(missile_type):
    if missile_type in ENHANCED_MISSILE_TYPES:
        return True
    return False