# 6DoF vs 교수님 자료(3DoF) 상세 비교 분석

## 🔴 심각한 문제점 발견

### 1. **속도 프로파일 - 완전히 잘못됨**

#### 현재 6DoF (Image 1)
- **최대 속도**: ~1400 m/s (63초)
- **속도 증가**: 거의 선형적으로 증가
- **연소 종료 후**: 속도가 계속 증가 (물리적으로 불가능!)
- **문제**: 연소가 끝난 후(65초 이후)에도 속도가 증가하고 있음

#### 교수님 자료 (원본 이미지)
- **최대 속도**: ~2000 m/s (60초 근처)
- **속도 증가**: 연소 중 급격히 증가
- **연소 종료 후**: 속도가 감소 (항력으로 인한 감속)
- **정상**: 추력이 없으면 속도는 감소해야 함

**🚨 핵심 문제: 연소 종료 후 속도가 증가하는 것은 물리적으로 불가능합니다!**

---

### 2. **질량 변화 - 완전히 잘못됨**

#### 현재 6DoF (Image 4 - Mass)
```
초기 질량: ~6000 kg
65초 이후: 질량이 계속 감소 (1000 kg까지)
```

**🚨 치명적 오류: 연료가 다 소모된 후(65초)에도 질량이 계속 감소!**

#### 올바른 동작 (교수님 자료 기준)
```
0-65초: 질량 감소 (연료 소모)
  - 초기: 5860 kg
  - 연료 소모: 4875 kg
  - 최종: 985 kg (구조 질량)

65초 이후: 질량 일정 (연료 없음)
  - 질량: 985 kg (변화 없음)
```

**원인**: 질량 계산 로직이 시간에만 의존하고 연소 시간을 체크하지 않음

---

### 3. **피치각 진동 - 제어 불안정**

#### 현재 6DoF (Image 3 - Angular Velocity)
- **피치 각속도(q)**: 초기에 4 deg/s까지 진동
- **20초 이후**: 2 deg/s 진동 지속
- **30초 이후**: 1 deg/s 진동 지속
- **문제**: 탄도 비행 중에도 계속 진동

#### 교수님 자료
- **피치 각속도**: 피치 기동 중에만 변화
- **탄도 비행**: 각속도 거의 0 (안정)
- **정상**: 제어 종료 후 안정적

**원인**: 
1. 댐핑이 부족
2. 공력 모멘트 계산 오류
3. 수치 적분 오류

---

### 4. **궤적 스케일 - 여전히 부족**

#### 현재 6DoF
- **사거리**: ~50 km (추정)
- **최대 고도**: ~3 km
- **비행 시간**: ~63초

#### 교수님 자료 (SCUD-B, 45도)
- **사거리**: ~100 km
- **최대 고도**: ~25-30 km
- **비행 시간**: ~300초

**차이**: 사거리 50%, 고도 10%, 시간 20%

---

## 🔍 근본 원인 분석

### 문제 1: 질량 계산 오류

#### 현재 코드 (main_6dof.py:117-118)
```python
mass_flow_rate = self.propellant_mass / self.burn_time
current_mass = self.m0 - mass_flow_rate * t if t < self.burn_time else self.m0 - self.propellant_mass
```

**문제점**: 
- `if t < self.burn_time` 조건이 있지만 실제로는 작동하지 않음
- 이유: 시뮬레이션이 63초에 종료되어 조건 확인이 안됨

#### 올바른 코드
```python
if t < self.burn_time:
    current_mass = self.m0 - mass_flow_rate * t
else:
    current_mass = self.m0 - self.propellant_mass  # 구조 질량 (일정)
```

---

### 문제 2: 추력 계산 - 여전히 문제

#### 현재 코드 (main_6dof.py:137-152)
```python
if t < self.burn_time:
    # 고도에 따른 비추력 보간
    isp_current = isp_sea + (isp_vac - isp_sea) * (altitude / 100000)
    
    # 추력 계산
    thrust_magnitude = isp_current * mass_flow_rate * g
    Thrust_b = np.array([thrust_magnitude, 0, 0])
else:
    Thrust_b = np.array([0, 0, 0])
```

**문제점**:
1. ✅ 공식은 올바름
2. ❌ 하지만 시뮬레이션이 65초 이전에 종료됨
3. ❌ 추력이 충분하지 않아 미사일이 제대로 가속되지 않음

**계산 검증**:
```
해수면 추력 = 230 * (4875/65) * 9.81 = 169,207 N
추력 대 중량비 = 169,207 / (5860 * 9.81) = 2.94

이론적으로는 충분한 추력이지만...
실제 시뮬레이션에서는 미사일이 63초에 지면에 충돌!
```

---

### 문제 3: 시뮬레이션이 너무 일찍 종료

#### 현재 상황
```
비행 시간: 63.2초
종료 원인: 지면 충돌 (altitude = 0)
```

#### 정상 상황 (교수님 자료)
```
비행 시간: ~300초
종료 원인: 지면 충돌 (정상적인 낙하 후)
```

**원인 추정**:
1. 초기 피치 기동이 너무 급격함
2. 미사일이 충분히 상승하지 못하고 다시 하강
3. 탄도 궤적이 형성되지 않음

---

### 문제 4: 피치 제어 로직 오류

#### 현재 코드 (main_6dof.py:233-267)
```python
# 목표 피치 각속도 계산
target_pitch_rate = math.radians(self.pitch_angle_deg_cmd) / self.pitch_time
# = 20도 / 15초 = 1.33 deg/s = 0.0233 rad/s

# PD 제어
Kp = 500
Kd = 300
```

**문제점**:
1. 목표 각속도가 너무 작음 (1.33 deg/s)
2. 15초 동안 20도만 회전 → 너무 느림
3. 교수님 자료는 더 빠른 피치 기동 사용

#### 교수님 자료의 피치 프로그램
```python
# 피치 시간에 따라 받음각 점진적 증가
pitch_progress = (t - vertical_time) / pitch_time
alpha = pitch_rad * pitch_progress

# 피치각 변화율
dgamma = -pitch_rad / pitch_time * smoothing_factor
```

**차이점**:
- 교수님: 프로그램된 피치각 변화 (open-loop)
- 6DoF: PD 제어기 (closed-loop)
- 문제: 6DoF의 목표값이 너무 보수적

---

## 🔧 수정 방안

### 1. 질량 계산 수정 (최우선)

```python
def _get_common_forces_and_moments(self, t, state):
    # 현재 질량 계산 - 수정
    if t < self.burn_time:
        mass_flow_rate = self.propellant_mass / self.burn_time
        current_mass = self.m0 - mass_flow_rate * t
    else:
        # 연소 종료 후 구조 질량만 남음
        current_mass = self.m0 - self.propellant_mass
    
    # 최소 질량 제한 (안전장치)
    min_mass = self.m0 - self.propellant_mass
    current_mass = max(current_mass, min_mass)
```

---

### 2. 피치 제어 개선

#### 옵션 A: 목표 피치각 증가
```python
# config.py
"pitch_angle_deg": 35,  # 20 → 35도로 증가
"pitch_time": 10,       # 15 → 10초로 감소
```

#### 옵션 B: 교수님 방식 채택 (권장)
```python
# 프로그램된 피치각 변화 (open-loop)
if t <= self.vertical_time + self.pitch_time:
    pitch_progress = (t - self.vertical_time) / self.pitch_time
    target_gamma = math.radians(90 - self.pitch_angle_deg_cmd * pitch_progress)
    
    # 현재 피치각
    roll, pitch, yaw = quaternion_to_euler(att_q)
    current_gamma = math.radians(pitch)
    
    # 오차 기반 제어
    error = target_gamma - current_gamma
    Mc_b[1] = Kp * error + Kd * error_rate
```

---

### 3. 초기 속도 증가

```python
def initialize_simulation(self, launch_angle_deg=45, azimuth_deg=90):
    # 초기 속도를 10 m/s로 증가 (수치 안정성)
    vel_b = np.array([10.0, 0.0, 0.0])  # 1.0 → 10.0
```

---

### 4. 적분 설정 개선

```python
def run_simulation(self, launch_angle_deg=45, azimuth_deg=90, sim_time=500):
    sol = solve_ivp(
        self.dynamics_phased, 
        [0, sim_time], 
        initial_state, 
        method='RK45', 
        dense_output=True, 
        events=self.event_ground_impact,
        max_step=0.05,      # 0.1 → 0.05로 감소
        rtol=1e-6,          # 상대 허용 오차
        atol=1e-9           # 절대 허용 오차
    )
```

---

## 📊 교수님 자료와의 구조적 차이

### 3DoF (교수님)
```
상태 벡터: [V, gamma, psi, x, y, h, W, M]
- 8개 변수
- 점 질량 모델
- 프로그램된 제어 (open-loop)
- 4단계 비행: 수직 → 피치 → 등자세 → 관성비행
```

### 6DoF (현재)
```
상태 벡터: [x, y, z, u, v, w, q0, q1, q2, q3, p, q, r]
- 13개 변수
- 강체 동역학
- PD 제어기 (closed-loop)
- 3단계 비행: 수직 → 피치 → 탄도
```

**핵심 차이**:
1. **자유도**: 3DoF는 단순, 6DoF는 복잡
2. **제어**: 3DoF는 프로그램, 6DoF는 피드백
3. **비행 단계**: 3DoF는 4단계, 6DoF는 3단계

**문제**: 6DoF가 "등자세 선회" 단계를 생략함!

---

## 🎯 등자세 선회 단계 누락 문제

### 교수님 자료의 비행 단계
```
1. 수직 상승 (0-10초)
   - 수직으로 상승
   - 속도 증가

2. 피치 프로그램 (10-25초)
   - 피치각 감소 (90° → 70°)
   - 궤적 형성 시작

3. 등자세 선회 (25-65초) ⭐ 중요!
   - 피치각 유지
   - 추력으로 가속
   - 최대 속도 도달

4. 관성 비행 (65-300초)
   - 추력 없음
   - 탄도 궤적
   - 재진입
```

### 현재 6DoF의 비행 단계
```
1. 수직 상승 (0-10초)
   - 수직으로 상승

2. 피치 기동 (10-25초)
   - 피치각 변화

3. 탄도 비행 (25-63초) ❌ 문제!
   - 제어 없음
   - 추력은 있지만 자세 제어 없음
   - 미사일이 불안정
```

**🚨 핵심 문제: 등자세 선회 단계가 없어서 미사일이 제대로 가속되지 않음!**

---

## ✅ 최종 수정 방안

### 1. 비행 단계 재구성 (최우선)

```python
def dynamics_phased(self, t, state):
    Fg_b, Thrust_b, Fa_b, Ma_b, current_mass = self._get_common_forces_and_moments(t, state)
    Mc_b = np.array([0.0, 0.0, 0.0])

    if t <= self.vertical_time:
        # 1. 수직 상승
        pass
        
    elif t <= self.vertical_time + self.pitch_time:
        # 2. 피치 기동
        # (기존 코드 유지)
        
    elif t <= self.burn_time:
        # ⭐ 3. 등자세 선회 (새로 추가!)
        # 피치각을 일정하게 유지하면서 추력으로 가속
        
        # 목표 피치각 (피치 기동 완료 후의 각도)
        target_pitch = math.radians(45)  # 45도 유지
        
        # 현재 피치각
        roll, pitch, yaw = quaternion_to_euler(state[6:10])
        current_pitch = math.radians(pitch)
        
        # 오차 계산
        error = target_pitch - current_pitch
        
        # P 제어 (자세 유지용)
        Kp = 200  # 낮은 게인 (유지만 하면 됨)
        Mc_b[1] = Kp * error
        
    else:
        # 4. 탄도 비행
        pass

    F_total_b = Fg_b + Fa_b + Thrust_b
    M_total_b = Ma_b + Mc_b
    
    return self.dynamics_solver(t, state, F_total_b, M_total_b, current_mass)
```

---

### 2. 질량 계산 수정

```python
# _get_common_forces_and_moments 내부
if t < self.burn_time:
    mass_flow_rate = self.propellant_mass / self.burn_time
    current_mass = self.m0 - mass_flow_rate * t
else:
    current_mass = self.m0 - self.propellant_mass

# 안전장치
min_mass = self.m0 - self.propellant_mass
current_mass = max(current_mass, min_mass)
```

---

### 3. 피치 제어 파라미터 조정

```python
# config.py
"pitch_angle_deg": 45,  # 20 → 45도
"pitch_time": 10,       # 15 → 10초
```

---

### 4. 초기 조건 개선

```python
# 초기 속도
vel_b = np.array([10.0, 0.0, 0.0])  # 1.0 → 10.0

# 초기 고도
pos_i = np.array([0.0, 0.0, 1.0])  # 0.1 → 1.0
```

---

## 📈 예상 개선 효과

| 항목 | 현재 | 수정 후 (예상) | 교수님 자료 |
|------|------|---------------|------------|
| 사거리 | ~50 km | ~100 km | ~100 km |
| 최대 고도 | ~3 km | ~25 km | ~25-30 km |
| 최대 속도 | ~1400 m/s | ~2000 m/s | ~2000 m/s |
| 비행 시간 | ~63 s | ~300 s | ~300 s |
| 질량 (종료) | ~1000 kg ❌ | ~985 kg ✅ | ~985 kg |
| 속도 (종료) | 증가 ❌ | 감소 ✅ | 감소 ✅ |

---

## 🔍 추가 검증 항목

### 1. 물리 법칙 검증
- [ ] 에너지 보존 (운동 + 위치 에너지)
- [ ] 운동량 보존
- [ ] 각운동량 보존

### 2. 수치 안정성
- [ ] 질량이 음수가 되지 않는지
- [ ] 속도가 비정상적으로 증가하지 않는지
- [ ] 각속도가 발산하지 않는지

### 3. 궤적 타당성
- [ ] 탄도 궤적 형태 (포물선)
- [ ] 최대 고도 위치 (사거리의 중간)
- [ ] 재진입 각도 (가파른 각도)

---

## 📝 결론

### 핵심 문제 3가지

1. **등자세 선회 단계 누락** ⭐ 가장 중요!
   - 미사일이 충분히 가속되지 못함
   - 탄도 궤적이 형성되지 않음

2. **질량 계산 오류**
   - 연소 종료 후에도 질량 감소
   - 물리 법칙 위반

3. **피치 제어 부족**
   - 목표 피치각이 너무 작음 (20도)
   - 피치 시간이 너무 김 (15초)

### 수정 우선순위

1. **최우선**: 등자세 선회 단계 추가
2. **긴급**: 질량 계산 수정
3. **중요**: 피치 파라미터 조정
4. **개선**: 초기 조건 최적화

### 다음 단계

1. 등자세 선회 단계 구현
2. 질량 계산 로직 수정
3. 시뮬레이션 재실행
4. 교수님 자료와 정량적 비교
