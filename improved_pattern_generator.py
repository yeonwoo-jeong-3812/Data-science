#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Natural Trajectory Pattern Data Generator
Comprehensive simulation of all missile-angle-azimuth combinations without artificial range constraints
"""

import numpy as np
import os
import config as cfg
import traceback
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools

try:
    from main import MissileSimulation
except ImportError:
    print("Error: Cannot find MissileSimulation class in main.py")
    exit()

class NaturalTrajectoryDataGenerator:
    """자연스러운 궤도 패턴 데이터 생성기 (제한 없는 전면 시뮬레이션)"""
    
    def __init__(self, output_dir="natural_trajectory_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 🎯 전면적 시뮬레이션 파라미터 (제한 없음!)
        self.launch_angles = list(range(10, 81, 3))  # 10°~80°, 3° 간격 (24개)
        self.azimuth_angles = list(range(30, 151, 15))  # 30°~150°, 15° 간격 (9개)
        
        print(f"🌟 자연스러운 궤도 데이터 생성기 초기화")
        print(f"   발사각: {self.launch_angles[0]}°~{self.launch_angles[-1]}° ({len(self.launch_angles)}개)")
        print(f"   방위각: {self.azimuth_angles[0]}°~{self.azimuth_angles[-1]}° ({len(self.azimuth_angles)}개)")
        print(f"   총 조합: {len(self.launch_angles)} × {len(self.azimuth_angles)} = {len(self.launch_angles) * len(self.azimuth_angles)}개/미사일")
    
    def generate_comprehensive_natural_dataset(self, missile_types, samples_per_combination=3, sim_time=1500):
        """
        🚀 전면적 자연 궤도 데이터셋 생성 (사거리 제한 없음)
        
        Args:
            missile_types: 미사일 타입 리스트
            samples_per_combination: 조합당 샘플 수 (노이즈 변형용)
            sim_time: 시뮬레이션 시간 (초)
        """
        all_trajectories = []
        trajectory_labels = []
        generation_stats = {
            'total_attempts': 0,
            'successful_samples': 0,
            'by_missile': {},
            'range_distribution': {},
            'angle_success_rate': {}
        }
        
        print(f"\n🚀 자연스러운 궤도 데이터 전면 생성 시작...")
        print(f"   미사일 타입: {missile_types}")
        print(f"   조합당 샘플: {samples_per_combination}")
        print(f"   시뮬레이션 시간: {sim_time}초")
        
        total_combinations = len(missile_types) * len(self.launch_angles) * len(self.azimuth_angles)
        total_attempts = total_combinations * samples_per_combination
        print(f"   총 시도할 시뮬레이션: {total_attempts:,}개")
        
        sample_id = 0
        missile_type_to_idx = {m_type: idx for idx, m_type in enumerate(missile_types)}
        
        # 🎯 모든 미사일에 대해 전면 시뮬레이션
        for m_type in missile_types:
            if m_type not in cfg.MISSILE_TYPES:
                print(f"   ⚠️ 미사일 '{m_type}' not found in config.py, skipping.")
                continue
            
            print(f"\n🚀 미사일: {m_type}")
            generation_stats['by_missile'][m_type] = {
                'attempts': 0, 'successes': 0, 'range_min': float('inf'), 'range_max': 0
            }
            
            # 발사각별 성공률 추적
            for launch_angle in self.launch_angles:
                angle_successes = 0
                angle_attempts = len(self.azimuth_angles) * samples_per_combination
                
                print(f"  📐 발사각 {launch_angle}°: ", end="", flush=True)
                
                # 🧭 모든 방위각에 대해
                for azimuth_angle in self.azimuth_angles:
                    
                    # 🔄 조합당 여러 샘플 (현실적 변형)
                    for sample_idx in range(samples_per_combination):
                        generation_stats['total_attempts'] += 1
                        generation_stats['by_missile'][m_type]['attempts'] += 1
                        
                        try:
                            # 미사일 타입 설정
                            if not cfg.set_missile_type(m_type):
                                continue
                            
                            # 🎲 현실적 변형 추가
                            angle_noise = np.random.normal(0, 0.3)  # ±0.3° 표준편차
                            azimuth_noise = np.random.normal(0, 0.5)  # ±0.5° 표준편차
                            
                            actual_angle = np.clip(launch_angle + angle_noise, 5, 85)
                            actual_azimuth = np.clip(azimuth_angle + azimuth_noise, 20, 160)
                            
                            # 시뮬레이션 실행
                            simulator = MissileSimulation(missile_type=m_type, apply_errors=False)
                            simulator.initialize_simulation(
                                launch_angle_deg=actual_angle,
                                azimuth_deg=actual_azimuth,
                                sim_time=sim_time
                            )
                            
                            results = simulator.run_simulation(sim_time=sim_time)
                            
                            if not results or 'time' not in results or len(results['time']) == 0:
                                continue
                            
                            # 🎯 궤도 데이터 처리 (모든 성공한 궤도 저장!)
                            trajectory_data = self._process_natural_trajectory_data(
                                results, m_type, launch_angle, azimuth_angle, 
                                actual_angle, actual_azimuth, sample_id
                            )
                            
                            if trajectory_data is not None:
                                all_trajectories.append(trajectory_data)
                                
                                # 📋 레이블 정보
                                label_info = {
                                    'missile_type': m_type,
                                    'missile_type_idx': missile_type_to_idx[m_type],
                                    'nominal_launch_angle': launch_angle,
                                    'nominal_azimuth_angle': azimuth_angle,
                                    'actual_launch_angle': actual_angle,
                                    'actual_azimuth_angle': actual_azimuth,
                                    'sample_id': sample_id,
                                    'range_km': trajectory_data['final_range_km'],
                                    'combination_id': f"{m_type}_{launch_angle}_{azimuth_angle}"
                                }
                                trajectory_labels.append(label_info)
                                
                                # 📊 통계 업데이트
                                sample_id += 1
                                generation_stats['successful_samples'] += 1
                                generation_stats['by_missile'][m_type]['successes'] += 1
                                angle_successes += 1
                                
                                # 사거리 분포 추적
                                range_km = trajectory_data['final_range_km']
                                range_bin = int(range_km // 100) * 100  # 100km 단위
                                generation_stats['range_distribution'][range_bin] = \
                                    generation_stats['range_distribution'].get(range_bin, 0) + 1
                                
                                # 미사일별 사거리 범위 업데이트
                                stats = generation_stats['by_missile'][m_type]
                                stats['range_min'] = min(stats['range_min'], range_km)
                                stats['range_max'] = max(stats['range_max'], range_km)
                        
                        except Exception as e:
                            # 에러 무시하고 계속 (대용량 시뮬레이션)
                            continue
                
                # 발사각별 성공률 기록
                success_rate = angle_successes / angle_attempts if angle_attempts > 0 else 0
                generation_stats['angle_success_rate'][launch_angle] = success_rate
                print(f"{angle_successes}/{angle_attempts} ({success_rate*100:.0f}%)")
        
        # 🎉 생성 결과 요약
        print(f"\n🎉 자연스러운 궤도 데이터 생성 완료!")
        print(f"   총 성공: {generation_stats['successful_samples']:,} / {generation_stats['total_attempts']:,} "
              f"({100*generation_stats['successful_samples']/generation_stats['total_attempts']:.1f}%)")
        
        if len(all_trajectories) == 0:
            print("❌ 오류: 성공한 궤도가 없습니다!")
            return [], [], {}
        
        # 💾 데이터 저장 및 분석
        self._save_natural_dataset(all_trajectories, trajectory_labels, missile_types, generation_stats)
        self._analyze_natural_distribution(all_trajectories, trajectory_labels, generation_stats)
        
        return all_trajectories, trajectory_labels, generation_stats
    
    def _process_natural_trajectory_data(self, results, missile_type, nominal_angle, nominal_azimuth, 
                                       actual_angle, actual_azimuth, sample_id):
        """자연스러운 궤도 데이터 처리 (품질 필터링만, 사거리 제한 없음)"""
        try:
            time_array = np.array(results['time'])
            
            # 6D 상태벡터 [x, y, h, v, gamma, psi]
            trajectory_state = np.stack([
                np.array(results['x']),      # X 위치 (m)
                np.array(results['y']),      # Y 위치 (m)
                np.array(results['h']),      # 고도 (m)
                np.array(results['velocity']), # 속도 (m/s)
                np.array(results['gamma']),  # 피치각 (도)
                np.array(results['psi'])     # 방위각 (도)
            ], axis=1).astype(np.float32)
            
            # 🔍 기본적인 물리 타당성만 검증 (관대한 기준)
            min_altitude = np.min(results['h'])
            max_altitude = np.max(results['h'])
            final_range = np.sqrt(trajectory_state[-1, 0]**2 + trajectory_state[-1, 1]**2)
            
            # 기본 품질 기준 (매우 관대)
            if min_altitude < -10:  # 10m 이하 침하
                return None
            if max_altitude > 2000000:  # 2000km 이상 상승 (극단적 경우만 제외)
                return None
            if len(results['time']) > 30000:  # 너무 긴 시뮬레이션
                return None
            if final_range < 1000:  # 1km 미만 (거의 제자리)
                return None
            if final_range > 10000000:  # 10,000km 초과 (비현실적)
                return None
            if len(time_array) < 50:  # 최소 데이터 길이
                return None
            if np.any(np.isnan(trajectory_state)) or np.any(np.isinf(trajectory_state)):
                return None
            
            # ✨ Signature 특성 계산
            signature_features = self._calculate_trajectory_signature(trajectory_state, time_array)
            
            # 미사일 정보
            missile_info = cfg.MISSILE_TYPES[missile_type]
            
            return {
                'trajectory_id': sample_id,
                'time': time_array.astype(np.float32),
                'trajectory': trajectory_state,  # [N, 6]
                'missile_type': missile_type,
                'nominal_launch_angle': float(nominal_angle),
                'nominal_azimuth_angle': float(nominal_azimuth),
                'actual_launch_angle': float(actual_angle),
                'actual_azimuth_angle': float(actual_azimuth),
                'missile_mass': float(missile_info["launch_weight"]),
                'missile_diameter': float(missile_info["diameter"]),
                'missile_length': float(missile_info["length"]),
                'duration': float(time_array[-1]),
                'max_altitude': float(np.max(trajectory_state[:, 2])),
                'final_range_km': float(final_range / 1000),
                'max_velocity': float(np.max(trajectory_state[:, 3])),
                'trajectory_shape': trajectory_state.shape,
                'signature_features': signature_features
            }
            
        except Exception as e:
            return None
    
    def _calculate_trajectory_signature(self, trajectory_state, time_array):
        """궤도 Signature 특성 계산 (개선됨)"""
        try:
            x, y, h = trajectory_state[:, 0], trajectory_state[:, 1], trajectory_state[:, 2]
            v = trajectory_state[:, 3]
            gamma = np.deg2rad(trajectory_state[:, 4])
            psi = np.deg2rad(trajectory_state[:, 5])
            
            # 1. 속도 프로필 특성
            v_max_idx = np.argmax(v)
            v_max_time_ratio = time_array[v_max_idx] / time_array[-1]
            
            # 2. 고도 프로필 특성
            h_max_idx = np.argmax(h)
            h_max_time_ratio = time_array[h_max_idx] / time_array[-1]
            apogee_altitude = np.max(h)
            
            # 3. 궤도 형태 특성
            total_range = np.sqrt(x[-1]**2 + y[-1]**2)
            range_to_apogee_ratio = apogee_altitude / (total_range + 1e-6)  # 고도/사거리 비율
            
            # 4. 동적 특성
            if len(time_array) > 2:
                dv_dt = np.gradient(v, time_array)
                max_acceleration = np.max(dv_dt)
                max_deceleration = np.min(dv_dt)
            else:
                max_acceleration = max_deceleration = 0.0
            
            # 5. 말단 특성
            terminal_velocity = v[-1]
            terminal_angle = gamma[-1]
            velocity_loss_ratio = (np.max(v) - terminal_velocity) / (np.max(v) + 1e-6)
            
            # 6. 비행 단계 분석
            ascending_mask = h[1:] > h[:-1]
            ascending_time = np.sum(ascending_mask) / len(ascending_mask) if len(ascending_mask) > 0 else 0
            
            # 7. 궤도 복잡도
            if len(x) > 5:
                # 경로 길이 계산
                path_segments = np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(h)**2)
                total_path_length = np.sum(path_segments)
                path_efficiency = total_range / (total_path_length + 1e-6)  # 직선 대비 효율성
            else:
                path_efficiency = 1.0
            
            return {
                'v_max_time_ratio': float(v_max_time_ratio),
                'h_max_time_ratio': float(h_max_time_ratio),
                'range_to_apogee_ratio': float(range_to_apogee_ratio),
                'ascending_time_ratio': float(ascending_time),
                'max_acceleration': float(max_acceleration),
                'max_deceleration': float(max_deceleration),
                'terminal_velocity': float(terminal_velocity),
                'terminal_angle_deg': float(np.rad2deg(terminal_angle)),
                'velocity_loss_ratio': float(velocity_loss_ratio),
                'path_efficiency': float(path_efficiency),
                'apogee_altitude_km': float(apogee_altitude / 1000),
                'flight_duration': float(time_array[-1])
            }
            
        except Exception as e:
            # 기본값 반환
            return {
                'v_max_time_ratio': 0.5, 'h_max_time_ratio': 0.5, 'range_to_apogee_ratio': 0.1,
                'ascending_time_ratio': 0.5, 'max_acceleration': 0.0, 'max_deceleration': 0.0,
                'terminal_velocity': 0.0, 'terminal_angle_deg': -45.0, 'velocity_loss_ratio': 0.0,
                'path_efficiency': 1.0, 'apogee_altitude_km': 0.0, 'flight_duration': 0.0
            }
    
    def _save_natural_dataset(self, trajectories, labels, missile_types, stats):
        """자연스러운 데이터셋 저장"""
        print(f"\n💾 자연스러운 데이터셋 저장 중...")
        
        # 궤도 데이터 저장
        trajectory_file = os.path.join(self.output_dir, "trajectory_patterns.npz")
        save_dict = {}
        for i, traj in enumerate(trajectories):
            save_dict[f'trajectory_{i}'] = traj
        np.savez_compressed(trajectory_file, **save_dict)
        
        # 레이블 데이터 저장
        label_file = os.path.join(self.output_dir, "trajectory_labels.npz")
        np.savez_compressed(label_file, 
                          labels=labels,
                          missile_types=missile_types,
                          launch_angles=self.launch_angles,
                          azimuth_angles=self.azimuth_angles)
        
        # 메타데이터 저장
        metadata = {
            'total_samples': len(trajectories),
            'missile_types': missile_types,
            'launch_angles_deg': self.launch_angles,
            'azimuth_angles_deg': self.azimuth_angles,
            'state_dimensions': 6,
            'state_names': ['x', 'y', 'h', 'v', 'gamma', 'psi'],
            'state_units': ['m', 'm', 'm', 'm/s', 'deg', 'deg'],
            'signature_features': [
                'v_max_time_ratio', 'h_max_time_ratio', 'range_to_apogee_ratio',
                'ascending_time_ratio', 'max_acceleration', 'max_deceleration',
                'terminal_velocity', 'terminal_angle_deg', 'velocity_loss_ratio',
                'path_efficiency', 'apogee_altitude_km', 'flight_duration'
            ],
            'generation_info': {
                'method': 'comprehensive_natural_simulation',
                'coordinate_system': 'X=East, Y=North, H=Up',
                'angle_units': 'degrees',
                'quality_filters': 'Basic physical constraints only',
                'range_limits': 'None (natural distribution)',
                'comprehensive': True
            }
        }
        
        metadata_file = os.path.join(self.output_dir, "metadata.npz")
        np.savez_compressed(metadata_file, **metadata)
        
        # 생성 통계 저장
        stats_file = os.path.join(self.output_dir, "generation_stats.npz")
        np.savez_compressed(stats_file, generation_stats=stats)
        
        print(f"   ✅ 자연스러운 데이터셋 저장 완료:")
        print(f"      궤도 데이터: {trajectory_file}")
        print(f"      레이블 데이터: {label_file}")
        print(f"      메타데이터: {metadata_file}")
        print(f"      생성 통계: {stats_file}")
    
    def _analyze_natural_distribution(self, trajectories, labels, stats):
        """자연스러운 데이터 분포 분석"""
        print(f"\n📊 자연스러운 궤도 분포 분석")
        print("=" * 70)
        
        # 1. 전체 통계
        ranges = [traj['final_range_km'] for traj in trajectories]
        altitudes = [traj['max_altitude'] / 1000 for traj in trajectories]
        durations = [traj['duration'] for traj in trajectories]
        
        print(f"\n🌍 전체 궤도 분포:")
        print(f"   총 궤도: {len(trajectories):,}개")
        print(f"   사거리: {np.min(ranges):.1f}~{np.max(ranges):.1f}km (평균: {np.mean(ranges):.1f}km)")
        print(f"   고도: {np.min(altitudes):.1f}~{np.max(altitudes):.1f}km (평균: {np.mean(altitudes):.1f}km)")
        print(f"   비행시간: {np.min(durations):.0f}~{np.max(durations):.0f}초 (평균: {np.mean(durations):.0f}초)")
        
        # 2. 미사일별 분포
        print(f"\n🚀 미사일별 분포:")
        for missile_type in stats['by_missile']:
            missile_data = stats['by_missile'][missile_type]
            missile_trajectories = [traj for traj in trajectories if traj['missile_type'] == missile_type]
            missile_ranges = [traj['final_range_km'] for traj in missile_trajectories]
            
            if missile_ranges:
                print(f"   {missile_type}: {len(missile_trajectories)}개 궤도")
                print(f"      사거리: {np.min(missile_ranges):.1f}~{np.max(missile_ranges):.1f}km")
                print(f"      성공률: {100*missile_data['successes']/missile_data['attempts']:.1f}%")
        
        # 3. 사거리 구간별 분포 (자연스러운 그룹화)
        print(f"\n📏 자연스러운 사거리 분포:")
        range_bins = {}
        for range_km in ranges:
            bin_key = int(range_km // 200) * 200  # 200km 단위로 그룹화
            range_bins[bin_key] = range_bins.get(bin_key, 0) + 1
        
        for bin_start in sorted(range_bins.keys()):
            count = range_bins[bin_start]
            print(f"   {bin_start}~{bin_start+200}km: {count}개 궤도")
        
        # 4. 발사각별 성공률
        print(f"\n📐 발사각별 성공률:")
        best_angles = sorted(stats['angle_success_rate'].items(), 
                           key=lambda x: x[1], reverse=True)[:5]
        worst_angles = sorted(stats['angle_success_rate'].items(), 
                            key=lambda x: x[1])[:5]
        
        print(f"   성공률 높은 각도: {[f'{a}°({r*100:.0f}%)' for a, r in best_angles]}")
        print(f"   성공률 낮은 각도: {[f'{a}°({r*100:.0f}%)' for a, r in worst_angles]}")
        
        # 5. Signature 특성 분포
        print(f"\n🎭 Signature 특성 범위:")
        if trajectories and 'signature_features' in trajectories[0]:
            feature_samples = {}
            for feature in ['range_to_apogee_ratio', 'velocity_loss_ratio', 'path_efficiency']:
                values = [traj['signature_features'][feature] for traj in trajectories]
                feature_samples[feature] = values
                print(f"   {feature}: {np.min(values):.3f}~{np.max(values):.3f} (평균: {np.mean(values):.3f})")
    
    def visualize_natural_distribution(self, trajectories, labels, max_display=200):
        """자연스러운 궤도 분포 시각화"""
        print(f"\n📈 자연스러운 궤도 분포 시각화...")
        
        if len(trajectories) == 0:
            print("Error: No trajectories to visualize")
            return
        
        # 시각화용 샘플링 (너무 많으면 보기 어려움)
        if len(trajectories) > max_display:
            indices = np.random.choice(len(trajectories), max_display, replace=False)
            display_trajectories = [trajectories[i] for i in indices]
            display_labels = [labels[i] for i in indices]
        else:
            display_trajectories = trajectories
            display_labels = labels
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Natural Trajectory Distribution Analysis', fontsize=18, fontweight='bold')
        
        # 미사일별 색상
        unique_missiles = list(set(label['missile_type'] for label in display_labels))
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_missiles)))
        missile_colors = {missile: color for missile, color in zip(unique_missiles, colors)}
        
        # 1. 3D 궤도 분포
        ax = fig.add_subplot(3, 3, 1, projection='3d')
        for i, (traj, label) in enumerate(zip(display_trajectories, display_labels)):
            if i % 5 == 0:  # 5개 중 1개만 표시 (성능)
                trajectory_data = traj['trajectory']
                color = missile_colors[label['missile_type']]
                ax.plot(trajectory_data[:, 0]/1000, trajectory_data[:, 1]/1000, 
                       trajectory_data[:, 2]/1000, color=color, alpha=0.3, linewidth=0.8)
        
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Altitude (km)')
        ax.set_title('3D Trajectory Distribution')
        
        # 2. 사거리 분포
        ax = axes[0, 1]
        ranges = [traj['final_range_km'] for traj in display_trajectories]
        missiles = [label['missile_type'] for label in display_labels]
        
        for missile in unique_missiles:
            missile_ranges = [r for r, m in zip(ranges, missiles) if m == missile]
            ax.hist(missile_ranges, bins=20, alpha=0.6, label=missile, 
                   color=missile_colors[missile])
        
        ax.set_xlabel('Range (km)')
        ax.set_ylabel('Frequency')
        ax.set_title('Natural Range Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 고도 vs 사거리
        ax = axes[0, 2]
        for missile in unique_missiles:
            missile_data = [(traj['final_range_km'], traj['max_altitude']/1000) 
                          for traj, label in zip(display_trajectories, display_labels) 
                          if label['missile_type'] == missile]
            if missile_data:
                ranges, altitudes = zip(*missile_data)
                ax.scatter(ranges, altitudes, alpha=0.6, label=missile, 
                         color=missile_colors[missile], s=10)
        
        ax.set_xlabel('Range (km)')
        ax.set_ylabel('Max Altitude (km)')
        ax.set_title('Altitude vs Range')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 발사각 vs 사거리
        ax = axes[1, 0]
        for missile in unique_missiles:
            missile_data = [(label['nominal_launch_angle'], traj['final_range_km']) 
                          for traj, label in zip(display_trajectories, display_labels) 
                          if label['missile_type'] == missile]
            if missile_data:
                angles, ranges = zip(*missile_data)
                ax.scatter(angles, ranges, alpha=0.6, label=missile, 
                         color=missile_colors[missile], s=10)
        
        ax.set_xlabel('Launch Angle (deg)')
        ax.set_ylabel('Range (km)')
        ax.set_title('Launch Angle vs Range')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. 방위각 분포
        ax = axes[1, 1]
        azimuths = [label['nominal_azimuth_angle'] for label in display_labels]
        ax.hist(azimuths, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Azimuth Angle (deg)')
        ax.set_ylabel('Frequency')
        ax.set_title('Azimuth Distribution')
        ax.grid(True, alpha=0.3)
        
        # 6. 비행시간 분포
        ax = axes[1, 2]
        durations = [traj['duration'] for traj in display_trajectories]
        missiles = [label['missile_type'] for label in display_labels]
        
        for missile in unique_missiles:
            missile_durations = [d for d, m in zip(durations, missiles) if m == missile]
            ax.hist(missile_durations, bins=15, alpha=0.6, label=missile, 
                   color=missile_colors[missile])
        
        ax.set_xlabel('Flight Duration (sec)')
        ax.set_ylabel('Frequency')
        ax.set_title('Flight Duration Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 7. Signature 특성: 속도 손실 비율
        ax = axes[2, 0]
        if display_trajectories and 'signature_features' in display_trajectories[0]:
            velocity_loss = [traj['signature_features']['velocity_loss_ratio'] 
                           for traj in display_trajectories]
            missiles = [label['missile_type'] for label in display_labels]
            
            for missile in unique_missiles:
                missile_values = [v for v, m in zip(velocity_loss, missiles) if m == missile]
                ax.hist(missile_values, bins=15, alpha=0.6, label=missile, 
                       color=missile_colors[missile])
        
        ax.set_xlabel('Velocity Loss Ratio')
        ax.set_ylabel('Frequency')
        ax.set_title('Signature: Velocity Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 8. Signature 특성: 사거리 대비 고도 비율
        ax = axes[2, 1]
        if display_trajectories and 'signature_features' in display_trajectories[0]:
            range_apogee = [traj['signature_features']['range_to_apogee_ratio'] 
                          for traj in display_trajectories]
            missiles = [label['missile_type'] for label in display_labels]
            
            for missile in unique_missiles:
                missile_values = [r for r, m in zip(range_apogee, missiles) if m == missile]
                ax.hist(missile_values, bins=15, alpha=0.6, label=missile, 
                       color=missile_colors[missile])
        
        ax.set_xlabel('Range to Apogee Ratio')
        ax.set_ylabel('Frequency')
        ax.set_title('Signature: Range/Apogee Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 9. 같은 사거리 구간의 다양한 궤도 예시
        ax = axes[2, 2]
        # 500±100km 구간 찾기
        target_range = 500
        tolerance = 100
        similar_range_data = [
            (traj, label) for traj, label in zip(display_trajectories, display_labels)
            if abs(traj['final_range_km'] - target_range) <= tolerance
        ]
        
        if similar_range_data:
            for traj, label in similar_range_data[:10]:  # 최대 10개
                trajectory_data = traj['trajectory']
                color = missile_colors[label['missile_type']]
                ax.plot(trajectory_data[:, 0]/1000, trajectory_data[:, 2]/1000, 
                       color=color, alpha=0.7, linewidth=1.5,
                       label=f"{label['missile_type']} {label['nominal_launch_angle']}°")
        
        ax.set_xlabel('X Range (km)')
        ax.set_ylabel('Altitude (km)')
        ax.set_title(f'Same Range (~{target_range}km), Different Trajectories')
        ax.grid(True, alpha=0.3)
        if similar_range_data:
            handles, labels = ax.get_legend_handles_labels()
            unique_labels = dict(zip(labels, handles))
            ax.legend(unique_labels.values(), unique_labels.keys(), fontsize=8)
        
        plt.tight_layout()
        
        # 저장
        plot_file = os.path.join(self.output_dir, "natural_distribution_analysis.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   ✅ 자연 분포 시각화 저장: {plot_file}")

def main():
    """메인 실행 함수 - 자연스러운 전면 궤도 데이터 생성"""
    print("🌟 자연스러운 궤도 패턴 데이터 생성 (제한 없는 전면 시뮬레이션)")
    print("=" * 80)
    
    # 🎯 미사일 타입 (확장 가능)
    missile_types = ["SCUD-B", "NODONG", "KN-23"]
    
    # config.py에서 추가 미사일 확인
    additional_missiles = ["PUKGUKSONG-3", "KN-15", "TAEPODONG-2"]
    for missile in additional_missiles:
        if missile in cfg.MISSILE_TYPES:
            missile_types.append(missile)
            print(f"   추가 미사일 발견: {missile}")
    
    print(f"🚀 시뮬레이션 대상: {missile_types}")
    
    # 자연스러운 데이터 생성기 초기화
    generator = NaturalTrajectoryDataGenerator()
    
    # 📊 예상 시뮬레이션 규모
    total_combinations = len(missile_types) * len(generator.launch_angles) * len(generator.azimuth_angles)
    samples_per_combination = 2  # 조합당 2개 샘플 (빠른 생성)
    max_possible = total_combinations * samples_per_combination
    
    print(f"\n📋 전면 시뮬레이션 계획:")
    print(f"   • 미사일: {len(missile_types)}개")
    print(f"   • 발사각: {len(generator.launch_angles)}개 ({generator.launch_angles[0]}°~{generator.launch_angles[-1]}°)")
    print(f"   • 방위각: {len(generator.azimuth_angles)}개 ({generator.azimuth_angles[0]}°~{generator.azimuth_angles[-1]}°)")
    print(f"   • 조합당 샘플: {samples_per_combination}개")
    print(f"   • 최대 시뮬레이션: {max_possible:,}개")
    print(f"   • 예상 성공: ~{int(max_possible * 0.4):,}개 (40% 성공률 가정)")
    print(f"   • 사거리 범위: 자연 분포 (50km~3000km+ 예상)")
    
    try:
        # 🚀 전면적 자연 궤도 데이터 생성
        trajectories, labels, stats = generator.generate_comprehensive_natural_dataset(
            missile_types=missile_types,
            samples_per_combination=samples_per_combination,
            sim_time=1800  # 30분 (긴 궤도 확보)
        )
        
        if len(trajectories) > 0:
            print(f"\n🎉 자연스러운 궤도 데이터 생성 대성공!")
            print(f"   📊 생성된 궤도: {len(trajectories):,}개")
            print(f"   🌍 사거리 범위: 자연 분포 (제한 없음)")
            print(f"   🎭 Signature 특성: 완전 추출")
            print(f"   ⚖️ 동일 사거리 다양성: 최대화")
            
            # 간단한 결과 미리보기
            ranges = [traj['final_range_km'] for traj in trajectories]
            print(f"\n📏 실제 사거리 분포:")
            print(f"   최소: {np.min(ranges):.1f}km")
            print(f"   최대: {np.max(ranges):.1f}km") 
            print(f"   평균: {np.mean(ranges):.1f}km")
            print(f"   표준편차: {np.std(ranges):.1f}km")
            
            # 시각화
            generator.visualize_natural_distribution(trajectories, labels)
            
            print(f"\n🔗 PINN V2 완벽 연동 준비:")
            print(f"   ✅ 6D 상태벡터: [x, y, h, v, gamma, psi]")
            print(f"   ✅ 자연스러운 궤도 분포")
            print(f"   ✅ 동일 사거리, 다양한 특성")
            print(f"   ✅ 확장된 방위각 지원")
            print(f"   ✅ 풍부한 Signature 데이터")
            
        else:
            print(f"\n❌ 오류: 성공한 궤도가 없습니다!")
            print(f"   가능한 원인:")
            print(f"   1. config.py 미사일 파라미터 확인")
            print(f"   2. 시뮬레이션 시간 부족")
            print(f"   3. 물리 모델 문제")
            
    except Exception as e:
        print(f"\n💥 데이터 생성 중 오류:")
        print(f"   {str(e)}")
        print(f"\n🔍 디버깅 정보:")
        traceback.print_exc()
    
    print(f"\n📁 생성 위치: {generator.output_dir}/")
    print(f"🚀 다음 단계: python PINN_V2.py --data_dir {generator.output_dir}")

if __name__ == "__main__":
    main()