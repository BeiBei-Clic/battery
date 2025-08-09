import numpy as np
from scipy import stats

def calculate_f31_f40_isu(battery_data):
    """计算ISU数据的F31-F40特征，严格按照指导文件定义"""
    
    cycle_data = battery_data.get('cycle_data', [])
    
    # 获取第100次循环的充电段数据
    def get_charge_segments_with_current(cycle_idx):
        if cycle_idx >= len(cycle_data):
            return [], [], []
        
        cycle = cycle_data[cycle_idx]
        current = np.array(cycle.get('current_in_A', []))
        voltage = np.array(cycle.get('voltage_in_V', []))
        time_data = np.array(cycle.get('time_in_s', []))
        
        if len(current) > 0 and len(voltage) > 0 and len(time_data) > 0:
            charge_mask = current > 0
            if np.any(charge_mask):
                charge_current = current[charge_mask]
                charge_voltage = voltage[charge_mask]
                charge_time = time_data[charge_mask]
                
                if len(charge_voltage) > 2:
                    # 分为前半段(CCCV-CCCT)和后半段(CVCC-CVCT)
                    half_point = len(charge_voltage) // 2
                    segment1_current = charge_current[:half_point]
                    segment1_voltage = charge_voltage[:half_point]
                    segment1_time = charge_time[:half_point]
                    
                    segment2_current = charge_current[half_point:]
                    segment2_voltage = charge_voltage[half_point:]
                    segment2_time = charge_time[half_point:]
                    
                    return (segment1_current, segment1_voltage, segment1_time), (segment2_current, segment2_voltage, segment2_time)
        return ([], [], []), ([], [], [])
    
    # 计算段能量的辅助函数（基于实际电流数据）
    def calculate_segment_energy(current_data, voltage_data, time_data):
        if len(current_data) < 2 or len(voltage_data) < 2 or len(time_data) < 2:
            return 0
        
        if len(time_data) > 1:
            dt = np.diff(time_data) / 1e9  # 转换为秒
            power = current_data[:-1] * voltage_data[:-1]  # 使用实际电流和电压
            energy = np.sum(power * dt) / 3600  # 转换为Wh
            return energy
        return 0
    
    # 计算段功率的辅助函数（用于F31，单位为W）
    def calculate_segment_power(current_data, voltage_data):
        if len(current_data) == 0 or len(voltage_data) == 0:
            return 0
        power = current_data * voltage_data
        return np.mean(power)  # 平均功率
    
    # 计算熵的辅助函数
    def calculate_entropy(data):
        if len(data) == 0:
            return 0
        
        data_clean = data[np.isfinite(data)]
        if len(data_clean) < 2:
            return 0
            
        n_bins = min(10, max(3, len(data_clean) // 5))
        hist, _ = np.histogram(data_clean, bins=n_bins)
        
        prob = hist / np.sum(hist)
        prob = prob[prob > 0]
        
        if len(prob) == 0:
            return 0
            
        entropy = -np.sum(prob * np.log2(prob))
        return entropy
    
    # 计算偏度的辅助函数
    def calculate_skewness(data):
        if len(data) < 3:
            return 0
        return stats.skew(data)
    
    # 获取第100次循环的充电段数据
    segment1, segment2 = get_charge_segments_with_current(99)  # 第100次循环
    
    segment1_current, segment1_voltage, segment1_time = segment1
    segment2_current, segment2_voltage, segment2_time = segment2
    
    # F31: CCCV-CCCT段的能量 [W] - 实际应为功率
    f31 = calculate_segment_power(segment1_current, segment1_voltage)
    
    # F32: CVCC-CVCT段的能量 [Wh]
    f32 = calculate_segment_energy(segment2_current, segment2_voltage, segment2_time)
    
    # F33: 两段能量比 CCCV-CCCT / CVCC-CVCT
    energy1 = calculate_segment_energy(segment1_current, segment1_voltage, segment1_time)
    f33 = energy1 / f32 if f32 != 0 else 0
    
    # F34: 两段能量差 (CCCV-CCCT) - (CVCC-CVCT)
    f34 = energy1 - f32
    
    # F35: CCCV-CCCT段的熵 eq 8
    f35 = calculate_entropy(segment1_voltage) if len(segment1_voltage) > 0 else 0
    
    # F36: CCCV-CCCT段的熵 eq 8 (与F35相同，按文档定义)
    f36 = f35
    
    # F37: CCCV段的香农熵
    f37 = calculate_entropy(segment1_voltage) if len(segment1_voltage) > 0 else 0
    
    # F38: CVCC段的香农熵
    f38 = calculate_entropy(segment2_voltage) if len(segment2_voltage) > 0 else 0
    
    # F39: CCCV-CCCT段的偏度系数 eq 4
    f39 = calculate_skewness(segment1_voltage) if len(segment1_voltage) > 0 else 0
    
    # F40: CVCC-CVCT段的偏度系数 eq 4
    f40 = calculate_skewness(segment2_voltage) if len(segment2_voltage) > 0 else 0
    
    return [f31, f32, f33, f34, f35, f36, f37, f38, f39, f40]