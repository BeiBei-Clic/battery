import numpy as np
from scipy import stats

def calculate_f31_f40_isu(battery_data):
    """计算ISU数据的F31-F40特征，严格按照指导文件定义"""
    
    cycle_data = battery_data.get('cycle_data', [])
    
    # 获取充电段数据的辅助函数
    def get_charge_segments(cycle_idx):
        if cycle_idx >= len(cycle_data):
            return [], []
        
        cycle = cycle_data[cycle_idx]
        current = np.array(cycle.get('current_in_A', []))
        voltage = np.array(cycle.get('voltage_in_V', []))
        time_data = np.array(cycle.get('time_in_s', []))
        
        if len(current) > 0 and len(voltage) > 0 and len(time_data) > 0:
            charge_mask = current > 0
            if np.any(charge_mask):
                charge_voltage = voltage[charge_mask]
                charge_time = time_data[charge_mask]
                
                if len(charge_voltage) > 2:
                    # 分为前半段(CCCV-CCCT)和后半段(CVCC-CVCT)
                    half_point = len(charge_voltage) // 2
                    segment1 = np.column_stack((charge_time[:half_point], charge_voltage[:half_point]))
                    segment2 = np.column_stack((charge_time[half_point:], charge_voltage[half_point:]))
                    return segment1, segment2
        return np.array([]), np.array([])
    
    # 计算段能量的辅助函数
    def calculate_segment_energy(segment, mean_current=0.5):
        if len(segment) < 2:
            return 0
        
        time_vals = segment[:, 0]
        voltage_vals = segment[:, 1]
        
        if len(time_vals) > 1:
            dt = np.diff(time_vals) / 1e9  # 转换为秒
            power = voltage_vals[:-1] * mean_current  # 假设平均电流
            energy = np.sum(power * dt) / 3600  # 转换为Wh
            return energy
        return 0
    
    # 修复熵计算的辅助函数
    def calculate_entropy(data):
        if len(data) == 0:
            return 0
        
        # 将数据分箱来计算概率分布
        data_clean = data[np.isfinite(data)]
        if len(data_clean) < 2:
            return 0
            
        # 使用固定的箱数或根据数据长度调整
        n_bins = min(10, max(3, len(data_clean) // 5))
        hist, _ = np.histogram(data_clean, bins=n_bins)
        
        # 计算概率分布
        prob = hist / np.sum(hist)
        prob = prob[prob > 0]  # 移除0概率
        
        if len(prob) == 0:
            return 0
            
        # 计算香农熵
        entropy = -np.sum(prob * np.log2(prob))
        return entropy
    
    # 计算偏度的辅助函数
    def calculate_skewness(data):
        if len(data) < 3:
            return 0
        return stats.skew(data)
    
    # 获取第10次和第100次循环的充电段数据
    segment1_10, segment2_10 = get_charge_segments(9)   # 第10次循环
    segment1_100, segment2_100 = get_charge_segments(99) # 第100次循环
    
    # F31: CCCV-CCCT段能量差值
    energy1_10 = calculate_segment_energy(segment1_10)
    energy1_100 = calculate_segment_energy(segment1_100)
    f31 = energy1_100 - energy1_10
    
    # F32: CVCC-CVCT段能量差值
    energy2_10 = calculate_segment_energy(segment2_10)
    energy2_100 = calculate_segment_energy(segment2_100)
    f32 = energy2_100 - energy2_10
    
    # F33: CCCV-CCCT段能量比差值
    ratio_10 = energy1_10 / energy2_10 if energy2_10 != 0 else 0
    ratio_100 = energy1_100 / energy2_100 if energy2_100 != 0 else 0
    f33 = ratio_100 - ratio_10
    
    # F34: CVCC-CVCT段能量差的差值
    energy_diff_10 = energy1_10 - energy2_10
    energy_diff_100 = energy1_100 - energy2_100
    f34 = energy_diff_100 - energy_diff_10
    
    # F35: CCCV-CCCT段熵差值 (修复熵计算)
    entropy1_10 = calculate_entropy(segment1_10[:, 1]) if len(segment1_10) > 0 else 0
    entropy1_100 = calculate_entropy(segment1_100[:, 1]) if len(segment1_100) > 0 else 0
    f35 = entropy1_100 - entropy1_10
    
    # F36: CVCC-CVCT段熵差值 (修复熵计算)
    entropy2_10 = calculate_entropy(segment2_10[:, 1]) if len(segment2_10) > 0 else 0
    entropy2_100 = calculate_entropy(segment2_100[:, 1]) if len(segment2_100) > 0 else 0
    f36 = entropy2_100 - entropy2_10
    
    # F37: CCCV-CCCT段香农熵差值 (与F35相同)
    f37 = f35
    
    # F38: CVCC-CVCT段香农熵差值 (与F36相同)
    f38 = f36
    
    # F39: CCCV-CCCT段偏度差值
    skew1_10 = calculate_skewness(segment1_10[:, 1]) if len(segment1_10) > 0 else 0
    skew1_100 = calculate_skewness(segment1_100[:, 1]) if len(segment1_100) > 0 else 0
    f39 = skew1_100 - skew1_10
    
    # F40: CVCC-CVCT段偏度差值
    skew2_10 = calculate_skewness(segment2_10[:, 1]) if len(segment2_10) > 0 else 0
    skew2_100 = calculate_skewness(segment2_100[:, 1]) if len(segment2_100) > 0 else 0
    f40 = skew2_100 - skew2_10
    
    return [f31, f32, f33, f34, f35, f36, f37, f38, f39, f40]