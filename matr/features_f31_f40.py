import numpy as np
import math
from scipy import stats

def calculate_f31_f40_matr(battery_data):
    """计算MATR数据的F31-F40特征，严格按照指导文件定义"""
    
    # 提取循环数据
    cycle_data = battery_data.get('cycle_data', [])
    
    # 获取CCCV和CVCC段数据的辅助函数
    def get_charge_segments(cycle_idx):
        if cycle_idx >= len(cycle_data):
            return None, None
        
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
                
                if len(charge_time) > 4:
                    # 简化分段：前半段为CCCV-CCCT，后半段为CVCC-CVCT
                    mid_point = len(charge_time) // 2
                    
                    cccv_ccct = {
                        'time': charge_time[:mid_point],
                        'voltage': charge_voltage[:mid_point],
                        'current': charge_current[:mid_point]
                    }
                    
                    cvcc_cvct = {
                        'time': charge_time[mid_point:],
                        'voltage': charge_voltage[mid_point:],
                        'current': charge_current[mid_point:]
                    }
                    
                    return cccv_ccct, cvcc_cvct
        
        return None, None
    
    # 获取第100次和第10次循环的充电段数据
    cccv_100, cvcc_100 = get_charge_segments(99) if len(cycle_data) > 99 else (None, None)
    cccv_10, cvcc_10 = get_charge_segments(9) if len(cycle_data) > 9 else (None, None)
    
    # F31: CCCV-CCCT段能量差值 (Energy during CCCV-CCCT segment of the curve [W])
    def calculate_segment_energy(segment):
        if segment is None or len(segment['time']) < 2:
            return 0
        
        voltage = segment['voltage']
        current = segment['current']
        time_data = segment['time']
        
        # 能量 = 功率 × 时间
        dt = np.diff(time_data)
        power = voltage[:-1] * current[:-1]  # 功率 = 电压 × 电流
        energy = np.sum(power * dt) / 3600  # 转换为Wh
        return energy
    
    energy_cccv_100 = calculate_segment_energy(cccv_100)
    energy_cccv_10 = calculate_segment_energy(cccv_10)
    f31 = energy_cccv_100 - energy_cccv_10
    
    # F32: CVCC-CVCT段能量差值 (Energy during CVCC-CVCT segment of the curve [Wh])
    energy_cvcc_100 = calculate_segment_energy(cvcc_100)
    energy_cvcc_10 = calculate_segment_energy(cvcc_10)
    f32 = energy_cvcc_100 - energy_cvcc_10
    
    # F33: 能量比差值 (Energy ratio CCCV-CCCT / CVCC-CVCT segment of the curve)
    ratio_100 = energy_cccv_100 / energy_cvcc_100 if energy_cvcc_100 != 0 else 0
    ratio_10 = energy_cccv_10 / energy_cvcc_10 if energy_cvcc_10 != 0 else 0
    f33 = ratio_100 - ratio_10
    
    # F34: 能量差的差值 (Energy Difference between the curve segments (CCCV-CCCT) - (CVCC-CVCT))
    diff_100 = energy_cccv_100 - energy_cvcc_100
    diff_10 = energy_cccv_10 - energy_cvcc_10
    f34 = diff_100 - diff_10
    
    # F35: CCCV-CCCT段熵差值 (Entropy of CCCV-CCCT segment of the curve eq 8)
    def calculate_entropy(segment):
        if segment is None or len(segment['voltage']) < 2:
            return 0
        
        voltage = segment['voltage']
        # 计算香农熵
        hist, _ = np.histogram(voltage, bins=10, density=True)
        hist = hist[hist > 0]  # 移除零值
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        return entropy
    
    entropy_cccv_100 = calculate_entropy(cccv_100)
    entropy_cccv_10 = calculate_entropy(cccv_10)
    f35 = entropy_cccv_100 - entropy_cccv_10
    
    # F36: CVCC-CVCT段熵差值 (Entropy of CVCC-CVCT segment of the curve eq 8)
    entropy_cvcc_100 = calculate_entropy(cvcc_100)
    entropy_cvcc_10 = calculate_entropy(cvcc_10)
    f36 = entropy_cvcc_100 - entropy_cvcc_10
    
    # F37: CCCV段香农熵差值 (Shannon entropy of CCCV segment of the curve)
    f37 = f35  # 与F35相同，都是CCCV段的熵
    
    # F38: CVCC段香农熵差值 (Shannon entropy of CVCC segment of the curve)
    f38 = f36  # 与F36相同，都是CVCC段的熵
    
    # F39: CCCV-CCCT段偏度差值 (Skewness coefficient of CCCV-CCCT segment of the curve eq 4)
    def calculate_skewness(segment):
        if segment is None or len(segment['voltage']) < 3:
            return 0
        
        voltage = segment['voltage']
        return stats.skew(voltage)
    
    skew_cccv_100 = calculate_skewness(cccv_100)
    skew_cccv_10 = calculate_skewness(cccv_10)
    f39 = skew_cccv_100 - skew_cccv_10
    
    # F40: CVCC-CVCT段偏度差值 (Skewness coefficient of CVCC-CVCT segment of the curve eq 4)
    skew_cvcc_100 = calculate_skewness(cvcc_100)
    skew_cvcc_10 = calculate_skewness(cvcc_10)
    f40 = skew_cvcc_100 - skew_cvcc_10
    
    return [f31, f32, f33, f34, f35, f36, f37, f38, f39, f40]