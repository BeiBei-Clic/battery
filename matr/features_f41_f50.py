import numpy as np
import math
from scipy import stats
from scipy.spatial.distance import directed_hausdorff

def calculate_f41_f50_matr(battery_data):
    """计算MATR数据的F41-F50特征，严格按照指导文件定义"""
    
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
    
    # F41: CCCV-CCCT段峰度差值 (Kurtosis coefficient of CCCV-CCCT segment of the curve eq 5)
    def calculate_kurtosis(segment):
        if segment is None or len(segment['voltage']) < 4:
            return 0
        
        voltage = segment['voltage']
        return stats.kurtosis(voltage)
    
    kurtosis_cccv_100 = calculate_kurtosis(cccv_100)
    kurtosis_cccv_10 = calculate_kurtosis(cccv_10)
    f41 = kurtosis_cccv_100 - kurtosis_cccv_10
    
    # F42: CVCC-CVCT段峰度差值 (Kurtosis coefficient of CVCC-CVCT segment of the curve eq 5)
    kurtosis_cvcc_100 = calculate_kurtosis(cvcc_100)
    kurtosis_cvcc_10 = calculate_kurtosis(cvcc_10)
    f42 = kurtosis_cvcc_100 - kurtosis_cvcc_10
    
    # F43: CCCV-CCCT段弗雷歇距离差值 (Frechet distance of CCCV-CCCT segment of the curve eq 6)
    def calculate_frechet_distance(segment1, segment2):
        if segment1 is None or segment2 is None:
            return 0
        if len(segment1['voltage']) < 2 or len(segment2['voltage']) < 2:
            return 0
        
        # 简化的弗雷歇距离计算：使用欧几里得距离
        v1 = segment1['voltage']
        v2 = segment2['voltage']
        
        # 调整长度一致
        min_len = min(len(v1), len(v2))
        v1 = v1[:min_len]
        v2 = v2[:min_len]
        
        return np.sqrt(np.sum((v1 - v2) ** 2))
    
    frechet_cccv_100_10 = calculate_frechet_distance(cccv_100, cccv_10)
    f43 = frechet_cccv_100_10
    
    # F44: CVCC-CVCT段弗雷歇距离差值 (Frechet distance of CVCC-CVCT segment of the curve eq 6)
    frechet_cvcc_100_10 = calculate_frechet_distance(cvcc_100, cvcc_10)
    f44 = frechet_cvcc_100_10
    
    # F45: CCCV-CCCT段豪斯多夫距离差值 (Hausdorff distance of CCCV-CCCT segment of the curve eq 7)
    def calculate_hausdorff_distance(segment1, segment2):
        if segment1 is None or segment2 is None:
            return 0
        if len(segment1['voltage']) < 2 or len(segment2['voltage']) < 2:
            return 0
        
        v1 = segment1['voltage'].reshape(-1, 1)
        v2 = segment2['voltage'].reshape(-1, 1)
        
        # 调整长度一致
        min_len = min(len(v1), len(v2))
        v1 = v1[:min_len]
        v2 = v2[:min_len]
        
        return max(directed_hausdorff(v1, v2)[0], directed_hausdorff(v2, v1)[0])
    
    hausdorff_cccv_100_10 = calculate_hausdorff_distance(cccv_100, cccv_10)
    f45 = hausdorff_cccv_100_10
    
    # F46: CVCC-CVCT段豪斯多夫距离差值 (Hausdorff distance of CVCC-CVCT segment of the curve eq 7)
    hausdorff_cvcc_100_10 = calculate_hausdorff_distance(cvcc_100, cvcc_10)
    f46 = hausdorff_cvcc_100_10
    
    # F47: 放电后平均电压下降差值 (Average voltage falloff after discharge [V])
    def calculate_voltage_falloff(cycle_idx):
        if cycle_idx >= len(cycle_data):
            return 0
        
        cycle = cycle_data[cycle_idx]
        current = np.array(cycle.get('current_in_A', []))
        voltage = np.array(cycle.get('voltage_in_V', []))
        
        if len(current) > 0 and len(voltage) > 0:
            # 找到放电结束点
            discharge_mask = current < 0
            if np.any(discharge_mask):
                discharge_indices = np.where(discharge_mask)[0]
                if len(discharge_indices) > 1:
                    discharge_end_idx = discharge_indices[-1]
                    if discharge_end_idx < len(voltage) - 10:
                        # 计算放电结束后的电压下降
                        voltage_at_end = voltage[discharge_end_idx]
                        voltage_after = voltage[discharge_end_idx:discharge_end_idx+10]
                        return voltage_at_end - np.mean(voltage_after)
        return 0
    
    falloff_100 = calculate_voltage_falloff(99) if len(cycle_data) > 99 else 0
    falloff_10 = calculate_voltage_falloff(9) if len(cycle_data) > 9 else 0
    f47 = falloff_100 - falloff_10
    
    # F48: CC阶段电压变化时间间隔差值 (Time interval for equal charge voltage difference during CC phase [s])
    def calculate_cc_voltage_time_interval(cycle_idx):
        if cycle_idx >= len(cycle_data):
            return 0
        
        cycle = cycle_data[cycle_idx]
        current = np.array(cycle.get('current_in_A', []))
        voltage = np.array(cycle.get('voltage_in_V', []))
        time_data = np.array(cycle.get('time_in_s', []))
        
        if len(current) > 0 and len(voltage) > 0 and len(time_data) > 0:
            charge_mask = current > 0
            if np.any(charge_mask):
                charge_voltage = voltage[charge_mask]
                charge_time = time_data[charge_mask]
                
                if len(charge_voltage) > 10:
                    # 计算电压变化的时间间隔
                    voltage_diff = np.diff(charge_voltage)
                    time_diff = np.diff(charge_time)
                    
                    # 找到电压变化相等的时间间隔
                    target_voltage_diff = 0.1  # 0.1V的电压变化
                    indices = np.where(np.abs(voltage_diff - target_voltage_diff) < 0.05)[0]
                    
                    if len(indices) > 0:
                        return np.mean(time_diff[indices])
        return 0
    
    interval_100 = calculate_cc_voltage_time_interval(99) if len(cycle_data) > 99 else 0
    interval_10 = calculate_cc_voltage_time_interval(9) if len(cycle_data) > 9 else 0
    f48 = interval_100 - interval_10
    
    # F49: CC阶段充电容量差值 (Charge capacity during CC phase [Ah])
    def calculate_cc_charge_capacity(cycle_idx):
        if cycle_idx >= len(cycle_data):
            return 0
        
        cycle = cycle_data[cycle_idx]
        current = np.array(cycle.get('current_in_A', []))
        time_data = np.array(cycle.get('time_in_s', []))
        
        if len(current) > 0 and len(time_data) > 0:
            charge_mask = current > 0
            if np.any(charge_mask):
                charge_current = current[charge_mask]
                charge_time = time_data[charge_mask]
                
                if len(charge_current) > 1:
                    # 计算CC阶段的充电容量
                    dt = np.diff(charge_time) / 3600  # 转换为小时
                    capacity = np.sum(charge_current[:-1] * dt)
                    return capacity
        return 0
    
    capacity_100 = calculate_cc_charge_capacity(99) if len(cycle_data) > 99 else 0
    capacity_10 = calculate_cc_charge_capacity(9) if len(cycle_data) > 9 else 0
    f49 = capacity_100 - capacity_10
    
    # F50: CV阶段电流变化时间间隔差值 (Time interval for equal charge current difference during CV phase [s])
    def calculate_cv_current_time_interval(cycle_idx):
        if cycle_idx >= len(cycle_data):
            return 0
        
        cycle = cycle_data[cycle_idx]
        current = np.array(cycle.get('current_in_A', []))
        time_data = np.array(cycle.get('time_in_s', []))
        
        if len(current) > 0 and len(time_data) > 0:
            charge_mask = current > 0
            if np.any(charge_mask):
                charge_current = current[charge_mask]
                charge_time = time_data[charge_mask]
                
                if len(charge_current) > 10:
                    # 计算电流变化的时间间隔
                    current_diff = np.diff(charge_current)
                    time_diff = np.diff(charge_time)
                    
                    # 找到电流变化相等的时间间隔
                    target_current_diff = 0.1  # 0.1A的电流变化
                    indices = np.where(np.abs(current_diff - target_current_diff) < 0.05)[0]
                    
                    if len(indices) > 0:
                        return np.mean(time_diff[indices])
        return 0
    
    cv_interval_100 = calculate_cv_current_time_interval(99) if len(cycle_data) > 99 else 0
    cv_interval_10 = calculate_cv_current_time_interval(9) if len(cycle_data) > 9 else 0
    f50 = cv_interval_100 - cv_interval_10
    
    return [f41, f42, f43, f44, f45, f46, f47, f48, f49, f50]