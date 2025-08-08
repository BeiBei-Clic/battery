import numpy as np
from scipy import stats
from scipy.spatial.distance import directed_hausdorff

def calculate_f41_f50_isu(battery_data):
    """计算ISU数据的F41-F50特征，严格按照指导文件定义"""
    
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
    
    # 计算峰度的辅助函数
    def calculate_kurtosis(data):
        if len(data) < 4:
            return 0
        return stats.kurtosis(data)
    
    # 计算弗雷歇距离的辅助函数
    def calculate_frechet_distance(data):
        if len(data) < 2:
            return 0
        # 简化为数据的标准差作为距离度量
        return np.std(data)
    
    # 修复豪斯多夫距离计算
    def calculate_hausdorff_distance(segment1, segment2):
        if len(segment1) < 2 or len(segment2) < 2:
            return 0
        
        try:
            # 确保数据是2D数组
            if segment1.ndim == 1:
                segment1 = segment1.reshape(-1, 1)
            if segment2.ndim == 1:
                segment2 = segment2.reshape(-1, 1)
                
            # 如果数据点太多，进行采样以提高计算效率
            if len(segment1) > 100:
                indices = np.linspace(0, len(segment1)-1, 100, dtype=int)
                segment1 = segment1[indices]
            if len(segment2) > 100:
                indices = np.linspace(0, len(segment2)-1, 100, dtype=int)
                segment2 = segment2[indices]
            
            # 计算双向豪斯多夫距离
            dist1 = directed_hausdorff(segment1, segment2)[0]
            dist2 = directed_hausdorff(segment2, segment1)[0]
            return max(dist1, dist2)
        except Exception as e:
            # 如果计算失败，使用简化的距离度量
            mean1 = np.mean(segment1, axis=0)
            mean2 = np.mean(segment2, axis=0)
            return np.linalg.norm(mean1 - mean2)
    
    # 获取第10次和第100次循环的充电段数据
    segment1_10, segment2_10 = get_charge_segments(9)   # 第10次循环
    segment1_100, segment2_100 = get_charge_segments(99) # 第100次循环
    
    # F41: CCCV-CCCT段峰度差值
    kurtosis1_10 = calculate_kurtosis(segment1_10[:, 1]) if len(segment1_10) > 0 else 0
    kurtosis1_100 = calculate_kurtosis(segment1_100[:, 1]) if len(segment1_100) > 0 else 0
    f41 = kurtosis1_100 - kurtosis1_10
    
    # F42: CVCC-CVCT段峰度差值
    kurtosis2_10 = calculate_kurtosis(segment2_10[:, 1]) if len(segment2_10) > 0 else 0
    kurtosis2_100 = calculate_kurtosis(segment2_100[:, 1]) if len(segment2_100) > 0 else 0
    f42 = kurtosis2_100 - kurtosis2_10
    
    # F43: CCCV-CCCT段弗雷歇距离差值
    frechet1_10 = calculate_frechet_distance(segment1_10[:, 1]) if len(segment1_10) > 0 else 0
    frechet1_100 = calculate_frechet_distance(segment1_100[:, 1]) if len(segment1_100) > 0 else 0
    f43 = frechet1_100 - frechet1_10
    
    # F44: CVCC-CVCT段弗雷歇距离差值
    frechet2_10 = calculate_frechet_distance(segment2_10[:, 1]) if len(segment2_10) > 0 else 0
    frechet2_100 = calculate_frechet_distance(segment2_100[:, 1]) if len(segment2_100) > 0 else 0
    f44 = frechet2_100 - frechet2_10
    
    # F45: CCCV-CCCT段豪斯多夫距离 (修复计算)
    f45 = calculate_hausdorff_distance(segment1_10, segment1_100) if len(segment1_10) > 0 and len(segment1_100) > 0 else 0
    
    # F46: CVCC-CVCT段豪斯多夫距离 (修复计算)
    f46 = calculate_hausdorff_distance(segment2_10, segment2_100) if len(segment2_10) > 0 and len(segment2_100) > 0 else 0
    
    # F47: 放电后平均电压下降差值 (进一步优化休息期识别)
    def get_voltage_falloff(cycle_idx):
        if cycle_idx >= len(cycle_data):
            return 0
        
        cycle = cycle_data[cycle_idx]
        current = np.array(cycle.get('current_in_A', []))
        voltage = np.array(cycle.get('voltage_in_V', []))
        time_data = np.array(cycle.get('time_in_s', []))
        
        if len(current) > 0 and len(voltage) > 0 and len(time_data) > 0:
            # 找到放电结束点
            discharge_mask = current < -0.01  # 更严格的放电判断
            if np.any(discharge_mask):
                discharge_indices = np.where(discharge_mask)[0]
                if len(discharge_indices) > 0:
                    discharge_end = discharge_indices[-1]
                    
                    # 寻找放电后的休息期
                    after_discharge = discharge_end + 1
                    if after_discharge < len(current) - 20:
                        # 扩大搜索范围
                        rest_current = current[after_discharge:after_discharge+100]
                        rest_voltage = voltage[after_discharge:after_discharge+100]
                        rest_time = time_data[after_discharge:after_discharge+100]
                        
                        # 找到电流接近0的连续区间
                        low_current_mask = np.abs(rest_current) < 0.005  # 更严格的休息期判断
                        if np.sum(low_current_mask) > 10:  # 至少10个连续低电流点
                            rest_voltage_filtered = rest_voltage[low_current_mask]
                            rest_time_filtered = rest_time[low_current_mask]
                            
                            if len(rest_voltage_filtered) > 5:
                                # 计算电压下降率
                                voltage_start = rest_voltage_filtered[0]
                                voltage_end = rest_voltage_filtered[-1]
                                time_span = (rest_time_filtered[-1] - rest_time_filtered[0]) / 1e9
                                
                                if time_span > 60:  # 至少1分钟的休息期
                                    voltage_drop = voltage_start - voltage_end
                                    return voltage_drop / time_span  # V/s
        return 0
    
    f47 = get_voltage_falloff(99) - get_voltage_falloff(9)
    
    # F48: CC阶段电压变化时间间隔差值
    def get_cc_voltage_time_interval(cycle_idx):
        if cycle_idx >= len(cycle_data):
            return 0
        
        cycle = cycle_data[cycle_idx]
        current = np.array(cycle.get('current_in_A', []))
        voltage = np.array(cycle.get('voltage_in_V', []))
        time_data = np.array(cycle.get('time_in_s', []))
        
        if len(current) > 0 and len(voltage) > 0 and len(time_data) > 0:
            charge_mask = current > 0
            if np.any(charge_mask):
                charge_current = current[charge_mask]
                charge_voltage = voltage[charge_mask]
                charge_time = time_data[charge_mask] / 1e9  # 转换为秒
                
                if len(charge_current) > 10:
                    current_diff = np.abs(np.diff(charge_current))
                    current_std = np.std(charge_current)
                    cv_start_candidates = np.where(current_diff > current_std * 0.1)[0]
                    
                    if len(cv_start_candidates) > 0:
                        cv_start = cv_start_candidates[0]
                        cc_voltage = charge_voltage[:cv_start]
                        cc_time = charge_time[:cv_start]
                        
                        if len(cc_voltage) > 1:
                            v_min, v_max = np.min(cc_voltage), np.max(cc_voltage)
                            if v_max > v_min:
                                v_range = v_max - v_min
                                v_low = v_min + 0.3 * v_range
                                v_high = v_min + 0.8 * v_range
                                
                                idx_low = np.where(cc_voltage >= v_low)[0]
                                idx_high = np.where(cc_voltage >= v_high)[0]
                                
                                if len(idx_low) > 0 and len(idx_high) > 0:
                                    time_interval = cc_time[idx_high[0]] - cc_time[idx_low[0]]
                                    return time_interval
        return 0
    
    f48 = get_cc_voltage_time_interval(99) - get_cc_voltage_time_interval(9)
    
    # F49: CC阶段充电容量差值
    def get_cc_capacity(cycle_idx):
        if cycle_idx >= len(cycle_data):
            return 0
        
        cycle = cycle_data[cycle_idx]
        current = np.array(cycle.get('current_in_A', []))
        charge_cap = np.array(cycle.get('charge_capacity_in_Ah', []))
        
        if len(current) > 0 and len(charge_cap) > 0:
            charge_mask = current > 0
            if np.any(charge_mask):
                charge_current = current[charge_mask]
                charge_capacity = charge_cap[charge_mask]
                
                if len(charge_current) > 10:
                    current_diff = np.abs(np.diff(charge_current))
                    current_std = np.std(charge_current)
                    cv_start_candidates = np.where(current_diff > current_std * 0.1)[0]
                    
                    if len(cv_start_candidates) > 0:
                        cv_start = cv_start_candidates[0]
                        cc_capacity = charge_capacity[:cv_start]
                        
                        if len(cc_capacity) > 0:
                            return np.max(cc_capacity) - np.min(cc_capacity)
        return 0
    
    f49 = get_cc_capacity(99) - get_cc_capacity(9)
    
    # F50: CV阶段电流变化时间间隔差值
    def get_cv_current_time_interval(cycle_idx):
        if cycle_idx >= len(cycle_data):
            return 0
        
        cycle = cycle_data[cycle_idx]
        current = np.array(cycle.get('current_in_A', []))
        time_data = np.array(cycle.get('time_in_s', []))
        
        if len(current) > 0 and len(time_data) > 0:
            charge_mask = current > 0
            if np.any(charge_mask):
                charge_current = current[charge_mask]
                charge_time = time_data[charge_mask] / 1e9  # 转换为秒
                
                if len(charge_current) > 10:
                    current_diff = np.abs(np.diff(charge_current))
                    current_std = np.std(charge_current)
                    cv_start_candidates = np.where(current_diff > current_std * 0.1)[0]
                    
                    if len(cv_start_candidates) > 0:
                        cv_start = cv_start_candidates[0]
                        cv_current = charge_current[cv_start:]
                        cv_time = charge_time[cv_start:]
                        
                        if len(cv_current) > 1:
                            i_max, i_min = np.max(cv_current), np.min(cv_current)
                            if i_max > i_min:
                                i_range = i_max - i_min
                                i_high = i_max - 0.2 * i_range
                                i_low = i_min + 0.2 * i_range
                                
                                idx_high = np.where(cv_current <= i_high)[0]
                                idx_low = np.where(cv_current <= i_low)[0]
                                
                                if len(idx_high) > 0 and len(idx_low) > 0:
                                    time_interval = cv_time[idx_low[-1]] - cv_time[idx_high[0]]
                                    return time_interval
        return 0
    
    f50 = get_cv_current_time_interval(99) - get_cv_current_time_interval(9)
    
    return [f41, f42, f43, f44, f45, f46, f47, f48, f49, f50]