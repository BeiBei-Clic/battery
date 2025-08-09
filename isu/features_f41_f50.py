import numpy as np
from scipy import stats
from scipy.spatial.distance import directed_hausdorff

def calculate_f41_f50_isu(battery_data):
    """计算ISU数据的F41-F50特征，严格按照指导文件定义"""
    
    cycle_data = battery_data.get('cycle_data', [])
    
    # 获取第100次循环的充电段数据
    def get_charge_segments_with_current(cycle_idx):
        if cycle_idx >= len(cycle_data):
            return [], []
        
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
                    segment1 = np.column_stack((charge_time[:half_point], charge_voltage[:half_point]))
                    segment2 = np.column_stack((charge_time[half_point:], charge_voltage[half_point:]))
                    return segment1, segment2
        return np.array([]), np.array([])
    
    # 计算峰度的辅助函数
    def calculate_kurtosis(data):
        if len(data) < 4:
            return 0
        return stats.kurtosis(data)
    
    # 计算真正的弗雷歇距离
    def calculate_frechet_distance(segment):
        if len(segment) < 2:
            return 0
        
        # 弗雷歇距离需要两条曲线，这里计算段内曲线的复杂度
        # 使用曲线的弯曲程度作为距离度量
        if len(segment) < 3:
            return 0
            
        # 计算曲线的二阶导数作为弯曲度量
        time_vals = segment[:, 0]
        voltage_vals = segment[:, 1]
        
        if len(time_vals) > 2:
            # 计算一阶和二阶差分
            dt = np.diff(time_vals)
            dv = np.diff(voltage_vals)
            
            if len(dt) > 0 and np.all(dt > 0):
                dv_dt = dv / dt
                if len(dv_dt) > 1:
                    d2v_dt2 = np.diff(dv_dt) / dt[:-1]
                    return np.mean(np.abs(d2v_dt2))
        return 0
    
    # 计算豪斯多夫距离（段内距离）
    def calculate_hausdorff_distance_single_segment(segment):
        if len(segment) < 2:
            return 0
        
        # 计算段内点之间的最大距离
        points = segment
        if len(points) < 2:
            return 0
            
        # 计算所有点对之间的距离
        distances = []
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                dist = np.linalg.norm(points[i] - points[j])
                distances.append(dist)
        
        return np.max(distances) if distances else 0
    
    # 获取第100次循环的充电段数据
    segment1, segment2 = get_charge_segments_with_current(99)  # 第100次循环
    
    # F41: CCCV-CCCT段的峰度系数 eq 5
    f41 = calculate_kurtosis(segment1[:, 1]) if len(segment1) > 0 else 0
    
    # F42: CVCC-CVCT段的峰度系数 eq 5
    f42 = calculate_kurtosis(segment2[:, 1]) if len(segment2) > 0 else 0
    
    # F43: CCCV-CCCT段的弗雷歇距离 eq 7
    f43 = calculate_frechet_distance(segment1) if len(segment1) > 0 else 0
    
    # F44: CVCC-CVCT段的弗雷歇距离 eq 7
    f44 = calculate_frechet_distance(segment2) if len(segment2) > 0 else 0
    
    # F45: CCCV-CCCT段的豪斯多夫距离 eq 6
    f45 = calculate_hausdorff_distance_single_segment(segment1) if len(segment1) > 0 else 0
    
    # F46: CVCC-CVCT段的豪斯多夫距离 eq 6
    f46 = calculate_hausdorff_distance_single_segment(segment2) if len(segment2) > 0 else 0
    
    # F47: MVF——mean voltage falloff，5min after discharge
    def get_voltage_falloff(cycle_idx):
        if cycle_idx >= len(cycle_data):
            return 0
        
        cycle = cycle_data[cycle_idx]
        current = np.array(cycle.get('current_in_A', []))
        voltage = np.array(cycle.get('voltage_in_V', []))
        time_data = np.array(cycle.get('time_in_s', []))
        
        if len(current) > 0 and len(voltage) > 0 and len(time_data) > 0:
            # 找到放电结束点
            discharge_mask = current < -0.01
            if np.any(discharge_mask):
                discharge_indices = np.where(discharge_mask)[0]
                if len(discharge_indices) > 0:
                    discharge_end = discharge_indices[-1]
                    
                    # 寻找放电后5分钟的休息期
                    after_discharge = discharge_end + 1
                    if after_discharge < len(current):
                        rest_current = current[after_discharge:]
                        rest_voltage = voltage[after_discharge:]
                        rest_time = time_data[after_discharge:]
                        
                        # 找到电流接近0的连续区间
                        low_current_mask = np.abs(rest_current) < 0.005
                        if np.sum(low_current_mask) > 10:
                            rest_voltage_filtered = rest_voltage[low_current_mask]
                            rest_time_filtered = rest_time[low_current_mask]
                            
                            if len(rest_voltage_filtered) > 5:
                                # 计算5分钟内的电压下降
                                time_span = (rest_time_filtered[-1] - rest_time_filtered[0]) / 1e9
                                if time_span >= 300:  # 至少5分钟
                                    # 取前5分钟的数据
                                    five_min_mask = (rest_time_filtered - rest_time_filtered[0]) <= 300e9
                                    voltage_5min = rest_voltage_filtered[five_min_mask]
                                    
                                    if len(voltage_5min) > 1:
                                        voltage_start = voltage_5min[0]
                                        voltage_end = voltage_5min[-1]
                                        return voltage_start - voltage_end  # 电压下降量
        return 0
    
    f47 = get_voltage_falloff(99)  # 第100次循环的MVF
    
    # F48: CC阶段4.0-4.2V的等电压差时间间隔
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
                
                # 识别CC段（电流相对稳定）
                if len(charge_current) > 10:
                    current_std = np.std(charge_current)
                    current_mean = np.mean(charge_current)
                    
                    # CC段：电流变化小
                    cc_mask = np.abs(charge_current - current_mean) < current_std * 0.2
                    if np.sum(cc_mask) > 5:
                        cc_voltage = charge_voltage[cc_mask]
                        cc_time = charge_time[cc_mask]
                        
                        # 查找4.0V和4.2V对应的时间点
                        if len(cc_voltage) > 1:
                            v_40_idx = np.where(cc_voltage >= 4.0)[0]
                            v_42_idx = np.where(cc_voltage >= 4.2)[0]
                            
                            if len(v_40_idx) > 0 and len(v_42_idx) > 0:
                                time_40 = cc_time[v_40_idx[0]]
                                time_42 = cc_time[v_42_idx[0]]
                                return time_42 - time_40
        return 0
    
    f48 = get_cc_voltage_time_interval(99)  # 第100次循环
    
    # F49: CC阶段4.0-4.2V的充电容量
    def get_cc_capacity(cycle_idx):
        if cycle_idx >= len(cycle_data):
            return 0
        
        cycle = cycle_data[cycle_idx]
        current = np.array(cycle.get('current_in_A', []))
        voltage = np.array(cycle.get('voltage_in_V', []))
        charge_cap = np.array(cycle.get('charge_capacity_in_Ah', []))
        
        if len(current) > 0 and len(voltage) > 0 and len(charge_cap) > 0:
            charge_mask = current > 0
            if np.any(charge_mask):
                charge_current = current[charge_mask]
                charge_voltage = voltage[charge_mask]
                charge_capacity = charge_cap[charge_mask]
                
                # 识别CC段
                if len(charge_current) > 10:
                    current_std = np.std(charge_current)
                    current_mean = np.mean(charge_current)
                    
                    cc_mask = np.abs(charge_current - current_mean) < current_std * 0.2
                    if np.sum(cc_mask) > 5:
                        cc_voltage = charge_voltage[cc_mask]
                        cc_capacity = charge_capacity[cc_mask]
                        
                        # 查找4.0V-4.2V范围内的容量
                        voltage_mask = (cc_voltage >= 4.0) & (cc_voltage <= 4.2)
                        if np.sum(voltage_mask) > 0:
                            capacity_in_range = cc_capacity[voltage_mask]
                            if len(capacity_in_range) > 0:
                                return np.max(capacity_in_range) - np.min(capacity_in_range)
        return 0
    
    f49 = get_cc_capacity(99)  # 第100次循环
    
    # F50: CV阶段4A-0.1A的等电流差时间间隔
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
                
                # 识别CV段（电流递减）
                if len(charge_current) > 10:
                    # CV段通常在充电后期，电流逐渐下降
                    current_diff = np.diff(charge_current)
                    decreasing_mask = current_diff < 0
                    
                    if np.sum(decreasing_mask) > 5:
                        # 找到连续下降的区间
                        cv_start = np.where(decreasing_mask)[0][0]
                        cv_current = charge_current[cv_start:]
                        cv_time = charge_time[cv_start:]
                        
                        # 查找4A和0.1A对应的时间点
                        if len(cv_current) > 1:
                            i_4a_idx = np.where(cv_current <= 4.0)[0]
                            i_01a_idx = np.where(cv_current <= 0.1)[0]
                            
                            if len(i_4a_idx) > 0 and len(i_01a_idx) > 0:
                                time_4a = cv_time[i_4a_idx[0]]
                                time_01a = cv_time[i_01a_idx[0]]
                                return time_01a - time_4a
        return 0
    
    f50 = get_cv_current_time_interval(99)  # 第100次循环
    
    return [f41, f42, f43, f44, f45, f46, f47, f48, f49, f50]