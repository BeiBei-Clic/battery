import numpy as np

def calculate_f51_f59_isu(battery_data):
    """计算ISU数据的F51-F59特征，严格按照指导文件定义"""
    
    cycle_data = battery_data.get('cycle_data', [])
    
    # 获取第100次循环数据
    def get_cycle_100_data():
        if len(cycle_data) <= 99:
            return None
        return cycle_data[99]  # 第100次循环
    
    cycle_100 = get_cycle_100_data()
    if cycle_100 is None:
        return [0, 0, 0, 0, 0, 0, 0, 1, 0]
    
    # F51: CV阶段4A-0.1A的充电容量
    def get_cv_capacity_4a_01a():
        current = np.array(cycle_100.get('current_in_A', []))
        charge_cap = np.array(cycle_100.get('charge_capacity_in_Ah', []))
        
        if len(current) > 0 and len(charge_cap) > 0:
            charge_mask = current > 0
            if np.any(charge_mask):
                charge_current = current[charge_mask]
                charge_capacity = charge_cap[charge_mask]
                
                # 识别CV段（电流递减）
                if len(charge_current) > 10:
                    current_diff = np.diff(charge_current)
                    decreasing_mask = current_diff < 0
                    
                    if np.sum(decreasing_mask) > 5:
                        cv_start = np.where(decreasing_mask)[0][0]
                        cv_current = charge_current[cv_start:]
                        cv_capacity = charge_capacity[cv_start:]
                        
                        # 查找4A-0.1A范围内的容量
                        if len(cv_current) > 1:
                            range_mask = (cv_current <= 4.0) & (cv_current >= 0.1)
                            if np.sum(range_mask) > 0:
                                capacity_in_range = cv_capacity[range_mask]
                                if len(capacity_in_range) > 0:
                                    return np.max(capacity_in_range) - np.min(capacity_in_range)
        return 0
    
    f51 = get_cv_capacity_4a_01a()
    
    # F52: CC阶段4.0-4.2V的温度变化率（ISU数据无温度，设为0）
    f52 = 0
    
    # F53: CV阶段4A-0.1A的温度变化率（ISU数据无温度，设为0）
    f53 = 0
    
    # F54: CC充电容量（全CC段）
    def get_cc_capacity_all():
        current = np.array(cycle_100.get('current_in_A', []))
        charge_cap = np.array(cycle_100.get('charge_capacity_in_Ah', []))
        
        if len(current) > 0 and len(charge_cap) > 0:
            charge_mask = current > 0
            if np.any(charge_mask):
                charge_current = current[charge_mask]
                charge_capacity = charge_cap[charge_mask]
                
                # 识别CC段（电流相对稳定）
                if len(charge_current) > 10:
                    current_std = np.std(charge_current)
                    current_mean = np.mean(charge_current)
                    
                    # CC段：电流变化小
                    cc_mask = np.abs(charge_current - current_mean) < current_std * 0.2
                    if np.sum(cc_mask) > 5:
                        cc_capacity = charge_capacity[cc_mask]
                        if len(cc_capacity) > 0:
                            return np.max(cc_capacity) - np.min(cc_capacity)
        return 0
    
    f54 = get_cc_capacity_all()
    
    # F55: CV充电容量（全CV段）
    def get_cv_capacity_all():
        current = np.array(cycle_100.get('current_in_A', []))
        charge_cap = np.array(cycle_100.get('charge_capacity_in_Ah', []))
        
        if len(current) > 0 and len(charge_cap) > 0:
            charge_mask = current > 0
            if np.any(charge_mask):
                charge_current = current[charge_mask]
                charge_capacity = charge_cap[charge_mask]
                
                # 识别CV段（电流递减）
                if len(charge_current) > 10:
                    current_diff = np.diff(charge_current)
                    decreasing_mask = current_diff < 0
                    
                    if np.sum(decreasing_mask) > 5:
                        cv_start = np.where(decreasing_mask)[0][0]
                        cv_capacity = charge_capacity[cv_start:]
                        
                        if len(cv_capacity) > 0:
                            return np.max(cv_capacity) - np.min(cv_capacity)
        return 0
    
    f55 = get_cv_capacity_all()
    
    # F56: CC充电模式结束时曲线的斜率
    def get_cc_end_slope():
        current = np.array(cycle_100.get('current_in_A', []))
        voltage = np.array(cycle_100.get('voltage_in_V', []))
        time_data = np.array(cycle_100.get('time_in_s', []))
        
        if len(current) > 0 and len(voltage) > 0 and len(time_data) > 0:
            charge_mask = current > 0
            if np.any(charge_mask):
                charge_current = current[charge_mask]
                charge_voltage = voltage[charge_mask]
                charge_time = time_data[charge_mask] / 1e9  # 转换为秒
                
                # 识别CC段
                if len(charge_current) > 10:
                    current_std = np.std(charge_current)
                    current_mean = np.mean(charge_current)
                    
                    cc_mask = np.abs(charge_current - current_mean) < current_std * 0.2
                    if np.sum(cc_mask) > 5:
                        cc_voltage = charge_voltage[cc_mask]
                        cc_time = charge_time[cc_mask]
                        
                        if len(cc_voltage) >= 3:
                            # 计算CC段结束时的斜率（取最后几个点）
                            end_points = min(5, len(cc_voltage))
                            end_voltage = cc_voltage[-end_points:]
                            end_time = cc_time[-end_points:]
                            
                            if len(end_voltage) >= 2:
                                slope = np.polyfit(end_time, end_voltage, 1)[0]
                                return slope
        return 0
    
    f56 = get_cc_end_slope()
    
    # F57: CC充电曲线拐角处的垂直斜率
    def get_cc_corner_slope():
        current = np.array(cycle_100.get('current_in_A', []))
        voltage = np.array(cycle_100.get('voltage_in_V', []))
        
        if len(current) > 0 and len(voltage) > 0:
            charge_mask = current > 0
            if np.any(charge_mask):
                charge_current = current[charge_mask]
                charge_voltage = voltage[charge_mask]
                
                # 找到CC到CV的转换点（拐角）
                if len(charge_current) > 10:
                    current_diff = np.abs(np.diff(charge_current))
                    current_std = np.std(charge_current)
                    cv_start_candidates = np.where(current_diff > current_std * 0.1)[0]
                    
                    if len(cv_start_candidates) > 0:
                        cv_start = cv_start_candidates[0]
                        
                        if cv_start > 5 and cv_start < len(charge_voltage) - 5:
                            # 计算拐角前后的斜率
                            before_corner = charge_voltage[cv_start-5:cv_start]
                            after_corner = charge_voltage[cv_start:cv_start+5]
                            
                            if len(before_corner) > 1 and len(after_corner) > 1:
                                slope_before = np.polyfit(range(len(before_corner)), before_corner, 1)[0]
                                slope_after = np.polyfit(range(len(after_corner)), after_corner, 1)[0]
                                corner_slope = abs(slope_after - slope_before)
                                return corner_slope
        return 0
    
    f57 = get_cc_corner_slope()
    
    # F58: 最大容量对应的循环次数
    discharge_capacities = []
    for cycle in cycle_data:
        current = np.array(cycle.get('current_in_A', []))
        discharge_cap = np.array(cycle.get('discharge_capacity_in_Ah', []))
        
        if len(current) > 0 and len(discharge_cap) > 0:
            discharge_mask = current < 0
            if np.any(discharge_mask):
                discharge_phase_cap = discharge_cap[discharge_mask]
                max_cap = np.max(discharge_phase_cap) if len(discharge_phase_cap) > 0 else 0
                discharge_capacities.append(max_cap)
            else:
                discharge_capacities.append(0)
        else:
            discharge_capacities.append(0)
    
    if discharge_capacities:
        max_cap_cycle = np.argmax(discharge_capacities) + 1
        f58 = max_cap_cycle
    else:
        f58 = 1
    
    # F59: 最大容量对应的时间
    if discharge_capacities:
        max_cap_idx = np.argmax(discharge_capacities)
        if max_cap_idx < len(cycle_data):
            max_cap_cycle = cycle_data[max_cap_idx]
            time_data = np.array(max_cap_cycle.get('time_in_s', []))
            if len(time_data) > 0:
                f59 = time_data[0] / 1e9  # 转换为秒，取循环开始时间
            else:
                f59 = 0
        else:
            f59 = 0
    else:
        f59 = 0
    
    return [f51, f52, f53, f54, f55, f56, f57, f58, f59]