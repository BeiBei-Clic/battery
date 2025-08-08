import numpy as np

def calculate_f51_f59_isu(battery_data):
    """计算ISU数据的F51-F59特征，严格按照指导文件定义"""
    
    cycle_data = battery_data.get('cycle_data', [])
    
    # F51: CV阶段充电容量差值
    def get_cv_capacity(cycle_idx):
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
                        cv_capacity = charge_capacity[cv_start:]
                        
                        if len(cv_capacity) > 0:
                            return np.max(cv_capacity) - np.min(cv_capacity)
        return 0
    
    f51 = get_cv_capacity(99) - get_cv_capacity(9)
    
    # F52: CC阶段温度变化率差值 (ISU数据无温度，用功率变化率代替)
    def get_cc_temp_rate(cycle_idx):
        if cycle_idx >= len(cycle_data):
            return 0
        
        cycle = cycle_data[cycle_idx]
        current = np.array(cycle.get('current_in_A', []))
        voltage = np.array(cycle.get('voltage_in_V', []))
        
        if len(current) > 0 and len(voltage) > 0:
            charge_mask = current > 0
            if np.any(charge_mask):
                charge_current = current[charge_mask]
                charge_voltage = voltage[charge_mask]
                
                if len(charge_current) > 10:
                    current_diff = np.abs(np.diff(charge_current))
                    current_std = np.std(charge_current)
                    cv_start_candidates = np.where(current_diff > current_std * 0.1)[0]
                    
                    if len(cv_start_candidates) > 0:
                        cv_start = cv_start_candidates[0]
                        cc_voltage = charge_voltage[:cv_start]
                        cc_current = charge_current[:cv_start]
                        
                        if len(cc_voltage) > 1:
                            power = cc_voltage * cc_current
                            power_change_rate = np.std(np.diff(power))
                            return power_change_rate
        return 0
    
    f52 = get_cc_temp_rate(99) - get_cc_temp_rate(9)
    
    # F53: CV阶段温度变化率差值 (ISU数据无温度，用功率变化率代替)
    def get_cv_temp_rate(cycle_idx):
        if cycle_idx >= len(cycle_data):
            return 0
        
        cycle = cycle_data[cycle_idx]
        current = np.array(cycle.get('current_in_A', []))
        voltage = np.array(cycle.get('voltage_in_V', []))
        
        if len(current) > 0 and len(voltage) > 0:
            charge_mask = current > 0
            if np.any(charge_mask):
                charge_current = current[charge_mask]
                charge_voltage = voltage[charge_mask]
                
                if len(charge_current) > 10:
                    current_diff = np.abs(np.diff(charge_current))
                    current_std = np.std(charge_current)
                    cv_start_candidates = np.where(current_diff > current_std * 0.1)[0]
                    
                    if len(cv_start_candidates) > 0:
                        cv_start = cv_start_candidates[0]
                        cv_voltage = charge_voltage[cv_start:]
                        cv_current = charge_current[cv_start:]
                        
                        if len(cv_voltage) > 1:
                            power = cv_voltage * cv_current
                            power_change_rate = np.std(np.diff(power))
                            return power_change_rate
        return 0
    
    f53 = get_cv_temp_rate(99) - get_cv_temp_rate(9)
    
    # F54: CC充电容量差值
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
    
    f54 = get_cc_capacity(99) - get_cc_capacity(9)
    
    # F55: CV充电容量差值
    f55 = f51  # 与F51相同
    
    # F56: CC充电模式结束时曲线斜率差值 (修复数据点不足问题)
    def get_cc_end_slope(cycle_idx):
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
                        
                        if len(cc_voltage) >= 3:  # 至少3个点才能计算斜率
                            # 计算CC段结束时的斜率（取最后几个点）
                            end_points = min(5, len(cc_voltage))
                            end_voltage = cc_voltage[-end_points:]
                            end_time = cc_time[-end_points:]
                            
                            if len(end_voltage) >= 2:
                                slope = np.polyfit(end_time, end_voltage, 1)[0]
                                return slope
        return 0
    
    f56 = get_cc_end_slope(99) - get_cc_end_slope(9)
    
    # F57: CC充电曲线拐角处垂直斜率差值
    def get_cc_corner_slope(cycle_idx):
        if cycle_idx >= len(cycle_data):
            return 0
        
        cycle = cycle_data[cycle_idx]
        current = np.array(cycle.get('current_in_A', []))
        voltage = np.array(cycle.get('voltage_in_V', []))
        
        if len(current) > 0 and len(voltage) > 0:
            charge_mask = current > 0
            if np.any(charge_mask):
                charge_current = current[charge_mask]
                charge_voltage = voltage[charge_mask]
                
                if len(charge_current) > 10:
                    current_diff = np.abs(np.diff(charge_current))
                    current_std = np.std(charge_current)
                    cv_start_candidates = np.where(current_diff > current_std * 0.1)[0]
                    
                    if len(cv_start_candidates) > 0:
                        cv_start = cv_start_candidates[0]
                        
                        if cv_start > 5 and cv_start < len(charge_voltage) - 5:
                            before_corner = charge_voltage[cv_start-5:cv_start]
                            after_corner = charge_voltage[cv_start:cv_start+5]
                            
                            if len(before_corner) > 0 and len(after_corner) > 0:
                                slope_before = np.polyfit(range(len(before_corner)), before_corner, 1)[0]
                                slope_after = np.polyfit(range(len(after_corner)), after_corner, 1)[0]
                                corner_slope = abs(slope_after - slope_before)
                                return corner_slope
        return 0
    
    f57 = get_cc_corner_slope(99) - get_cc_corner_slope(9)
    
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