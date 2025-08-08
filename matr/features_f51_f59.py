import numpy as np
import math
from scipy import stats

def calculate_f51_f59_matr(battery_data):
    """计算MATR数据的F51-F59特征，严格按照指导文件定义"""
    
    # 提取循环数据
    cycle_data = battery_data.get('cycle_data', [])
    
    # F51: CV阶段充电容量差值 (Charge capacity during CV phase [Ah])
    def calculate_cv_charge_capacity(cycle_idx):
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
                charge_time = time_data[charge_mask]
                
                if len(charge_current) > 10:
                    # 简化的CV段识别：电压相对稳定的部分
                    voltage_std = np.std(charge_voltage)
                    if voltage_std < 0.05:  # CV段
                        dt = np.diff(charge_time) / 3600  # 转换为小时
                        capacity = np.sum(charge_current[:-1] * dt)
                        return capacity
                    else:
                        # 取后半段作为CV段
                        mid_point = len(charge_current) // 2
                        cv_current = charge_current[mid_point:]
                        cv_time = charge_time[mid_point:]
                        if len(cv_current) > 1:
                            dt = np.diff(cv_time) / 3600
                            capacity = np.sum(cv_current[:-1] * dt)
                            return capacity
        return 0
    
    cv_capacity_100 = calculate_cv_charge_capacity(99) if len(cycle_data) > 99 else 0
    cv_capacity_10 = calculate_cv_charge_capacity(9) if len(cycle_data) > 9 else 0
    f51 = cv_capacity_100 - cv_capacity_10
    
    # F52: CC阶段温度变化率差值 (Temperature change rate during CC phase [°C/s])
    def calculate_cc_temp_change_rate(cycle_idx):
        if cycle_idx >= len(cycle_data):
            return 0
        
        cycle = cycle_data[cycle_idx]
        current = np.array(cycle.get('current_in_A', []))
        time_data = np.array(cycle.get('time_in_s', []))
        
        # 检查多种可能的温度字段名
        temp_fields = ['temperature_in_C', 'temp_in_C', 'T_in_C', 'temperature', 'temp']
        temp_data = None
        
        for temp_field in temp_fields:
            if temp_field in cycle:
                temp_data = np.array(cycle[temp_field])
                if len(temp_data) > 0 and not np.all(np.isnan(temp_data)):
                    break
        
        if temp_data is not None and len(current) > 0 and len(time_data) > 0:
            charge_mask = current > 0
            if np.any(charge_mask):
                charge_temp = temp_data[charge_mask]
                charge_time = time_data[charge_mask]
                
                if len(charge_temp) > 10:
                    # CC段：前半段
                    mid_point = len(charge_temp) // 2
                    cc_temp = charge_temp[:mid_point]
                    cc_time = charge_time[:mid_point]
                    
                    if len(cc_temp) > 2:
                        temp_change = np.diff(cc_temp)
                        time_change = np.diff(cc_time)
                        temp_change_rate = temp_change / time_change
                        return np.mean(temp_change_rate)
        return 0
    
    cc_temp_rate_100 = calculate_cc_temp_change_rate(99) if len(cycle_data) > 99 else 0
    cc_temp_rate_10 = calculate_cc_temp_change_rate(9) if len(cycle_data) > 9 else 0
    f52 = cc_temp_rate_100 - cc_temp_rate_10
    
    # F53: CV阶段温度变化率差值 (Temperature change rate during CV phase [°C/s])
    def calculate_cv_temp_change_rate(cycle_idx):
        if cycle_idx >= len(cycle_data):
            return 0
        
        cycle = cycle_data[cycle_idx]
        current = np.array(cycle.get('current_in_A', []))
        time_data = np.array(cycle.get('time_in_s', []))
        
        # 检查多种可能的温度字段名
        temp_fields = ['temperature_in_C', 'temp_in_C', 'T_in_C', 'temperature', 'temp']
        temp_data = None
        
        for temp_field in temp_fields:
            if temp_field in cycle:
                temp_data = np.array(cycle[temp_field])
                if len(temp_data) > 0 and not np.all(np.isnan(temp_data)):
                    break
        
        if temp_data is not None and len(current) > 0 and len(time_data) > 0:
            charge_mask = current > 0
            if np.any(charge_mask):
                charge_temp = temp_data[charge_mask]
                charge_time = time_data[charge_mask]
                
                if len(charge_temp) > 10:
                    # CV段：后半段
                    mid_point = len(charge_temp) // 2
                    cv_temp = charge_temp[mid_point:]
                    cv_time = charge_time[mid_point:]
                    
                    if len(cv_temp) > 2:
                        temp_change = np.diff(cv_temp)
                        time_change = np.diff(cv_time)
                        temp_change_rate = temp_change / time_change
                        return np.mean(temp_change_rate)
        return 0
    
    cv_temp_rate_100 = calculate_cv_temp_change_rate(99) if len(cycle_data) > 99 else 0
    cv_temp_rate_10 = calculate_cv_temp_change_rate(9) if len(cycle_data) > 9 else 0
    f53 = cv_temp_rate_100 - cv_temp_rate_10
    
    # F54: CC充电容量差值 (CC charge capacity [Ah])
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
                
                if len(charge_current) > 10:
                    # CC段：前半段
                    mid_point = len(charge_current) // 2
                    cc_current = charge_current[:mid_point]
                    cc_time = charge_time[:mid_point]
                    
                    if len(cc_current) > 1:
                        dt = np.diff(cc_time) / 3600  # 转换为小时
                        capacity = np.sum(cc_current[:-1] * dt)
                        return capacity
        return 0
    
    cc_capacity_100 = calculate_cc_charge_capacity(99) if len(cycle_data) > 99 else 0
    cc_capacity_10 = calculate_cc_charge_capacity(9) if len(cycle_data) > 9 else 0
    f54 = cc_capacity_100 - cc_capacity_10
    
    # F55: CV充电容量差值 (CV charge capacity [Ah])
    f55 = f51  # 与F51相同
    
    # F56: CC充电模式结束时曲线斜率差值 (Slope at the end of CC charge mode [V/s])
    def calculate_cc_end_slope(cycle_idx):
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
                    # CC段结束时的斜率：取前半段的最后几个点
                    mid_point = len(charge_voltage) // 2
                    end_voltage = charge_voltage[mid_point-5:mid_point]
                    end_time = charge_time[mid_point-5:mid_point]
                    
                    if len(end_voltage) > 2:
                        slope, _ = np.polyfit(end_time, end_voltage, 1)
                        return slope
        return 0
    
    cc_slope_100 = calculate_cc_end_slope(99) if len(cycle_data) > 99 else 0
    cc_slope_10 = calculate_cc_end_slope(9) if len(cycle_data) > 9 else 0
    f56 = cc_slope_100 - cc_slope_10
    
    # F57: CC充电曲线拐角处垂直斜率差值 (Vertical slope at the corner of CC charge curve [A/s])
    def calculate_cc_corner_slope(cycle_idx):
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
                    # 拐角处：CC到CV的转换点，大约在中间位置
                    mid_point = len(charge_current) // 2
                    corner_current = charge_current[mid_point-3:mid_point+3]
                    corner_time = charge_time[mid_point-3:mid_point+3]
                    
                    if len(corner_current) > 2:
                        slope, _ = np.polyfit(corner_time, corner_current, 1)
                        return slope
        return 0
    
    corner_slope_100 = calculate_cc_corner_slope(99) if len(cycle_data) > 99 else 0
    corner_slope_10 = calculate_cc_corner_slope(9) if len(cycle_data) > 9 else 0
    f57 = corner_slope_100 - corner_slope_10
    
    # F58: 最大容量对应的循环次数 (Cycle number corresponding to maximum capacity)
    def get_max_capacity_cycle():
        max_capacity = 0
        max_cycle = 1
        
        for i in range(min(100, len(cycle_data))):
            cycle = cycle_data[i]
            cap_data = np.array(cycle.get('discharge_capacity_in_Ah', []))
            
            if len(cap_data) > 0:
                cycle_capacity = np.max(cap_data[cap_data > 0]) if np.any(cap_data > 0) else 0
                if cycle_capacity > max_capacity:
                    max_capacity = cycle_capacity
                    max_cycle = i + 1
        
        return max_cycle
    
    f58 = get_max_capacity_cycle()
    
    # F59: 最大容量对应的时间 (Time corresponding to maximum capacity [h])
    def get_max_capacity_time():
        max_capacity = 0
        max_time = 0
        
        for i in range(min(100, len(cycle_data))):
            cycle = cycle_data[i]
            cap_data = np.array(cycle.get('discharge_capacity_in_Ah', []))
            time_data = np.array(cycle.get('time_in_s', []))
            
            if len(cap_data) > 0 and len(time_data) > 0:
                cycle_capacity = np.max(cap_data[cap_data > 0]) if np.any(cap_data > 0) else 0
                if cycle_capacity > max_capacity:
                    max_capacity = cycle_capacity
                    max_time = time_data[-1] / 3600  # 转换为小时
        
        return max_time
    
    f59 = get_max_capacity_time()
    
    return [f51, f52, f53, f54, f55, f56, f57, f58, f59]