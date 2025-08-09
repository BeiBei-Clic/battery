import numpy as np
import math

def calculate_f11_f20_matr(battery_data):
    """计算MATR数据的F11-F20特征，严格按照指导文件定义"""
    
    cycle_data = battery_data.get('cycle_data', [])
    
    # MATR数据的cycle_data是列表，不是字典
    if not isinstance(cycle_data, list) or len(cycle_data) < 2:
        return [0] * 10
    
    # 获取放电容量的辅助函数
    def get_discharge_capacity(cycle_idx):
        if cycle_idx >= len(cycle_data):
            return 0
        
        cycle = cycle_data[cycle_idx]
        if not isinstance(cycle, dict):
            return 0
            
        # 检查放电容量字段
        capacity_fields = ['discharge_capacity_in_Ah', 'discharge_capacity', 'capacity']
        for field in capacity_fields:
            if field in cycle:
                capacity_value = cycle[field]
                if isinstance(capacity_value, (list, np.ndarray)):
                    capacity_array = np.array(capacity_value)
                    # 获取放电阶段的最大容量
                    valid_caps = capacity_array[capacity_array > 0]
                    return np.max(valid_caps) if len(valid_caps) > 0 else 0
                else:
                    return capacity_value if capacity_value > 0 else 0
        return 0
    
    # F11: 第2次循环的放电容量 (Discharge capacity, cycle 2)
    f11 = get_discharge_capacity(1)  # 索引1对应第2次循环
    
    # F12: 最大放电容量与第2次循环的差值 (Difference between max discharge capacity and cycle 2)
    # 计算所有周期的放电容量，找到最大值
    all_discharge_caps = []
    for i in range(min(100, len(cycle_data))):
        cap = get_discharge_capacity(i)
        if cap > 0:
            all_discharge_caps.append(cap)
    
    max_discharge_cap = np.max(all_discharge_caps) if len(all_discharge_caps) > 0 else 0
    f12 = max_discharge_cap - f11
    
    # F13: 第100次循环的放电容量 (Discharge capacity, cycle 100)
    f13 = get_discharge_capacity(99)  # 索引99对应第100次循环
    
    # F14: 前5个循环的平均充电时间 (Average charge time, first 5 cycles)
    charge_times = []
    for i in range(min(5, len(cycle_data))):
        cycle = cycle_data[i]
        if not isinstance(cycle, dict):
            continue
            
        # 检查电流和时间字段
        current_fields = ['current_in_A', 'current', 'I']
        time_fields = ['time_in_s', 'time', 'timestamp']
        
        current = None
        time_data = None
        
        for field in current_fields:
            if field in cycle:
                current = np.array(cycle[field])
                break
                
        for field in time_fields:
            if field in cycle:
                time_data = np.array(cycle[field])
                break
        
        if current is not None and time_data is not None and len(current) > 0 and len(time_data) > 0:
            # 找到充电阶段（电流>0）
            charge_mask = current > 0
            if np.any(charge_mask):
                charge_indices = np.where(charge_mask)[0]
                if len(charge_indices) > 1:
                    charge_start_time = time_data[charge_indices[0]]
                    charge_end_time = time_data[charge_indices[-1]]
                    charge_duration = charge_end_time - charge_start_time
                    
                    # 过滤异常值：充电时间应该在合理范围内
                    if 600 <= charge_duration <= 36000:  # 10分钟到10小时
                        charge_times.append(charge_duration)
    
    f14 = np.mean(charge_times) if len(charge_times) > 0 else 0
    
    # F15: 第2-100次循环的最高温度 (Maximum temperature, cycles 2 to 100)
    # F16: 第2-100次循环的最低温度 (Minimum temperature, cycles 2 to 100)  
    # F17: 第2-100次循环的温度积分 (Integral of temperature over time, cycles 2 to 100)
    temp_data_all = []
    temp_time_integral = 0
    
    for i in range(1, min(100, len(cycle_data))):  # 第2-100次循环
        cycle = cycle_data[i]
        if not isinstance(cycle, dict):
            continue
            
        # 检查时间字段
        time_data = None
        for field in ['time_in_s', 'time', 'timestamp']:
            if field in cycle:
                time_data = np.array(cycle[field])
                break
        
        # 检查多种可能的温度字段名
        temp_fields = ['temperature_in_C', 'temp_in_C', 'T_in_C', 'temperature', 'temp']
        temp_data = None
        
        for temp_field in temp_fields:
            if temp_field in cycle:
                temp_data = np.array(cycle[temp_field])
                if len(temp_data) > 0 and not np.all(np.isnan(temp_data)):
                    break
        
        if temp_data is not None and len(temp_data) > 0 and time_data is not None and len(time_data) > 0:
            valid_temp = temp_data[~np.isnan(temp_data)]
            if len(valid_temp) > 0:
                temp_data_all.extend(valid_temp)
                
                # 计算温度-时间积分（简化为平均温度×时间）
                if len(time_data) > 1:
                    cycle_duration = time_data[-1] - time_data[0]
                    avg_temp = np.mean(valid_temp)
                    temp_time_integral += avg_temp * cycle_duration
    
    if len(temp_data_all) > 0:
        f15 = np.max(temp_data_all)  # 最高温度
        f16 = np.min(temp_data_all)  # 最低温度
        f17 = temp_time_integral     # 温度积分
    else:
        f15 = f16 = f17 = 0
    
    # F18: 第2次循环的内阻 (Internal resistance, cycle 2) - MATR数据集中无
    # F19: 第2-100次循环的最小内阻 (Minimum internal resistance, cycles 2 to 100) - MATR数据集中无
    # F20: 第100次与第2次循环的内阻差值 (Internal resistance, difference between cycle 100 and cycle 2) - MATR数据集中无
    f18 = f19 = f20 = 0
    
    return [f11, f12, f13, f14, f15, f16, f17, f18, f19, f20]