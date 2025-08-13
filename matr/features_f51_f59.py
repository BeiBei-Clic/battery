import numpy as np
import math
from scipy import stats

def calculate_f51_f59_matr(battery_data):
    """计算MATR数据的F51-F59特征，严格按照指导文件定义"""
    
    # 提取循环数据
    cycle_data = battery_data.get('cycle_data', [])
    
    # 获取第100次循环数据
    if len(cycle_data) > 99:
        cycle_100 = cycle_data[99]
    else:
        # 如果没有第100次循环，使用最后一次循环
        cycle_100 = cycle_data[-1] if cycle_data else {}
    
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
    
    # F52: CC阶段4.0-4.2V的温度变化率
    def get_cc_temp_change_rate_4v():
        current = np.array(cycle_100.get('current_in_A', []))
        voltage = np.array(cycle_100.get('voltage_in_V', []))
        time_data = np.array(cycle_100.get('time_in_s', []))
        
        # 检查多种可能的温度字段名
        temp_fields = ['temperature_in_C', 'temp_in_C', 'T_in_C', 'temperature', 'temp']
        temp_data = None
        
        for temp_field in temp_fields:
            if temp_field in cycle_100:
                temp_data = np.array(cycle_100[temp_field])
                if len(temp_data) > 0 and not np.all(np.isnan(temp_data)):
                    break
        
        if temp_data is not None and len(current) > 0 and len(voltage) > 0 and len(time_data) > 0:
            charge_mask = current > 0
            if np.any(charge_mask):
                charge_current = current[charge_mask]
                charge_voltage = voltage[charge_mask]
                charge_temp = temp_data[charge_mask]
                charge_time = time_data[charge_mask]
                
                # 识别CC段并限定4.0-4.2V范围
                if len(charge_current) > 10:
                    current_std = np.std(charge_current)
                    current_mean = np.mean(charge_current)
                    cc_mask = np.abs(charge_current - current_mean) < current_std * 0.2
                    
                    if np.sum(cc_mask) > 5:
                        cc_voltage = charge_voltage[cc_mask]
                        cc_temp = charge_temp[cc_mask]
                        cc_time = charge_time[cc_mask]
                        
                        # 限定4.0-4.2V范围
                        voltage_mask = (cc_voltage >= 4.0) & (cc_voltage <= 4.2)
                        if np.sum(voltage_mask) > 2:
                            range_temp = cc_temp[voltage_mask]
                            range_time = cc_time[voltage_mask]
                            
                            if len(range_temp) > 1:
                                temp_change = np.diff(range_temp)
                                time_change = np.diff(range_time)
                                temp_change_rate = temp_change / time_change
                                return np.mean(temp_change_rate)
        return 0
    
    f52 = get_cc_temp_change_rate_4v()
    
    # F53: CV阶段4A-0.1A的温度变化率
    def get_cv_temp_change_rate_4a():
        current = np.array(cycle_100.get('current_in_A', []))
        time_data = np.array(cycle_100.get('time_in_s', []))
        
        # 检查多种可能的温度字段名
        temp_fields = ['temperature_in_C', 'temp_in_C', 'T_in_C', 'temperature', 'temp']
        temp_data = None
        
        for temp_field in temp_fields:
            if temp_field in cycle_100:
                temp_data = np.array(cycle_100[temp_field])
                if len(temp_data) > 0 and not np.all(np.isnan(temp_data)):
                    break
        
        if temp_data is not None and len(current) > 0 and len(time_data) > 0:
            charge_mask = current > 0
            if np.any(charge_mask):
                charge_current = current[charge_mask]
                charge_temp = temp_data[charge_mask]
                charge_time = time_data[charge_mask]
                
                # 识别CV段并限定4A-0.1A范围
                if len(charge_current) > 10:
                    current_diff = np.diff(charge_current)
                    decreasing_mask = current_diff < 0
                    
                    if np.sum(decreasing_mask) > 5:
                        cv_start = np.where(decreasing_mask)[0][0]
                        cv_current = charge_current[cv_start:]
                        cv_temp = charge_temp[cv_start:]
                        cv_time = charge_time[cv_start:]
                        
                        # 限定4A-0.1A范围
                        current_mask = (cv_current <= 4.0) & (cv_current >= 0.1)
                        if np.sum(current_mask) > 2:
                            range_temp = cv_temp[current_mask]
                            range_time = cv_time[current_mask]
                            
                            if len(range_temp) > 1:
                                temp_change = np.diff(range_temp)
                                time_change = np.diff(range_time)
                                temp_change_rate = temp_change / time_change
                                return np.mean(temp_change_rate)
        return 0
    
    f53 = get_cv_temp_change_rate_4a()
    
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
                charge_time = time_data[charge_mask]  # MATR数据已经是秒，不需要转换
                
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
    
    # F58: 最大容量对应的循环次数 (保持原有实现，与文档一致)
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
    
    # F59: 达到最大容量时的累计时间
    def get_c_dc_time():
        """计算从初始循环到最大放电容量所在循环的总充电时间与总放电时间之和"""
        
        # 1. 定位最大放电容量所在的循环
        # 提取前100次循环的放电容量数据
        qdischarge = []
        for i in range(1, min(100, len(cycle_data))):  # 从第2次循环开始（索引1）
            cycle = cycle_data[i]
            current = np.array(cycle.get('current_in_A', []))
            discharge_cap = np.array(cycle.get('discharge_capacity_in_Ah', []))
            
            if len(current) > 0 and len(discharge_cap) > 0:
                discharge_mask = current < 0
                if np.any(discharge_mask):
                    discharge_phase_cap = discharge_cap[discharge_mask]
                    max_cap = np.max(discharge_phase_cap) if len(discharge_phase_cap) > 0 else 0
                    qdischarge.append(max_cap)
                else:
                    qdischarge.append(0)
            else:
                qdischarge.append(0)
        
        if not qdischarge:
            return 0
        
        # 过滤异常值：将放电容量大于1.3的值置为0
        qdischarge = np.array(qdischarge)
        qdischarge[qdischarge > 1.3] = 0
        
        # 找到最大放电容量对应的循环索引
        max_qd_index = np.argmax(qdischarge) + 2  # 加2是因为从第2次循环开始计数
        
        # 2. 计算累计充电时间与放电时间
        all_discharge_time = 0
        all_charge_time = 0
        
        # 遍历从第1次循环到最大容量所在循环
        for cycle_idx in range(min(max_qd_index, len(cycle_data))):
            cycle = cycle_data[cycle_idx]
            current = np.array(cycle.get('current_in_A', []))
            time_data = np.array(cycle.get('time_in_s', []))
            
            if len(current) > 0 and len(time_data) > 0:
                # 计算放电时间
                discharge_time = get_discharge_time(cycle)
                all_discharge_time += discharge_time
                
                # 计算充电时间
                charge_time = get_charge_time(cycle)
                if charge_time > 100:  # 时间异常处理
                    # 尝试从后续循环获取合理值
                    for next_idx in range(cycle_idx + 1, min(cycle_idx + 5, len(cycle_data))):
                        next_charge_time = get_charge_time(cycle_data[next_idx])
                        if next_charge_time <= 100:
                            charge_time = next_charge_time
                            break
                    else:
                        charge_time = 0  # 如果都异常，则设为0
                
                all_charge_time += charge_time
        
        # 3. 计算F59的值
        charge_and_dis_time = all_charge_time + all_discharge_time
        return charge_and_dis_time
    
    def get_discharge_time(cycle):
        """获取单个循环的放电时间"""
        current = np.array(cycle.get('current_in_A', []))
        time_data = np.array(cycle.get('time_in_s', []))
        
        if len(current) > 0 and len(time_data) > 0:
            discharge_mask = current < 0
            if np.any(discharge_mask):
                discharge_indices = np.where(discharge_mask)[0]
                if len(discharge_indices) > 0:
                    start_time = time_data[discharge_indices[0]]
                    end_time = time_data[discharge_indices[-1]]
                    duration = end_time - start_time  # MATR数据已经是秒，直接使用
                    return duration
        return 0
    
    def get_charge_time(cycle):
        """获取单个循环的充电时间"""
        current = np.array(cycle.get('current_in_A', []))
        time_data = np.array(cycle.get('time_in_s', []))
        
        if len(current) > 0 and len(time_data) > 0:
            charge_mask = current > 0
            if np.any(charge_mask):
                charge_indices = np.where(charge_mask)[0]
                if len(charge_indices) > 0:
                    start_time = time_data[charge_indices[0]]
                    end_time = time_data[charge_indices[-1]]
                    duration = end_time - start_time  # MATR数据已经是秒，直接使用
                    return duration
        return 0
    
    f59 = get_c_dc_time()
    
    return [f51, f52, f53, f54, f55, f56, f57, f58, f59]