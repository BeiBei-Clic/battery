import numpy as np

def calculate_f21_f30_isu(battery_data):
    """计算ISU数据的F21-F30特征，严格按照指导文件定义"""
    
    cycle_data = battery_data.get('cycle_data', [])
    
    # 获取放电容量
    def get_discharge_capacity(cycle_idx):
        if cycle_idx >= len(cycle_data):
            return 0
        
        cycle = cycle_data[cycle_idx]
        current = np.array(cycle.get('current_in_A', []))
        cap_data = np.array(cycle.get('discharge_capacity_in_Ah', []))
        
        if len(current) > 0 and len(cap_data) > 0:
            discharge_mask = current < 0
            if np.any(discharge_mask):
                discharge_phase_cap = cap_data[discharge_mask]
                valid_caps = discharge_phase_cap[discharge_phase_cap > 0]
                return np.max(valid_caps) if len(valid_caps) > 0 else 0
        return 0
    
    # 获取放电能量
    def get_discharge_energy(cycle_idx):
        if cycle_idx >= len(cycle_data):
            return 0
        
        cycle = cycle_data[cycle_idx]
        current = np.array(cycle.get('current_in_A', []))
        voltage = np.array(cycle.get('voltage_in_V', []))
        time_data = np.array(cycle.get('time_in_s', []))
        
        if len(current) > 0 and len(voltage) > 0 and len(time_data) > 1:
            discharge_mask = current < 0
            if np.any(discharge_mask):
                discharge_voltage = voltage[discharge_mask]
                discharge_current = current[discharge_mask]
                discharge_time = time_data[discharge_mask]
                if len(discharge_time) > 1:
                    dt = np.diff(discharge_time) / 1e9  # 纳秒转秒
                    power = discharge_voltage[:-1] * abs(discharge_current[:-1])
                    energy = np.sum(power * dt) / 3600  # 转换为Wh
                    return energy
        return 0
    
    # 获取循环时间
    def get_cycle_time(cycle_idx):
        if cycle_idx >= len(cycle_data):
            return 0
        
        cycle = cycle_data[cycle_idx]
        time_data = np.array(cycle.get('time_in_s', []))
        
        if len(time_data) > 1:
            return (time_data[-1] - time_data[0]) / 1e9  # 纳秒转秒
        return 0
    
    # F21: Discharge Capacity [Ah] 100-10 (差值)
    f21 = get_discharge_capacity(99) - get_discharge_capacity(9)
    
    # F22: Discharge Energy [Wh] 100-10 (差值)
    f22 = get_discharge_energy(99) - get_discharge_energy(9)
    
    # F23: Cycle Time [s] 100-10 (差值)
    f23 = get_cycle_time(99) - get_cycle_time(9)
    
    # F24: Terminal Voltage @ Start of charge [V] (单次值，取第100次循环)
    def get_charge_start_voltage(cycle_idx):
        if cycle_idx >= len(cycle_data):
            return 0
        
        cycle = cycle_data[cycle_idx]
        current = np.array(cycle.get('current_in_A', []))
        voltage = np.array(cycle.get('voltage_in_V', []))
        
        if len(current) > 0 and len(voltage) > 0:
            charge_mask = current > 0
            if np.any(charge_mask):
                charge_indices = np.where(charge_mask)[0]
                if len(charge_indices) > 0:
                    return voltage[charge_indices[0]]
        return 0
    
    f24 = get_charge_start_voltage(99)  # 第100次循环的充电开始端电压
    
    # 获取CC/CV段数据
    def get_cc_cv_data(cycle_idx):
        if cycle_idx >= len(cycle_data):
            return None, None, None, None, None, None
        
        cycle = cycle_data[cycle_idx]
        current = np.array(cycle.get('current_in_A', []))
        voltage = np.array(cycle.get('voltage_in_V', []))
        time_data = np.array(cycle.get('time_in_s', []))
        
        if len(current) > 0 and len(voltage) > 0 and len(time_data) > 0:
            charge_mask = current > 0
            if np.any(charge_mask):
                charge_current = current[charge_mask]
                charge_voltage = voltage[charge_mask]
                charge_time = time_data[charge_mask] / 1e9  # 纳秒转秒
                
                if len(charge_current) > 10:
                    # 基于电流变化识别CC/CV转换点
                    current_diff = np.abs(np.diff(charge_current))
                    threshold = np.std(charge_current) * 0.5
                    
                    cv_start_idx = 0
                    for i in range(len(current_diff)):
                        if current_diff[i] > threshold:
                            cv_start_idx = i + 1
                            break
                    
                    if cv_start_idx > 0 and cv_start_idx < len(charge_current) - 1:
                        cc_current = charge_current[:cv_start_idx]
                        cc_voltage = charge_voltage[:cv_start_idx]
                        cc_time = charge_time[:cv_start_idx]
                        
                        cv_current = charge_current[cv_start_idx:]
                        cv_voltage = charge_voltage[cv_start_idx:]
                        cv_time = charge_time[cv_start_idx:]
                        
                        return cc_current, cc_voltage, cc_time, cv_current, cv_voltage, cv_time
                    else:
                        # 简单分割：前半段CC，后半段CV
                        mid_point = len(charge_current) // 2
                        cc_current = charge_current[:mid_point]
                        cc_voltage = charge_voltage[:mid_point]
                        cc_time = charge_time[:mid_point]
                        
                        cv_current = charge_current[mid_point:]
                        cv_voltage = charge_voltage[mid_point:]
                        cv_time = charge_time[mid_point:]
                        
                        return cc_current, cc_voltage, cc_time, cv_current, cv_voltage, cv_time
        
        return None, None, None, None, None, None
    
    # F25: Charge time of CC segment [s] (单次值，取第100次循环)
    cc_current, cc_voltage, cc_time, _, _, _ = get_cc_cv_data(99)
    if cc_time is not None and len(cc_time) > 1:
        f25 = cc_time[-1] - cc_time[0]
    else:
        f25 = 0
    
    # F26: Charge time of CV segment [s] (单次值，取第100次循环)
    _, _, _, cv_current, cv_voltage, cv_time = get_cc_cv_data(99)
    if cv_time is not None and len(cv_time) > 1:
        f26 = cv_time[-1] - cv_time[0]
    else:
        f26 = 0
    
    # F27: Mean current during CC segment [A] (单次值，取第100次循环)
    if cc_current is not None and len(cc_current) > 0:
        f27 = np.mean(cc_current)
    else:
        f27 = 0
    
    # F28: Mean voltage during CV segment [V] (单次值，取第100次循环)
    if cv_voltage is not None and len(cv_voltage) > 0:
        f28 = np.mean(cv_voltage)
    else:
        f28 = 0
    
    # F29: Slope of CCCV-CCCT segment (单次值，取第100次循环的CC段电压-时间斜率)
    if cc_voltage is not None and cc_time is not None and len(cc_voltage) > 2:
        slope, _ = np.polyfit(cc_time, cc_voltage, 1)
        f29 = slope
    else:
        f29 = 0
    
    # F30: Slope of CVCC-CVCT segment (单次值，取第100次循环的CV段电流-时间斜率)
    if cv_current is not None and cv_time is not None and len(cv_current) > 2:
        slope, _ = np.polyfit(cv_time, cv_current, 1)
        f30 = slope
    else:
        f30 = 0
    
    return [f21, f22, f23, f24, f25, f26, f27, f28, f29, f30]