import numpy as np

def calculate_f21_f30_isu(battery_data):
    """计算ISU数据的F21-F30特征，严格按照指导文件定义"""
    
    cycle_data = battery_data.get('cycle_data', [])
    
    # 获取放电容量的辅助函数
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
    
    # 获取放电能量的辅助函数
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
                    dt = np.diff(discharge_time) / 1e9  # 转换为秒
                    power = discharge_voltage[:-1] * abs(discharge_current[:-1])
                    energy = np.sum(power * dt) / 3600  # 转换为Wh
                    return energy
        return 0
    
    # 获取循环时间的辅助函数
    def get_cycle_time(cycle_idx):
        if cycle_idx >= len(cycle_data):
            return 0
        
        cycle = cycle_data[cycle_idx]
        time_data = np.array(cycle.get('time_in_s', []))
        
        if len(time_data) > 1:
            return (time_data[-1] - time_data[0]) / (1e9 * 3600)  # 转换为小时
        return 0
    
    # 获取充电开始端电压的辅助函数
    def get_charge_start_voltage(cycle_idx):
        if cycle_idx >= len(cycle_data):
            return 0
        
        cycle = cycle_data[cycle_idx]
        current = np.array(cycle.get('current_in_A', []))
        voltage = np.array(cycle.get('voltage_in_V', []))
        
        if len(current) > 0 and len(voltage) > 0:
            charge_mask = current > 0
            if np.any(charge_mask):
                charge_voltage = voltage[charge_mask]
                return charge_voltage[0] if len(charge_voltage) > 0 else 0
        return 0
    
    # F21: 第100次与第10次循环的放电容量差值
    f21 = get_discharge_capacity(99) - get_discharge_capacity(9)
    
    # F22: 第100次与第10次循环的放电能量差值
    f22 = get_discharge_energy(99) - get_discharge_energy(9)
    
    # F23: 第100次与第10次循环的循环时间差值
    f23 = get_cycle_time(99) - get_cycle_time(9)
    
    # F24: 第100次与第10次循环充电开始时的端电压差值
    f24 = get_charge_start_voltage(99) - get_charge_start_voltage(9)
    
    # F25-F26: CC段和CV段充电时间差值
    def get_cc_cv_times(cycle_idx):
        if cycle_idx >= len(cycle_data):
            return 0, 0
        
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
                    # 基于电流变化识别CC/CV转换点
                    current_diff = np.abs(np.diff(charge_current))
                    current_std = np.std(charge_current)
                    cv_start_candidates = np.where(current_diff > current_std * 0.1)[0]
                    
                    if len(cv_start_candidates) > 0:
                        cv_start = cv_start_candidates[0]
                        cc_time = (charge_time[cv_start] - charge_time[0]) / 1e9
                        cv_time = (charge_time[-1] - charge_time[cv_start]) / 1e9
                        return cc_time, cv_time
        return 0, 0
    
    cc_time_10, cv_time_10 = get_cc_cv_times(9)
    cc_time_100, cv_time_100 = get_cc_cv_times(99)
    
    f25 = cc_time_100 - cc_time_10  # CC段充电时间差值
    f26 = cv_time_100 - cv_time_10  # CV段充电时间差值
    
    # F27: CC段平均电流差值
    def get_cc_mean_current(cycle_idx):
        if cycle_idx >= len(cycle_data):
            return 0
        
        cycle = cycle_data[cycle_idx]
        current = np.array(cycle.get('current_in_A', []))
        
        if len(current) > 0:
            charge_mask = current > 0
            if np.any(charge_mask):
                charge_current = current[charge_mask]
                if len(charge_current) > 10:
                    current_diff = np.abs(np.diff(charge_current))
                    current_std = np.std(charge_current)
                    cv_start_candidates = np.where(current_diff > current_std * 0.1)[0]
                    
                    if len(cv_start_candidates) > 0:
                        cv_start = cv_start_candidates[0]
                        cc_current = charge_current[:cv_start]
                        return np.mean(cc_current) if len(cc_current) > 0 else 0
        return 0
    
    f27 = get_cc_mean_current(99) - get_cc_mean_current(9)
    
    # F28: CV段平均电压差值
    def get_cv_mean_voltage(cycle_idx):
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
                        return np.mean(cv_voltage) if len(cv_voltage) > 0 else 0
        return 0
    
    f28 = get_cv_mean_voltage(99) - get_cv_mean_voltage(9)
    
    # F29: CCCV段斜率差值 (修复时间单位转换问题)
    def get_cccv_slope(cycle_idx):
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
                charge_time = time_data[charge_mask] / 1e9  # 转换为秒
                
                if len(charge_voltage) > 1 and len(charge_time) > 1:
                    slope = np.polyfit(charge_time, charge_voltage, 1)[0]
                    return slope
        return 0
    
    f29 = get_cccv_slope(99) - get_cccv_slope(9)
    
    # F30: CVCC段斜率差值 (修复时间单位转换问题)
    def get_cvcc_slope(cycle_idx):
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
                charge_time = time_data[charge_mask] / 1e9  # 转换为秒
                
                if len(charge_voltage) > 2:
                    # 取后半段数据
                    half_point = len(charge_voltage) // 2
                    voltage_half = charge_voltage[half_point:]
                    time_half = charge_time[half_point:]
                    
                    if len(voltage_half) > 1:
                        slope = np.polyfit(time_half, voltage_half, 1)[0]
                        return slope
        return 0
    
    f30 = get_cvcc_slope(99) - get_cvcc_slope(9)
    
    return [f21, f22, f23, f24, f25, f26, f27, f28, f29, f30]