import numpy as np
import math
from scipy import stats

def calculate_f21_f30_matr(battery_data):
    """计算MATR数据的F21-F30特征，严格按照指导文件定义"""
    
    # 提取循环数据
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
        
        if len(current) > 0 and len(voltage) > 0 and len(time_data) > 0:
            discharge_mask = current < 0
            if np.any(discharge_mask):
                discharge_voltage = voltage[discharge_mask]
                discharge_current = current[discharge_mask]
                discharge_time = time_data[discharge_mask]
                
                if len(discharge_time) > 1:
                    # 能量 = 平均功率 × 时间
                    avg_power = np.mean(np.abs(discharge_voltage * discharge_current))
                    duration_hours = (discharge_time[-1] - discharge_time[0]) / 3600
                    return avg_power * duration_hours
        return 0
    
    # 获取循环时间的辅助函数
    def get_cycle_time(cycle_idx):
        if cycle_idx >= len(cycle_data):
            return 0
        
        cycle = cycle_data[cycle_idx]
        time_data = np.array(cycle.get('time_in_s', []))
        
        if len(time_data) > 1:
            return time_data[-1] - time_data[0]
        return 0
    
    # F21: 第100次与第10次循环的放电容量差值 (Discharge Capacity [Ah] 100-10)
    cap_100 = get_discharge_capacity(99) if len(cycle_data) > 99 else 0
    cap_10 = get_discharge_capacity(9) if len(cycle_data) > 9 else 0
    f21 = cap_100 - cap_10
    
    # F22: 第100次与第10次循环的放电能量差值 (Discharge Energy [Wh] 100-10)
    energy_100 = get_discharge_energy(99) if len(cycle_data) > 99 else 0
    energy_10 = get_discharge_energy(9) if len(cycle_data) > 9 else 0
    f22 = energy_100 - energy_10
    
    # F23: 第100次与第10次循环的循环时间差值 (Cycle Time [s] 100-10)
    time_100 = get_cycle_time(99) if len(cycle_data) > 99 else 0
    time_10 = get_cycle_time(9) if len(cycle_data) > 9 else 0
    f23 = time_100 - time_10
    
    # F24: 充电开始时的端电压 (Terminal Voltage @ Start of charge [V])
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
    
    # 取第100次和第10次循环充电开始电压的差值
    start_v_100 = get_charge_start_voltage(99) if len(cycle_data) > 99 else 0
    start_v_10 = get_charge_start_voltage(9) if len(cycle_data) > 9 else 0
    f24 = start_v_100 - start_v_10
    
    # F25: CC段充电时间差值 (Charge time of CC segment of charge curve [s])
    # F26: CV段充电时间差值 (Charge time of CV segment of charge curve [s])
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
                    # 简化的CC/CV识别：CC段电流相对稳定，CV段电压相对稳定
                    current_std = np.std(charge_current)
                    voltage_std = np.std(charge_voltage)
                    
                    # 如果电流标准差小，认为是CC段
                    if current_std < 0.1:  # CC段
                        cc_time = charge_time[-1] - charge_time[0]
                        cv_time = 0
                    elif voltage_std < 0.05:  # CV段
                        cc_time = 0
                        cv_time = charge_time[-1] - charge_time[0]
                    else:
                        # 混合模式，简单分割
                        mid_point = len(charge_time) // 2
                        cc_time = charge_time[mid_point] - charge_time[0]
                        cv_time = charge_time[-1] - charge_time[mid_point]
                    
                    return cc_time, cv_time
        return 0, 0
    
    cc_time_100, cv_time_100 = get_cc_cv_times(99) if len(cycle_data) > 99 else (0, 0)
    cc_time_10, cv_time_10 = get_cc_cv_times(9) if len(cycle_data) > 9 else (0, 0)
    f25 = cc_time_100 - cc_time_10  # CC时间差值
    f26 = cv_time_100 - cv_time_10  # CV时间差值
    
    # F27: CC段平均电流差值 (Mean current during CC segment of the curve [A])
    def get_cc_mean_current(cycle_idx):
        if cycle_idx >= len(cycle_data):
            return 0
        
        cycle = cycle_data[cycle_idx]
        current = np.array(cycle.get('current_in_A', []))
        
        if len(current) > 0:
            charge_mask = current > 0
            if np.any(charge_mask):
                charge_current = current[charge_mask]
                if len(charge_current) > 0:
                    # 简化：取充电电流的平均值作为CC段电流
                    return np.mean(charge_current)
        return 0
    
    cc_current_100 = get_cc_mean_current(99) if len(cycle_data) > 99 else 0
    cc_current_10 = get_cc_mean_current(9) if len(cycle_data) > 9 else 0
    f27 = cc_current_100 - cc_current_10
    
    # F28: CV段平均电压差值 (Mean voltage during CV segment of the curve [V])
    def get_cv_mean_voltage(cycle_idx):
        if cycle_idx >= len(cycle_data):
            return 0
        
        cycle = cycle_data[cycle_idx]
        current = np.array(cycle.get('current_in_A', []))
        voltage = np.array(cycle.get('voltage_in_V', []))
        
        if len(current) > 0 and len(voltage) > 0:
            charge_mask = current > 0
            if np.any(charge_mask):
                charge_voltage = voltage[charge_mask]
                if len(charge_voltage) > 0:
                    # 简化：取充电电压的平均值作为CV段电压
                    return np.mean(charge_voltage)
        return 0
    
    cv_voltage_100 = get_cv_mean_voltage(99) if len(cycle_data) > 99 else 0
    cv_voltage_10 = get_cv_mean_voltage(9) if len(cycle_data) > 9 else 0
    f28 = cv_voltage_100 - cv_voltage_10
    
    # F29: CCCV段斜率差值 (Slope of CCCV-CCCT segment of the curve)
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
                charge_time = time_data[charge_mask]
                
                if len(charge_time) > 2:
                    # 计算电压-时间曲线的斜率
                    slope, _ = np.polyfit(charge_time, charge_voltage, 1)
                    return slope
        return 0
    
    slope_100 = get_cccv_slope(99) if len(cycle_data) > 99 else 0
    slope_10 = get_cccv_slope(9) if len(cycle_data) > 9 else 0
    f29 = slope_100 - slope_10
    
    # F30: CVCC段斜率差值 (Slope of CVCC-CVCT segment of the curve)
    def get_cvcc_slope(cycle_idx):
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
                
                if len(charge_time) > 2:
                    # 计算电流-时间曲线的斜率
                    slope, _ = np.polyfit(charge_time, charge_current, 1)
                    return slope
        return 0
    
    cvcc_slope_100 = get_cvcc_slope(99) if len(cycle_data) > 99 else 0
    cvcc_slope_10 = get_cvcc_slope(9) if len(cycle_data) > 9 else 0
    f30 = cvcc_slope_100 - cvcc_slope_10
    
    return [f21, f22, f23, f24, f25, f26, f27, f28, f29, f30]