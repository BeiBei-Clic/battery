import numpy as np

def calculate_f11_f20_isu(battery_data):
    """计算ISU数据的F11-F20特征，严格按照指导文件定义"""
    
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
    
    # F11: 第2次循环的放电容量
    f11 = get_discharge_capacity(1)
    
    # F12: 最大放电容量与第2次循环的差值
    all_discharge_caps = []
    for i in range(min(100, len(cycle_data))):
        cap = get_discharge_capacity(i)
        if cap > 0:
            all_discharge_caps.append(cap)
    
    max_discharge_cap = np.max(all_discharge_caps) if len(all_discharge_caps) > 0 else 0
    f12 = max_discharge_cap - f11
    
    # F13: 第100次循环的放电容量
    f13 = get_discharge_capacity(99)
    
    # F14: 前5个循环的平均充电时间
    charge_times = []
    for i in range(min(5, len(cycle_data))):
        cycle = cycle_data[i]
        current = np.array(cycle.get('current_in_A', []))
        time_data = np.array(cycle.get('time_in_s', []))
        
        if len(current) > 0 and len(time_data) > 0:
            charge_mask = current > 0
            if np.any(charge_mask):
                charge_indices = np.where(charge_mask)[0]
                if len(charge_indices) > 1:
                    # ISU数据中时间是纳秒，需要转换为秒
                    charge_start_time = time_data[charge_indices[0]] / 1e9
                    charge_end_time = time_data[charge_indices[-1]] / 1e9
                    charge_duration = charge_end_time - charge_start_time
                    
                    if 600 <= charge_duration <= 36000:  # 10分钟到10小时
                        charge_times.append(charge_duration)
    
    f14 = np.mean(charge_times) if len(charge_times) > 0 else 0
    
    # F15-F17: 温度特征 (ISU数据集中temperature_in_C为None)
    f15 = f16 = f17 = 0
    
    # F18-F20: 内阻特征 (ISU数据集中internal_resistance_in_ohm为None)
    f18 = f19 = f20 = 0
    
    return [f11, f12, f13, f14, f15, f16, f17, f18, f19, f20]