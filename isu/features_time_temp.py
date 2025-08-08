import numpy as np

def calculate_f14_f20(cycle_data):
    """计算F14-F20充电时间、温度及内阻特征"""
    
    # F14: 前5次循环的平均充电时间
    charge_times = []
    for i in range(min(5, len(cycle_data))):
        cycle = cycle_data[i]
        time_data = np.array(cycle.get('time_in_s', []))
        current_data = np.array(cycle.get('current_in_A', []))
        
        if len(time_data) > 1 and len(current_data) > 0:
            charge_mask = current_data > 0
            if np.any(charge_mask):
                charge_times_ns = time_data[charge_mask]
                if len(charge_times_ns) > 1:
                    charge_duration_hours = (charge_times_ns[-1] - charge_times_ns[0]) / (1e9 * 3600)
                    charge_times.append(charge_duration_hours)
    
    f14 = np.mean(charge_times) if charge_times else 1.0
    
    # F15-F17: 温度特征 (ISU数据集中temperature_in_C为None，设为0)
    f15 = f16 = f17 = 0
    
    # F18-F20: 内阻特征 (ISU数据集中internal_resistance_in_ohm为None，设为0)
    f18 = f19 = f20 = 0
    
    return [f14, f15, f16, f17, f18, f19, f20]