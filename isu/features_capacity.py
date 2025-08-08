import numpy as np

def calculate_f11_f13(cycle_data):
    """计算F11-F13放电容量特征（基于文档collect features_XJTU.docx定义）"""
    # 存储每个循环的放电容量（仅保留放电阶段的最大容量，代表单次循环总放电量）
    discharge_capacities = []
    
    # 遍历每个循环的数据
    for cycle in cycle_data:
        # 从循环数据中提取电流和放电容量数组
        current = np.array(cycle.get('current_in_A', []))  # 电流数据（单位：A）
        discharge_cap = np.array(cycle.get('discharge_capacity_in_Ah', []))  # 放电容量（单位：Ah）
        
        # 检查电流和容量数据是否有效（数组非空）
        if len(current) > 0 and len(discharge_cap) > 0:
            # 筛选放电阶段：电流为负值（放电时电流方向与充电相反）
            discharge_mask = current < 0
            if np.any(discharge_mask):  # 存在放电阶段数据
                # 提取放电阶段的容量数据
                discharge_phase_cap = discharge_cap[discharge_mask]
                if len(discharge_phase_cap) > 0:
                    # 放电容量为累积值，取最大值作为该循环的总放电容量
                    max_discharge = np.max(discharge_phase_cap)
                    discharge_capacities.append(max_discharge)
                else:
                    # 放电阶段无有效容量数据，填充0
                    discharge_capacities.append(0.0)
            else:
                # 无放电阶段（异常情况），填充0
                discharge_capacities.append(0.0)
        else:
            # 电流或容量数据为空（无效），填充0
            discharge_capacities.append(0.0)
    
    # F11：第2次循环的放电容量（文档定义："Discharge capacity, cycle 2"）
    # 注意：数组索引从0开始，第2次循环对应索引1
    f11 = discharge_capacities[1] if len(discharge_capacities) > 1 else 0
    
    # F12：最大放电容量与第2次循环放电容量的差值（文档定义："Difference between max discharge capacity and cycle 2"）
    max_capacity = np.max(discharge_capacities) if discharge_capacities else 0  # 所有循环中的最大放电容量
    f12 = max_capacity - f11  # 差值计算
    
    # F13：第100次循环的放电容量（文档定义："Discharge capacity, cycle 100"）
    # 第100次循环对应索引99（索引从0开始）
    f13 = discharge_capacities[99] if len(discharge_capacities) > 99 else 0
    
    return [f11, f12, f13]