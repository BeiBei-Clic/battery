import pickle
import numpy as np
import math
import os
from scipy.interpolate import interp1d

def extract_isu_features(battery_data, filename):
    """从ISU数据中提取特征"""
    cycle_data = battery_data['cycle_data']
    if len(cycle_data) < 100:
        print(f"跳过 {filename}: 周期数不足100个")
        return None, None
    
    # 初始化存储列表
    discharge_capacities = []
    charge_times = []
    qv_curves = []  # 存储每个周期的Q-V曲线 (voltage, capacity)
    
    for cycle in cycle_data:
        # 提取放电容量
        cap_data = cycle.get('discharge_capacity_in_Ah', [])
        if cap_data:
            discharge_capacities.append(np.max(cap_data))
        else:
            discharge_capacities.append(0.0)
        
        # 提取充电时间
        time_data = cycle.get('time_in_s', [])
        if len(time_data) > 1:
            time_diff_ns = time_data[-1] - time_data[0]
            charge_times.append(time_diff_ns / (1e9 * 3600))  # 转换为小时
        else:
            charge_times.append(1.0)
        
        # 提取放电阶段的Q-V曲线用于F1计算
        voltage = np.array(cycle.get('voltage_in_V', []))
        capacity = np.array(cycle.get('discharge_capacity_in_Ah', []))
        current = np.array(cycle.get('current_in_A', []))
        
        if len(voltage) > 0 and len(current) > 0:
            # 找到放电阶段（电流为负值）
            discharge_mask = current < 0
            if np.any(discharge_mask):
                discharge_voltage = voltage[discharge_mask]
                discharge_capacity = capacity[discharge_mask]
                
                if len(discharge_voltage) > 1:
                    # 按电压排序，去除重复值
                    sorted_indices = np.argsort(discharge_voltage)
                    sorted_voltage = discharge_voltage[sorted_indices]
                    sorted_capacity = discharge_capacity[sorted_indices]
                    
                    unique_indices = np.unique(sorted_voltage, return_index=True)[1]
                    if len(unique_indices) > 1:
                        final_voltage = sorted_voltage[unique_indices]
                        final_capacity = sorted_capacity[unique_indices]
                        qv_curves.append((final_voltage, final_capacity))
                        continue
        
        qv_curves.append((np.array([]), np.array([])))
    
    # ---------------------- 特征F1计算 ----------------------
    # 获取第10圈和第100圈的Q-V曲线
    v10, q10 = qv_curves[9]   # 第10圈
    v100, q100 = qv_curves[99] # 第100圈
    
    if len(v10) < 2 or len(v100) < 2:
        # Q-V曲线数据不足，使用简化计算
        diff_capacity = abs(discharge_capacities[99] - discharge_capacities[9])
        F1 = math.log10(diff_capacity) if diff_capacity > 0 else -10
        delta_q = np.array([diff_capacity])
    else:
        # 统一电压维度（插值到共同的电压范围）
        min_v = max(np.min(v10), np.min(v100))
        max_v = min(np.max(v10), np.max(v100))
        
        if min_v >= max_v:
            # 电压范围无交集，使用简化计算
            diff_capacity = abs(discharge_capacities[99] - discharge_capacities[9])
            F1 = math.log10(diff_capacity) if diff_capacity > 0 else -10
            delta_q = np.array([diff_capacity])
        else:
            # 插值计算
            common_v = np.linspace(min_v, max_v, 1000)
            f10 = interp1d(v10, q10, kind='linear', bounds_error=False, fill_value="extrapolate")
            f100 = interp1d(v100, q100, kind='linear', bounds_error=False, fill_value="extrapolate")
            
            q10_interp = f10(common_v)
            q100_interp = f100(common_v)
            
            # 计算△Q(V) = Q100(V) - Q10(V)
            delta_q = q100_interp - q10_interp
            
            # 取△Q(V)的最小绝对值，计算对数
            min_abs_delta = np.min(np.abs(delta_q))
            F1 = math.log10(min_abs_delta) if min_abs_delta > 0 else -10
    
    # ---------------------- 其他特征计算 ----------------------
    # 过滤异常值
    discharge_capacities = [0 if cap > 1.3 else cap for cap in discharge_capacities]
    
    # F5: △Q(V)的峰度的对数
    mean = np.mean(delta_q)
    numerator = np.mean((delta_q - mean) **4)
    denominator = (np.mean((delta_q - mean)** 2)) **2
    F5 = np.log(abs(numerator / denominator)) if denominator > 0 else -10
    
    # F11: 第2周期的放电容量
    F11 = discharge_capacities[1]
    
    # F14: 第2-6周期的平均充电时间
    F14 = np.mean(charge_times[1:6])
    
    # F59: 从第1周期到最大容量周期的总充放电时间
    qdischarge = np.array(discharge_capacities[1:100])
    qdischarge[qdischarge > 1.3] = 0
    max_cycle = np.argmax(qdischarge) + 2
    F59 = sum(charge_times[:max_cycle])
    
    # 标签：循环寿命
    y = len(discharge_capacities)
    
    return [F1, F5, F11, F14, F59], y

def process_isu_data():
    data_dir = "data/ISU_ILCC"
    pkl_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    print(f"找到 {len(pkl_files)} 个ISU文件")
    
    all_features = []
    all_labels = []
    processed_files = []
    
    for filename in pkl_files:
        file_path = os.path.join(data_dir, filename)
        with open(file_path, 'rb') as f:
            battery_data = pickle.load(f)
        
        features, label = extract_isu_features(battery_data, filename)
        if features is not None:
            all_features.append(features)
            all_labels.append(label)
            processed_files.append(filename)
            print(f"处理 {filename}，特征数: {len(features)}，标签: {label}")

    with open('isu_battery_features.txt', 'w') as f:
        f.write("Battery_Name\tF1\tF5\tF11\tF14\tF59\tCycle_Life\n")
        for i, filename in enumerate(processed_files):
            features = all_features[i]
            label = all_labels[i]
            f.write(f"{filename}\t{features[0]:.6f}\t{features[1]:.6f}\t{features[2]:.6f}\t{features[3]:.6f}\t{features[4]:.6f}\t{label}\n")

    print(f"ISU数据处理完成，共处理 {len(processed_files)} 个文件")
    print(f"结果保存到: isu_battery_features.txt")

if __name__ == "__main__":
    process_isu_data()