import pickle
import numpy as np
import math
import os

def extract_matr_features(battery_data, filename):
    """从MATR数据中提取F1, F5, F11, F14, F59特征"""
    
    if 'cycle_data' not in battery_data:
        return None, None
    
    cycle_data = battery_data['cycle_data']
    
    if not isinstance(cycle_data, list) or len(cycle_data) < 2:
        return None, None
    
    # 提取每个周期的关键数据
    qd_list = []
    chargetime_list = []
    
    for cycle in cycle_data:
        if not isinstance(cycle, dict):
            continue
        
        # 提取放电容量
        if 'discharge_capacity_in_Ah' in cycle:
            capacity_value = cycle['discharge_capacity_in_Ah']
            if isinstance(capacity_value, (list, np.ndarray)):
                capacity_array = np.array(capacity_value)
                non_zero_indices = np.where(capacity_array > 0)[0]
                if len(non_zero_indices) > 0:
                    final_capacity = capacity_array[non_zero_indices[-1]]
                else:
                    final_capacity = 0.0
            else:
                final_capacity = capacity_value
            qd_list.append(final_capacity)
        else:
            qd_list.append(0.0)
        
        # 提取充电时间
        if 'time_in_s' in cycle:
            time_value = cycle['time_in_s']
            if isinstance(time_value, (list, np.ndarray)):
                time_array = np.array(time_value)
                if len(time_array) > 0:
                    total_time = (time_array[-1] - time_array[0]) / 3600
                    chargetime_list.append(total_time)
                else:
                    chargetime_list.append(1.0)
            else:
                chargetime_list.append(time_value / 3600)
        else:
            chargetime_list.append(1.0)
    
    if len(qd_list) < 2:
        return None, None
    
    qd_length = len(qd_list)
    
    # 获取Qdlin数据
    def get_qdlin(cycle_id):
        if cycle_id <= len(cycle_data):
            cycle = cycle_data[cycle_id - 1]
            if isinstance(cycle, dict) and 'Qdlin' in cycle:
                return np.array(cycle['Qdlin'])
        return None
    
    # 获取放电时间
    def get_discharge_time(current):
        discharge_begin = 0
        discharge_end = len(current) - 1
        
        for i in range(len(current) - 1):
            if current[i] >= 0 and current[i + 1] < 0:
                discharge_begin = i + 1
                break
        
        for i in range(discharge_begin, len(current) - 1):
            if current[i] < 0 and current[i + 1] >= 0:
                discharge_end = i
                break
        
        return discharge_begin, discharge_end
    
    # F1和F5: 基于Qdlin - 与collect_1.py一致
    if qd_length >= 10:
        max_cycle = min(100, qd_length)
        cycle_10 = 10
        
        Qdlin_10 = get_qdlin(cycle_10)
        Qdlin_100 = get_qdlin(max_cycle)
        
        if Qdlin_10 is not None and Qdlin_100 is not None and len(Qdlin_10) == len(Qdlin_100):
            # 计算△Q(V)
            Diff_100_10 = Qdlin_100 - Qdlin_10
            
            # F1: △Q(V)最小绝对值的对数 (修正)
            min_abs_diff = np.min(np.abs(Diff_100_10))
            F1 = math.log(abs(min_abs_diff), 10) if min_abs_diff != 0 else -10
            
            # F5: △Q(V)的峰度的对数
            mean = np.mean(Diff_100_10)
            numerator = np.mean((Diff_100_10 - mean) ** 4)
            denominator = (np.mean((Diff_100_10 - mean) ** 2)) ** 2
            if denominator > 0:
                kurtosis = numerator / denominator
                F5 = np.log(abs(kurtosis))
            else:
                F5 = -10
        else:
            F1 = -10
            F5 = -10
    else:
        F1 = -10
        F5 = -10
    
    # F11: 第2周期的放电容量
    F11 = qd_list[1] if len(qd_list) > 1 else 0
    
    # F14: 第2-6周期的平均充电时间
    if len(chargetime_list) >= 6:
        F14 = np.mean(chargetime_list[1:6])
    else:
        F14 = np.mean(chargetime_list[1:min(6, len(chargetime_list))])
    
    # F59: 从第1周期到最大容量周期的总充电时间 (修正为与ISU一致)
    qdischarge = np.array(qd_list[1:min(100, len(qd_list))])
    qdischarge[qdischarge > 1.3] = 0
    max_cycle = np.argmax(qdischarge) + 2
    
    F59 = sum(chargetime_list[:min(max_cycle, len(chargetime_list))])
    all_charge_time = 0
    all_discharge_time = 0
    
    for i in range(min(max_cycle, len(chargetime_list))):
        # 计算放电时间
        if i < len(cycle_data):
            cycle = cycle_data[i]
            if 'current_in_A' in cycle and 'time_in_s' in cycle:
                current = np.array(cycle['current_in_A'])
                timeline = np.array(cycle['time_in_s'])
                dis_begin, dis_end = get_discharge_time(current)
                discharge_time = (timeline[dis_end] - timeline[dis_begin]) / 3600
                all_discharge_time += discharge_time
        
        # 累加充电时间
        temp_idx = i
        while temp_idx < len(chargetime_list) and chargetime_list[temp_idx] > 100 and temp_idx < 100:
            temp_idx += 1
        if temp_idx < len(chargetime_list):
            all_charge_time += chargetime_list[temp_idx]
    
    F59 = all_charge_time + all_discharge_time
    
    # 标签：循环寿命
    y = len(qd_list)
    
    return [F1, F5, F11, F14, F59], y

def process_matr_data():
    """处理MATR数据集"""
    data_dir = "data/MATR"
    
    if not os.path.exists(data_dir):
        print(f"数据目录不存在: {data_dir}")
        return
    
    pkl_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    print(f"找到 {len(pkl_files)} 个MATR文件")
    
    all_features = []
    all_labels = []
    processed_files = []
    
    for filename in pkl_files:
        file_path = os.path.join(data_dir, filename)
        
        with open(file_path, 'rb') as f:
            battery_data = pickle.load(f)
        
        features, label = extract_matr_features(battery_data, filename)
        
        if features is not None:
            all_features.append(features)
            all_labels.append(label)
            processed_files.append(filename)
            print(f"处理完成: {filename}")
    
    # 保存结果
    with open('matr_battery_features.txt', 'w') as f:
        f.write("Battery_Name\tF1\tF5\tF11\tF14\tF59\tCycle_Life\n")
        for i, filename in enumerate(processed_files):
            features = all_features[i]
            label = all_labels[i]
            f.write(f"{filename}\t{features[0]:.6f}\t{features[1]:.6f}\t{features[2]:.6f}\t{features[3]:.6f}\t{features[4]:.6f}\t{label}\n")
    
    print(f"MATR数据处理完成，共处理 {len(processed_files)} 个文件")
    print(f"结果保存到: matr_battery_features.txt")

if __name__ == "__main__":
    process_matr_data()