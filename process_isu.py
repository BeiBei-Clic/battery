import pickle
import numpy as np
import math
import os

def extract_isu_features(battery_data, filename):
    """从ISU数据中提取F1, F5, F11, F14, F59特征"""
    
    if 'cycle_data' not in battery_data:
        return None, None
    
    cycle_data = battery_data['cycle_data']
    
    if not isinstance(cycle_data, list) or len(cycle_data) < 100:
        print(f"跳过 {filename}: 周期数不足100个")
        return None, None
    
    # 提取放电容量和时间序列
    discharge_capacities = []
    charge_times = []
    
    for cycle in cycle_data:
        if isinstance(cycle, dict):
            # 提取放电容量
            if 'discharge_capacity_in_Ah' in cycle:
                capacity_data = cycle['discharge_capacity_in_Ah']
                if isinstance(capacity_data, (list, np.ndarray)) and len(capacity_data) > 0:
                    discharge_capacities.append(capacity_data[-1])
                else:
                    discharge_capacities.append(0.0)
            else:
                discharge_capacities.append(0.0)
            
            # 提取时间 - ISU数据的time_in_s是纳秒时间戳
            if 'time_in_s' in cycle:
                time_data = cycle['time_in_s']
                if isinstance(time_data, (list, np.ndarray)) and len(time_data) > 0:
                    time_diff_ns = time_data[-1] - time_data[0]
                    time_hours = time_diff_ns / (1e9 * 3600)  # 纳秒转小时
                    charge_times.append(time_hours)
                else:
                    charge_times.append(1.0)
            else:
                charge_times.append(1.0)
    
    if len(discharge_capacities) < 100:
        print(f"跳过 {filename}: 有效容量数据不足")
        return None, None
    
    # 过滤异常值
    for i in range(len(discharge_capacities)):
        if discharge_capacities[i] > 1.3:
            discharge_capacities[i] = 0
    
    # F1: △Q(V)最小值的绝对值的对数 - 与collect_1.py第26行一致
    capacity_10 = discharge_capacities[9]   # 第10周期
    capacity_100 = discharge_capacities[99] # 第100周期
    capacities_100 = np.array(discharge_capacities[:100])
    capacities_10_extended = np.full_like(capacities_100, capacity_10)
    diff_array = capacities_100 - capacities_10_extended
    min_diff = np.min(diff_array)
    F1 = math.log(abs(min_diff), 10) if min_diff != 0 else -10
    
    # F5: △Q(V)的峰度的对数 - 与collect_1.py第134行一致
    mean = np.mean(diff_array)
    numerator = np.mean((diff_array - mean) ** 4)
    denominator = (np.mean((diff_array - mean) ** 2)) ** 2
    if denominator > 0:
        kurtosis = numerator / denominator
        F5 = np.log(abs(kurtosis))
    else:
        F5 = -10
    
    # F11: 第2周期的放电容量 - 与collect_1.py第48行一致
    F11 = discharge_capacities[1]
    
    # F14: 第2-6周期的平均充电时间 - 与collect_1.py第54行一致
    if len(charge_times) >= 6:
        F14 = np.mean(charge_times[1:6])
    else:
        F14 = np.mean(charge_times[1:min(6, len(charge_times))])
    
    # F59: 从第1周期到最大容量周期的总充放电时间 - 与collect_3.py第218行一致
    qdischarge = np.array(discharge_capacities[1:100])
    qdischarge[qdischarge > 1.3] = 0  # 过滤异常值
    max_cycle = np.argmax(qdischarge) + 2  # +2因为从第2周期开始
    
    total_time = 0
    for i in range(min(max_cycle, len(charge_times))):
        if i < len(charge_times):
            total_time += charge_times[i]
    F59 = total_time
    
    # 标签：循环寿命
    y = len(discharge_capacities)
    
    return [F1, F5, F11, F14, F59], y

def process_isu_data():
    """处理ISU数据集"""
    data_dir = "data/ISU_ILCC"
    
    if not os.path.exists(data_dir):
        print(f"数据目录不存在: {data_dir}")
        return
    
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

    # 保存结果
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