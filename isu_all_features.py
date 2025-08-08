import pickle
import numpy as np
import os
from isu.features_qv_stats import calculate_f1_f10
from isu.features_capacity import calculate_f11_f13
from isu.features_time_temp import calculate_f14_f20
from isu.features_segments import calculate_f21_f46
from isu.features_charge_capacity import calculate_f47_f59

def extract_all_isu_features(battery_data, filename):
    """提取ISU数据的所有59个特征"""
    cycle_data = battery_data['cycle_data']
    if len(cycle_data) < 100:
        print(f"跳过 {filename}: 周期数不足100个，实际周期数: {len(cycle_data)}")
        return None, None
    
    # 计算各组特征
    f1_f10 = calculate_f1_f10(cycle_data)
    f11_f13 = calculate_f11_f13(cycle_data)
    f14_f20 = calculate_f14_f20(cycle_data)
    f21_f46 = calculate_f21_f46(cycle_data)
    f47_f59 = calculate_f47_f59(cycle_data)
    
    # 验证特征数量
    print(f"  特征数量检查: F1-F10({len(f1_f10)}), F11-F13({len(f11_f13)}), F14-F20({len(f14_f20)}), F21-F46({len(f21_f46)}), F47-F59({len(f47_f59)})")
    
    # 合并所有特征
    all_features = f1_f10 + f11_f13 + f14_f20 + f21_f46 + f47_f59
    
    # 标签：循环寿命
    y = len(cycle_data)
    
    print(f"  总特征数: {len(all_features)}")
    print(f"  F1-F5: {[f'{x:.3f}' for x in f1_f10[:5]]}")
    print(f"  F11-F13: {[f'{x:.3f}' for x in f11_f13]}")
    print(f"  F14: {f14_f20[0]:.3f}")
    print(f"  F21-F23: {[f'{x:.3f}' for x in f21_f46[:3]]}")
    print(f"  F47-F49: {[f'{x:.3f}' for x in f47_f59[:3]]}")
    
    return all_features, y

def process_isu_all_features():
    """处理ISU数据集提取所有特征"""
    data_dir = "data/ISU_ILCC"
    pkl_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    print(f"找到 {len(pkl_files)} 个ISU文件")
    
    all_features = []
    all_labels = []
    processed_files = []
    
    for filename in pkl_files[:3]:  # 先测试前3个文件
        file_path = os.path.join(data_dir, filename)
        with open(file_path, 'rb') as f:
            battery_data = pickle.load(f)
        
        features, label = extract_all_isu_features(battery_data, filename)
        if features is not None:
            all_features.append(features)
            all_labels.append(label)
            processed_files.append(filename)
            print(f"处理 {filename}，特征数: {len(features)}，标签: {label}")

    # 保存结果
    with open('isu_all_features.txt', 'w') as f:
        # 写入表头
        header = "Battery_Name\t" + "\t".join([f"F{i}" for i in range(1, 60)]) + "\tCycle_Life\n"
        f.write(header)
        
        for i, filename in enumerate(processed_files):
            features = all_features[i]
            label = all_labels[i]
            feature_str = "\t".join([f"{feat:.6f}" for feat in features])
            f.write(f"{filename}\t{feature_str}\t{label}\n")

    print(f"ISU所有特征处理完成，共处理 {len(processed_files)} 个文件")
    print(f"结果保存到: isu_all_features.txt")

if __name__ == "__main__":
    process_isu_all_features()