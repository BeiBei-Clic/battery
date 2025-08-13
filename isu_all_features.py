import pickle
import numpy as np
import os
from datetime import datetime  # 导入datetime模块获取当前时间
from isu.features_f1_f10 import calculate_f1_f10_isu
from isu.features_f11_f20 import calculate_f11_f20_isu
from isu.features_f21_f30 import calculate_f21_f30_isu
from isu.features_f31_f40 import calculate_f31_f40_isu
from isu.features_f41_f50 import calculate_f41_f50_isu
from isu.features_f51_f59 import calculate_f51_f59_isu

def extract_all_isu_features(battery_data, filename):
    """提取ISU数据的所有59个特征"""
    cycle_data = battery_data['cycle_data']
    if len(cycle_data) < 100:
        print(f"跳过 {filename}: 周期数不足100个，实际周期数: {len(cycle_data)}")
        return None, None
    
    # 计算各组特征
    f1_f10 = calculate_f1_f10_isu(battery_data)
    f11_f20 = calculate_f11_f20_isu(battery_data)
    f21_f30 = calculate_f21_f30_isu(battery_data)
    f31_f40 = calculate_f31_f40_isu(battery_data)
    f41_f50 = calculate_f41_f50_isu(battery_data)
    f51_f59 = calculate_f51_f59_isu(battery_data)
    
    # 验证特征数量
    print(f"  特征数量检查: F1-F10({len(f1_f10)}), F11-F20({len(f11_f20)}), F21-F30({len(f21_f30)}), F31-F40({len(f31_f40)}), F41-F50({len(f41_f50)}), F51-F59({len(f51_f59)})")
    
    # 合并所有特征
    all_features = f1_f10 + f11_f20 + f21_f30 + f31_f40 + f41_f50 + f51_f59
    
    # 标签：循环寿命
    y = len(cycle_data)
        
    return all_features, y

def process_isu_all_features():
    """处理ISU数据集提取所有特征"""
    data_dir = "data/ISU_ILCC"
    pkl_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    print(f"找到 {len(pkl_files)} 个ISU文件")
    
    all_features = []
    all_labels = []
    processed_files = []
    
    for filename in pkl_files :  

        file_path = os.path.join(data_dir, filename)
        with open(file_path, 'rb') as f:
            battery_data = pickle.load(f)
        
        features, label = extract_all_isu_features(battery_data, filename)
        if features is not None:
            all_features.append(features)
            all_labels.append(label)
            processed_files.append(filename)
            print(f"处理 {filename}，特征数: {len(features)}，标签: {label}")

    # 获取当前时间并格式化为"月日时分"
    current_time = datetime.now().strftime("%m%d%H%M")
    # 构建带时间戳的文件名
    output_filename = f"./result/isu_{current_time}.txt"

    
    # 保存结果
    with open(output_filename, 'w') as f:
        # 写入表头
        header = "Battery_Name\t" + "\t".join([f"F{i}" for i in range(1, 60)]) + "\tCycle_Life\n"
        f.write(header)
        
        for i, filename in enumerate(processed_files):
            features = all_features[i]
            label = all_labels[i]
            feature_str = "\t".join([f"{feat:.6f}" for feat in features])
            f.write(f"{filename}\t{feature_str}\t{label}\n")

    print(f"ISU所有特征处理完成，共处理 {len(processed_files)} 个文件")
    print(f"结果保存到: {output_filename}")

if __name__ == "__main__":
    process_isu_all_features()
