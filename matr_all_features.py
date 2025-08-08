import os
import numpy as np
import pickle
from matr.features_f1_f10 import calculate_f1_f10_matr
from matr.features_f11_f20 import calculate_f11_f20_matr
from matr.features_f21_f30 import calculate_f21_f30_matr
from matr.features_f31_f40 import calculate_f31_f40_matr
from matr.features_f41_f50 import calculate_f41_f50_matr
from matr.features_f51_f59 import calculate_f51_f59_matr

def extract_matr_all_features():
    """提取MATR数据集的F1-F59所有特征"""
    
    # MATR数据路径
    matr_data_path = "d:/desktop/电池寿命预测/ISU+MATR/data/MATR"
    output_file = "d:/desktop/电池寿命预测/ISU+MATR/matr_all_features.txt"
    
    # 获取所有电池文件
    battery_files = [f for f in os.listdir(matr_data_path) if f.endswith('.pkl')]
    
    all_features = []
    
    for battery_file in battery_files:
        battery_path = os.path.join(matr_data_path, battery_file)
        battery_name = battery_file.replace('.pkl', '')
        
        print(f"处理电池: {battery_name}")
        
        # 加载电池数据
        with open(battery_path, 'rb') as f:
            battery_data = pickle.load(f)
        
        # 提取循环数据
        cycle_data = battery_data.get('cycle_data', [])
        
        if len(cycle_data) < 10:
            print(f"警告: {battery_name} 循环数据不足10次，跳过")
            continue
        
        # 计算F1-F10特征
        f1_f10 = calculate_f1_f10_matr(battery_data)
        
        # 计算F11-F20特征
        f11_f20 = calculate_f11_f20_matr(battery_data)
        
        # 计算F21-F30特征
        f21_f30 = calculate_f21_f30_matr(battery_data)
        
        # 计算F31-F40特征
        f31_f40 = calculate_f31_f40_matr(battery_data)
        
        # 计算F41-F50特征
        f41_f50 = calculate_f41_f50_matr(battery_data)
        
        # 计算F51-F59特征
        f51_f59 = calculate_f51_f59_matr(battery_data)
        
        # 合并所有特征
        all_battery_features = f1_f10 + f11_f20 + f21_f30 + f31_f40 + f41_f50 + f51_f59
        
        # 获取循环寿命
        cycle_life = len(cycle_data)
        
        # 保存特征：电池名称 + F1-F59特征 + 循环寿命
        feature_row = [battery_name] + all_battery_features + [cycle_life]
        all_features.append(feature_row)
        
        print(f"完成 {battery_name}: F1-F59特征已计算，循环寿命={cycle_life}")
    
    # 保存所有特征到文件
    with open(output_file, 'w') as f:
        # 写入表头
        header = ['Battery_Name'] + [f'F{i}' for i in range(1, 60)] + ['Cycle_Life']
        f.write('\t'.join(header) + '\n')
        
        # 写入数据
        for row in all_features:
            f.write('\t'.join(map(str, row)) + '\n')
    
    print(f"\n所有特征已保存到: {output_file}")
    print(f"总共处理了 {len(all_features)} 个电池")
    print(f"每个电池包含 59 个特征")

if __name__ == "__main__":
    extract_matr_all_features()