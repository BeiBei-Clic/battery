import pickle
import numpy as np
import os
import glob

def check_isu_fields():
    """检查ISU-ILCC数据集中的字段结构"""
    print("=== ISU-ILCC数据集字段检查 ===")
    
    isu_dir = "data/ISU_ILCC"
    if not os.path.exists(isu_dir):
        print(f"ISU-ILCC数据目录不存在: {isu_dir}")
        return
    
    isu_files = glob.glob(os.path.join(isu_dir, "*.pkl"))
    print(f"找到 {len(isu_files)} 个ISU-ILCC文件")
    
    all_top_level_keys = set()
    all_cycle_keys = set()
    
    for i, file_path in enumerate(isu_files[:3]):  # 只检查前3个文件
        print(f"\n--- 文件 {i+1}: {os.path.basename(file_path)} ---")
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"数据类型: {type(data)}")
        
        if isinstance(data, dict):
            top_keys = list(data.keys())
            print(f"顶层字段 ({len(top_keys)}个): {top_keys}")
            all_top_level_keys.update(top_keys)
            
            # 检查cycle_data字段
            if 'cycle_data' in data:
                cycle_data = data['cycle_data']
                print(f"cycle_data类型: {type(cycle_data)}")
                print(f"cycle_data长度: {len(cycle_data) if hasattr(cycle_data, '__len__') else 'N/A'}")
                
                if isinstance(cycle_data, list) and len(cycle_data) > 0:
                    # 检查第一个周期的字段
                    first_cycle = cycle_data[0]
                    if isinstance(first_cycle, dict):
                        cycle_keys = list(first_cycle.keys())
                        print(f"第1个周期字段 ({len(cycle_keys)}个): {cycle_keys}")
                        all_cycle_keys.update(cycle_keys)
                        
                        # 显示每个字段的数据类型和示例值
                        for key in cycle_keys:
                            value = first_cycle[key]
                            if isinstance(value, (list, np.ndarray)):
                                print(f"  {key}: {type(value).__name__}[{len(value)}] - 示例: {value[:3] if len(value) > 0 else '空'}")
                            else:
                                print(f"  {key}: {type(value).__name__} - 值: {value}")
                    
                    # 检查最后一个周期
                    if len(cycle_data) > 1:
                        last_cycle = cycle_data[-1]
                        if isinstance(last_cycle, dict):
                            print(f"最后一个周期 (第{len(cycle_data)}个) 字段: {list(last_cycle.keys())}")
            
            # 检查其他顶层字段
            for key in top_keys:
                if key != 'cycle_data':
                    value = data[key]
                    if isinstance(value, (list, np.ndarray)):
                        print(f"{key}: {type(value).__name__}[{len(value)}]")
                    else:
                        print(f"{key}: {type(value).__name__} - {value}")
        else:
            print(f"数据不是字典类型: {type(data)}")
    
    print(f"\n=== 汇总信息 ===")
    print(f"所有顶层字段: {sorted(all_top_level_keys)}")
    print(f"所有周期字段: {sorted(all_cycle_keys)}")
    
    # 查找可能的容量字段
    capacity_fields = [key for key in all_cycle_keys if any(word in key.lower() for word in ['capacity', 'cap', 'qd', 'discharge'])]
    print(f"可能的容量字段: {capacity_fields}")
    
    # 查找可能的时间字段
    time_fields = [key for key in all_cycle_keys if any(word in key.lower() for word in ['time', 'duration', 'charge'])]
    print(f"可能的时间字段: {time_fields}")
    
    # 查找可能的电压字段
    voltage_fields = [key for key in all_cycle_keys if any(word in key.lower() for word in ['voltage', 'volt', 'v'])]
    print(f"可能的电压字段: {voltage_fields}")
    
    # 查找可能的电流字段
    current_fields = [key for key in all_cycle_keys if any(word in key.lower() for word in ['current', 'i', 'amp'])]
    print(f"可能的电流字段: {current_fields}")

if __name__ == "__main__":
    check_isu_fields()