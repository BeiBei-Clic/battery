import pickle
import numpy as np
import os
import glob

def check_charge_discharge_order():
    """判断MATR数据集中电池是先充电还是先放电"""
    print("=== MATR数据集充放电顺序检查 ===")
    
    matr_dir = "data/MATR"
    if not os.path.exists(matr_dir):
        print(f"MATR数据目录不存在: {matr_dir}")
        return
    
    matr_files = glob.glob(os.path.join(matr_dir, "*.pkl"))
    print(f"找到 {len(matr_files)} 个MATR文件")
    
    charge_first_count = 0
    discharge_first_count = 0
    unclear_count = 0
    
    # 检查前10个文件作为样本
    sample_files = matr_files[:10]
    
    for i, file_path in enumerate(sample_files):
        print(f"\n--- 文件 {i+1}: {os.path.basename(file_path)} ---")
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        if 'cycle_data' not in data:
            print("未找到cycle_data字段")
            continue
            
        cycle_data = data['cycle_data']
        if not cycle_data or len(cycle_data) == 0:
            print("cycle_data为空")
            continue
        
        # 检查第一个周期
        first_cycle = cycle_data[0]
        
        # 方法1: 检查电流方向
        current_data = first_cycle.get('current_in_A', [])
        if current_data and len(current_data) > 1:
            # 找到第一个非零电流值
            first_nonzero_current = None
            for current in current_data:
                if abs(current) > 0.01:  # 忽略很小的电流值
                    first_nonzero_current = current
                    break
            
            if first_nonzero_current is not None:
                if first_nonzero_current > 0:
                    current_direction = "充电 (正电流)"
                else:
                    current_direction = "放电 (负电流)"
                print(f"电流方向判断: {current_direction}, 首个非零电流: {first_nonzero_current:.4f}A")
        
        # 方法2: 检查容量变化
        charge_capacity = first_cycle.get('charge_capacity_in_Ah', [])
        discharge_capacity = first_cycle.get('discharge_capacity_in_Ah', [])
        
        charge_starts_first = False
        discharge_starts_first = False
        
        if charge_capacity and discharge_capacity:
            # 找到第一个显著的容量变化
            charge_threshold = 0.001  # 1mAh
            discharge_threshold = 0.001  # 1mAh
            
            first_charge_idx = None
            first_discharge_idx = None
            
            for idx, (charge, discharge) in enumerate(zip(charge_capacity, discharge_capacity)):
                if first_charge_idx is None and charge > charge_threshold:
                    first_charge_idx = idx
                if first_discharge_idx is None and discharge > discharge_threshold:
                    first_discharge_idx = idx
                if first_charge_idx is not None and first_discharge_idx is not None:
                    break
            
            if first_charge_idx is not None and first_discharge_idx is not None:
                if first_charge_idx < first_discharge_idx:
                    charge_starts_first = True
                    capacity_direction = "充电先开始"
                    print(f"容量变化判断: {capacity_direction} (充电在索引{first_charge_idx}, 放电在索引{first_discharge_idx})")
                else:
                    discharge_starts_first = True
                    capacity_direction = "放电先开始"
                    print(f"容量变化判断: {capacity_direction} (放电在索引{first_discharge_idx}, 充电在索引{first_charge_idx})")
            elif first_charge_idx is not None:
                charge_starts_first = True
                capacity_direction = "仅检测到充电"
                print(f"容量变化判断: {capacity_direction} (充电在索引{first_charge_idx})")
            elif first_discharge_idx is not None:
                discharge_starts_first = True
                capacity_direction = "仅检测到放电"
                print(f"容量变化判断: {capacity_direction} (放电在索引{first_discharge_idx})")
            else:
                capacity_direction = "未检测到显著容量变化"
                print(f"容量变化判断: {capacity_direction}")
        
        # 方法3: 检查电压变化趋势
        voltage_data = first_cycle.get('voltage_in_V', [])
        if voltage_data and len(voltage_data) > 10:
            # 计算前10个数据点的电压变化趋势
            voltage_trend = np.polyfit(range(10), voltage_data[:10], 1)[0]
            if voltage_trend > 0.001:
                voltage_direction = "电压上升 (可能充电)"
            elif voltage_trend < -0.001:
                voltage_direction = "电压下降 (可能放电)"
            else:
                voltage_direction = "电压基本稳定"
            print(f"电压趋势判断: {voltage_direction}, 斜率: {voltage_trend:.6f}V/步")
        
        # 综合判断
        if charge_starts_first:
            print("综合判断: 先充电")
            charge_first_count += 1
        elif discharge_starts_first:
            print("综合判断: 先放电")
            discharge_first_count += 1
        else:
            print("综合判断: 无法确定")
            unclear_count += 1
    
    print(f"\n=== 统计结果 ===")
    print(f"检查文件数: {len(sample_files)}")
    print(f"先充电的文件: {charge_first_count}")
    print(f"先放电的文件: {discharge_first_count}")
    print(f"无法确定的文件: {unclear_count}")
    
    if charge_first_count > discharge_first_count:
        print("结论: MATR数据集主要是先充电")
    elif discharge_first_count > charge_first_count:
        print("结论: MATR数据集主要是先放电")
    else:
        print("结论: 充放电顺序不一致或无法确定")

if __name__ == "__main__":
    check_charge_discharge_order()