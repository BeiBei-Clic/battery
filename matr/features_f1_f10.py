import numpy as np
import math
from scipy.interpolate import interp1d
from scipy import stats

def extract_qv_curves_matr(cycle_data):
    """从MATR数据中提取每个周期的Q-V曲线（放电阶段的容量-电压关系）"""
    qv_curves = []
    
    for cycle in cycle_data:
        voltage = np.array(cycle.get('voltage_in_V', []))
        discharge_capacity = np.array(cycle.get('discharge_capacity_in_Ah', []))
        current = np.array(cycle.get('current_in_A', []))
        
        if len(voltage) > 0 and len(current) > 0 and len(discharge_capacity) > 0:
            # 筛选放电阶段：电流为负值
            discharge_mask = current < 0
            if np.any(discharge_mask):
                discharge_voltage = voltage[discharge_mask]
                discharge_cap = discharge_capacity[discharge_mask]
                
                if len(discharge_voltage) > 1 and len(discharge_cap) > 1:
                    # 修改有效数据筛选条件：降低容量阈值，因为MATR数据中容量值很小
                    valid_mask = (discharge_cap > 1e-8) & np.isfinite(discharge_voltage) & np.isfinite(discharge_cap)
                    if np.sum(valid_mask) > 10:  # 需要更多有效点
                        valid_voltage = discharge_voltage[valid_mask]
                        valid_capacity = discharge_cap[valid_mask]
                        
                        # 按电压降序排序：放电过程电压从高到低
                        sort_idx = np.argsort(-valid_voltage)
                        sorted_voltage = valid_voltage[sort_idx]
                        sorted_capacity = valid_capacity[sort_idx]
                        
                        qv_curves.append((sorted_voltage, sorted_capacity))
                        continue
        
        # 若数据无效，存储空数组
        qv_curves.append((np.array([]), np.array([])))
    
    return qv_curves

def calculate_delta_q_matr(qv_curves, cycle_10=10, cycle_100=100):
    """计算ΔQ₁₀₀₋₁₀(V)：第100次与第10次循环的放电容量差值"""
    # 动态调整周期选择，如果数据不足100个周期
    actual_cycle_100 = min(cycle_100, len(qv_curves))
    if actual_cycle_100 < cycle_10:
        return None
        
    if len(qv_curves) < max(cycle_10, actual_cycle_100):
        return None
    
    v10, q10 = qv_curves[cycle_10-1]
    v100, q100 = qv_curves[actual_cycle_100-1]
    
    if len(v10) < 10 or len(v100) < 10:  # 需要更多数据点
        return None
    
    # 确定公共电压范围
    min_v = max(np.min(v10), np.min(v100))
    max_v = min(np.max(v10), np.max(v100))
    
    if min_v >= max_v or (max_v - min_v) < 0.1:  # 电压范围太小
        return None
    
    # 生成100个均匀分布的电压点
    common_v = np.linspace(max_v, min_v, 100)
    
    # 线性插值
    f10 = interp1d(v10, q10, kind='linear', bounds_error=False, fill_value="extrapolate")
    f100 = interp1d(v100, q100, kind='linear', bounds_error=False, fill_value="extrapolate")
    
    q10_interp = f10(common_v)
    q100_interp = f100(common_v)
    
    # 检查插值结果是否有效
    if np.any(np.isnan(q10_interp)) or np.any(np.isnan(q100_interp)):
        return None
    
    # 计算ΔQ(V) = Q₁₀₀(V) - Q₁₀(V)
    delta_q = q100_interp - q10_interp
    
    return delta_q, common_v

def calculate_f1_f10_matr(battery_data):
    """计算MATR数据的F1-F10特征，使用Qdlin字段"""
    
    cycle_data = battery_data.get('cycle_data', [])
    
    # 获取Qdlin数据的函数
    def get_qdlin(cycle_idx):
        if cycle_idx < len(cycle_data):
            cycle = cycle_data[cycle_idx]
            if isinstance(cycle, dict) and 'Qdlin' in cycle:
                return np.array(cycle['Qdlin'])
        return None
    
    # F1-F6: 基于Qdlin计算ΔQ₁₀₀₋₁₀
    if len(cycle_data) >= 100:
        cycle_10_idx = 9   # 第10次循环的索引
        cycle_100_idx = 99 # 第100次循环的索引
        
        Qdlin_10 = get_qdlin(cycle_10_idx)
        Qdlin_100 = get_qdlin(cycle_100_idx)
        
        if Qdlin_10 is not None and Qdlin_100 is not None and len(Qdlin_10) == len(Qdlin_100):
            # 计算ΔQ(V) = Q₁₀₀(V) - Q₁₀(V)
            delta_q = Qdlin_100 - Qdlin_10
            
            # F1: 最小绝对值的对数
            min_abs_delta_q = np.min(np.abs(delta_q))
            f1 = math.log10(min_abs_delta_q) if min_abs_delta_q > 0 else -10
            
            # F2: ΔQ₁₀₀₋₁₀的平均值
            f2 = np.mean(delta_q)
            
            # F3: ΔQ₁₀₀₋₁₀的方差
            f3 = np.var(delta_q)
            
            # F4: ΔQ₁₀₀₋₁₀的偏度
            f4 = stats.skew(delta_q)
            
            # F5: ΔQ₁₀₀₋₁₀的峰度
            f5 = stats.kurtosis(delta_q)
            
            # F6: ΔQ₁₀₀₋₁₀在2V处的值
            # Qdlin通常是在固定电压点的数据，假设是从3.5V到2V的1000个点
            # 取最后一个点作为2V处的值
            f6 = delta_q[-1] if len(delta_q) > 0 else 0
            
        else:
            # 如果没有Qdlin数据，使用Q-V曲线计算
            qv_curves = extract_qv_curves_matr(cycle_data)
            delta_q_result = calculate_delta_q_matr(qv_curves, 10, 100)
            
            if delta_q_result is not None:
                delta_q, common_v = delta_q_result
                
                min_abs_delta_q = np.min(np.abs(delta_q))
                f1 = math.log10(min_abs_delta_q) if min_abs_delta_q > 0 else -10
                f2 = np.mean(delta_q)
                f3 = np.var(delta_q)
                f4 = stats.skew(delta_q)
                f5 = stats.kurtosis(delta_q)
                f6 = delta_q[-1] if len(delta_q) > 0 else 0
            else:
                f1 = f2 = f3 = f4 = f5 = f6 = 0
    else:
        # 如果周期数不足100，尝试使用最大可用周期
        max_cycle = len(cycle_data)
        if max_cycle >= 10:
            cycle_10_idx = 9
            cycle_max_idx = max_cycle - 1
            
            Qdlin_10 = get_qdlin(cycle_10_idx)
            Qdlin_max = get_qdlin(cycle_max_idx)
            
            if Qdlin_10 is not None and Qdlin_max is not None and len(Qdlin_10) == len(Qdlin_max):
                delta_q = Qdlin_max - Qdlin_10
                
                min_abs_delta_q = np.min(np.abs(delta_q))
                f1 = math.log10(min_abs_delta_q) if min_abs_delta_q > 0 else -10
                f2 = np.mean(delta_q)
                f3 = np.var(delta_q)
                f4 = stats.skew(delta_q)
                f5 = stats.kurtosis(delta_q)
                f6 = delta_q[-1] if len(delta_q) > 0 else 0
            else:
                # 使用Q-V曲线计算
                qv_curves = extract_qv_curves_matr(cycle_data)
                delta_q_result = calculate_delta_q_matr(qv_curves, 10, max_cycle)
                
                if delta_q_result is not None:
                    delta_q, common_v = delta_q_result
                    
                    min_abs_delta_q = np.min(np.abs(delta_q))
                    f1 = math.log10(min_abs_delta_q) if min_abs_delta_q > 0 else -10
                    f2 = np.mean(delta_q)
                    f3 = np.var(delta_q)
                    f4 = stats.skew(delta_q)
                    f5 = stats.kurtosis(delta_q)
                    f6 = delta_q[-1] if len(delta_q) > 0 else 0
                else:
                    f1 = f2 = f3 = f4 = f5 = f6 = 0
        else:
            f1 = f2 = f3 = f4 = f5 = f6 = 0
    
    # F7-F8: 第2-100次循环的容量衰减曲线线性拟合的斜率和截距
    discharge_caps = []
    for i in range(1, min(100, len(cycle_data))):
        current = np.array(cycle_data[i].get('current_in_A', []))
        cap_data = np.array(cycle_data[i].get('discharge_capacity_in_Ah', []))
        
        if len(current) > 0 and len(cap_data) > 0:
            discharge_mask = current < 0
            if np.any(discharge_mask):
                discharge_phase_cap = cap_data[discharge_mask]
                # 取放电阶段的最大容量值
                valid_caps = discharge_phase_cap[discharge_phase_cap > 0]
                if len(valid_caps) > 0:
                    discharge_caps.append(np.max(valid_caps))
                else:
                    discharge_caps.append(0)
            else:
                discharge_caps.append(0)
        else:
            discharge_caps.append(0)
    
    # 线性拟合
    if len(discharge_caps) > 1:
        cycles = np.arange(2, 2 + len(discharge_caps))
        slope_2_100, intercept_2_100 = np.polyfit(cycles, discharge_caps, 1)
        f7 = slope_2_100
        f8 = intercept_2_100
    else:
        f7 = f8 = 0
    
    # F9-F10: 第91-100次循环的容量衰减曲线线性拟合的斜率和截距
    if len(discharge_caps) >= 90:
        cycles_91_100 = np.arange(91, 101)
        caps_91_100 = discharge_caps[89:99]
        if len(caps_91_100) > 1:
            slope_91_100, intercept_91_100 = np.polyfit(cycles_91_100, caps_91_100, 1)
            f9 = slope_91_100
            f10 = intercept_91_100
        else:
            f9 = f10 = 0
    else:
        f9 = f10 = 0
    
    return [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]