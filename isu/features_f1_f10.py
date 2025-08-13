import numpy as np
import math
from scipy.interpolate import interp1d
from scipy import stats

def extract_qv_curves_isu(cycle_data):
    """提取ISU数据每个周期的Q-V曲线（放电阶段的容量-电压关系）"""
    qv_curves = []
    
    for cycle in cycle_data:
        voltage = np.array(cycle.get('voltage_in_V', []))
        discharge_capacity = np.array(cycle.get('discharge_capacity_in_Ah', []))
        current = np.array(cycle.get('current_in_A', []))
        
        if len(voltage) > 0 and len(current) > 0 and len(discharge_capacity) > 0:
            discharge_mask = current < 0
            if np.any(discharge_mask):
                discharge_voltage = voltage[discharge_mask]
                discharge_cap = discharge_capacity[discharge_mask]
                
                if len(discharge_voltage) > 1 and len(discharge_cap) > 1:
                    valid_mask = (discharge_cap > 0) & np.isfinite(discharge_voltage) & np.isfinite(discharge_cap)
                    if np.sum(valid_mask) > 1:
                        valid_voltage = discharge_voltage[valid_mask]
                        valid_capacity = discharge_cap[valid_mask]
                        
                        sort_idx = np.argsort(-valid_voltage)
                        sorted_voltage = valid_voltage[sort_idx]
                        sorted_capacity = valid_capacity[sort_idx]
                        
                        qv_curves.append((sorted_voltage, sorted_capacity))
                        continue
        
        qv_curves.append((np.array([]), np.array([])))
    
    return qv_curves

def calculate_delta_q_isu(qv_curves, cycle_10=10, cycle_100=100):
    """计算ΔQ₁₀₀₋₁₀(V)：第100次与第10次循环的放电容量差值"""
    if len(qv_curves) < max(cycle_10, cycle_100):
        return None
    
    v10, q10 = qv_curves[cycle_10-1]
    v100, q100 = qv_curves[cycle_100-1]
    
    if len(v10) < 2 or len(v100) < 2:
        return None
    
    min_v = max(np.min(v10), np.min(v100))
    max_v = min(np.max(v10), np.max(v100))
    
    if min_v >= max_v:
        return None
    
    common_v = np.linspace(max_v, min_v, 100)
    
    f10 = interp1d(v10, q10, kind='linear', bounds_error=False, fill_value="extrapolate")
    f100 = interp1d(v100, q100, kind='linear', bounds_error=False, fill_value="extrapolate")
    
    q10_interp = f10(common_v)
    q100_interp = f100(common_v)
    
    delta_q = q100_interp - q10_interp
    
    return delta_q, common_v

def calculate_f1_f10_isu(battery_data):
    """计算ISU数据的F1-F10特征，严格按照指导文件定义"""
    
    cycle_data = battery_data.get('cycle_data', [])
    
    # 提取Q-V曲线
    qv_curves = extract_qv_curves_isu(cycle_data)
    
    # 计算ΔQ₁₀₀₋₁₀(V)
    delta_q_result = calculate_delta_q_isu(qv_curves)
    
    if delta_q_result is None:
        return [0] * 10
    delta_q, common_v = delta_q_result
    # F1: ΔQ₁₀₀₋₁₀的最小值
    f1 = np.min(delta_q)
    # F2: ΔQ₁₀₀₋₁₀的平均值
    f2 = np.mean(delta_q)
    # F3: ΔQ₁₀₀₋₁₀的方差
    f3 = np.var(delta_q)
    # F4: ΔQ₁₀₀₋₁₀的偏度
    f4 = stats.skew(delta_q)
    # F5: ΔQ₁₀₀₋₁₀的峰度
    f5 = stats.kurtosis(delta_q)
    # F6: ΔQ₁₀₀₋₁₀在2V处的值
    if len(common_v) > 0:
        idx_2v = np.argmin(np.abs(common_v - 2.0))
        f6 = delta_q[idx_2v]
    else:
        f6 = 0
    # F7-F8: 第2-100次循环的容量衰减曲线线性拟合的斜率和截距
    discharge_caps = []
    for i in range(1, min(100, len(cycle_data))):
        current = np.array(cycle_data[i].get('current_in_A', []))
        cap_data = np.array(cycle_data[i].get('discharge_capacity_in_Ah', []))
        
        if len(current) > 0 and len(cap_data) > 0:
            discharge_mask = current < 0
            if np.any(discharge_mask):
                discharge_phase_cap = cap_data[discharge_mask]
                if len(discharge_phase_cap) > 0:
                    discharge_caps.append(np.max(discharge_phase_cap))
                else:
                    discharge_caps.append(0)
            else:
                discharge_caps.append(0)
        else:
            discharge_caps.append(0)
    
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