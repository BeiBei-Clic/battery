import numpy as np
import math
from scipy.interpolate import interp1d
from scipy import stats

def extract_qv_curves(cycle_data):
    """提取每个周期的Q-V曲线（放电阶段的容量-电压关系）"""
    qv_curves = []  # 存储每个循环的(电压数组, 容量数组)
    
    for cycle in cycle_data:
        # 从循环数据中提取电压、放电容量、电流数据
        voltage = np.array(cycle.get('voltage_in_V', []))  # 电压数据（单位：V）
        discharge_capacity = np.array(cycle.get('discharge_capacity_in_Ah', []))  # 放电容量（单位：Ah）
        current = np.array(cycle.get('current_in_A', []))  # 电流数据（单位：A）
        
        # 检查数据有效性：电压、电流、容量数组均非空
        if len(voltage) > 0 and len(current) > 0 and len(discharge_capacity) > 0:
            # 筛选放电阶段：电流为负值（放电时电流方向与充电相反）
            discharge_mask = current < 0
            if np.any(discharge_mask):  # 存在放电阶段数据
                # 提取放电阶段的电压和容量
                discharge_voltage = voltage[discharge_mask]
                discharge_cap = discharge_capacity[discharge_mask]
                
                # 确保放电阶段数据有效（至少2个点，避免插值失败）
                if len(discharge_voltage) > 1 and len(discharge_cap) > 1:
                    # 筛选有效数据：容量>0，且电压、容量无异常值（非无穷/NaN）
                    valid_mask = (discharge_cap > 0) & np.isfinite(discharge_voltage) & np.isfinite(discharge_cap)
                    if np.sum(valid_mask) > 1:  # 有效数据点数量足够
                        valid_voltage = discharge_voltage[valid_mask]
                        valid_capacity = discharge_cap[valid_mask]
                        
                        # 按电压降序排序：放电过程电压从高到低，确保Q-V曲线单调
                        sort_idx = np.argsort(-valid_voltage)  # 降序索引
                        sorted_voltage = valid_voltage[sort_idx]
                        sorted_capacity = valid_capacity[sort_idx]
                        
                        # 存储排序后的Q-V曲线
                        qv_curves.append((sorted_voltage, sorted_capacity))
                        continue  # 进入下一个循环
        
        # 若数据无效，存储空数组
        qv_curves.append((np.array([]), np.array([])))
    
    return qv_curves

def calculate_delta_q(qv_curves, cycle_10=10, cycle_100=100):
    """计算ΔQ₁₀₀₋₁₀(V)：第100次与第10次循环的放电容量差值（统一电压维度后）"""
    # 检查循环数据是否足够（至少包含第10和第100次循环）
    if len(qv_curves) < max(cycle_10, cycle_100):
        return None  # 数据不足，返回空
    
    # 提取第10次和第100次循环的Q-V曲线（索引从0开始，故-1）
    v10, q10 = qv_curves[cycle_10-1]  # 第10圈的电压和容量
    v100, q100 = qv_curves[cycle_100-1]  # 第100圈的电压和容量
    
    # 检查曲线有效性：至少2个点才能进行插值
    if len(v10) < 2 or len(v100) < 2:
        return None  # 数据点不足，返回空
    

    min_v = max(np.min(v10), np.min(v100))  # 避免超出任意曲线的电压下限
    max_v = min(np.max(v10), np.max(v100))  # 避免超出任意曲线的电压上限
    
    # 检查电压范围有效性：下限需小于上限
    if min_v >= max_v:
        return None  # 电压范围无效，返回空
    
    # 生成100个均匀分布的电压点（从高到低，对应放电过程）
    common_v = np.linspace(max_v, min_v, 100)  # 文档中F1标注为100个插值点
    
    # 线性插值：将第10和第100次循环的容量映射到统一电压点
    # bounds_error=False：超出原始数据范围时不报错；fill_value="extrapolate"：超出范围时外推
    f10 = interp1d(v10, q10, kind='linear', bounds_error=False, fill_value="extrapolate")
    f100 = interp1d(v100, q100, kind='linear', bounds_error=False, fill_value="extrapolate")
    
    # 计算插值后的容量
    q10_interp = f10(common_v)  # 第10次循环在统一电压下的容量
    q100_interp = f100(common_v)  # 第100次循环在统一电压下的容量
    
    # 计算ΔQ(V) = Q₁₀₀(V) - Q₁₀(V)
    delta_q = q100_interp - q10_interp
    
    return delta_q, common_v  # 返回容量差值和对应的电压点

def calculate_f1_f10(cycle_data):
    """计算F1-F10特征（基于ΔQ₁₀₀₋₁₀和容量衰减曲线）"""
    # 步骤1：提取所有循环的Q-V曲线
    qv_curves = extract_qv_curves(cycle_data)
    
    # 步骤2：计算ΔQ₁₀₀₋₁₀(V)
    delta_q_result = calculate_delta_q(qv_curves)
    
    # 若ΔQ计算失败，返回全0
    if delta_q_result is None:
        return [0] * 10
    
    delta_q, common_v = delta_q_result  # 解包容量差值和电压点
    
    # F1: 最小绝对值的对数
    min_abs_delta_q = np.min(np.abs(delta_q))
    f1 = math.log10(min_abs_delta_q) if min_abs_delta_q > 0 else -10
    
    # F2：ΔQ₁₀₀₋₁₀的平均值（文档中F2定义为"Mean"）
    f2 = np.mean(delta_q)  # 计算正确
    
    # F3：ΔQ₁₀₀₋₁₀的方差（文档中F3定义为"Variance"）
    f3 = np.var(delta_q)  # 计算正确（默认除以n，若需除以n-1需加ddof=1）
    
    # F4：ΔQ₁₀₀₋₁₀的偏度（文档中F4定义为"Skewness"）
    f4 = stats.skew(delta_q)  # 计算正确（scipy默认无偏估计）
    
    # F5：ΔQ₁₀₀₋₁₀的峰度（文档中F5定义为"Kurtosis"）
    # 原代码中计算的是"峰度绝对值的对数"，与文档定义不符，此处需修正
    f5 = stats.kurtosis(delta_q)  # 修正为直接取峰度（scipy默认 excess kurtosis）
    
    # F6：ΔQ₁₀₀₋₁₀在2V处的值（文档中F6定义为"Value at 2V"）
    if len(common_v) > 0:
        # 找到最接近2V的电压点的索引
        idx_2v = np.argmin(np.abs(common_v - 2.0))
        f6 = delta_q[idx_2v]  # 提取对应ΔQ值
    else:
        f6 = 0  # 无数据时取0，计算逻辑正确
    
    # F7-F8：第2-100次循环的容量衰减曲线线性拟合的斜率和截距
    # （文档中F7定义为"Slope of linear fit, cycles 2 to 100"，F8为截距）
    discharge_caps = []  # 存储第2-100次循环的放电容量
    for i in range(1, min(100, len(cycle_data))):  # i从1开始（对应第2次循环，索引0为第1次）
        # 提取当前循环的电流和容量数据
        current = np.array(cycle_data[i].get('current_in_A', []))
        cap_data = np.array(cycle_data[i].get('discharge_capacity_in_Ah', []))
        
        # 检查数据有效性
        if len(current) > 0 and len(cap_data) > 0:
            # 筛选放电阶段（电流<0）
            discharge_mask = current < 0
            if np.any(discharge_mask):
                # 提取放电阶段的容量并取最大值（单次循环的总放电容量）
                discharge_phase_cap = cap_data[discharge_mask]
                if len(discharge_phase_cap) > 0:
                    discharge_caps.append(np.max(discharge_phase_cap))
                else:
                    discharge_caps.append(0)  # 无放电数据时取0
            else:
                discharge_caps.append(0)  # 无放电阶段时取0
        else:
            discharge_caps.append(0)  # 数据为空时取0
    
    # 线性拟合：循环次数为x，放电容量为y
    if len(discharge_caps) > 1:  # 至少2个点才能拟合
        cycles = np.arange(2, 2 + len(discharge_caps))  # 循环次数：2,3,...,len+1
        slope_2_100, intercept_2_100 = np.polyfit(cycles, discharge_caps, 1)  # 一阶拟合
        f7 = slope_2_100  # F7为斜率
        f8 = intercept_2_100  # F8为截距
    else:
        f7 = f8 = 0  # 数据不足时取0，逻辑正确
    
    # F9-F10：第91-100次循环的容量衰减曲线线性拟合的斜率和截距
    # （文档中F9定义为"Slope of linear fit, cycles 91 to 100"，F10为截距）
    if len(discharge_caps) >= 90:  # 第2-100次共99个循环，索引0-98，第91-100次对应索引89-98
        cycles_91_100 = np.arange(91, 101)  # 循环次数：91-100
        caps_91_100 = discharge_caps[89:99]  # 提取对应容量（索引89-98，共10个值）
        if len(caps_91_100) > 1:  # 至少2个点才能拟合
            slope_91_100, intercept_91_100 = np.polyfit(cycles_91_100, caps_91_100, 1)
            f9 = slope_91_100  # F9为斜率
            f10 = intercept_91_100  # F10为截距
        else:
            f9 = f10 = 0  # 数据不足时取0
    else:
        f9 = f10 = 0  # 总循环数不足时取0，逻辑正确
    
    return [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]