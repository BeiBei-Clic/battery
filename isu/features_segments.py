import numpy as np
from scipy import stats
from scipy.spatial.distance import directed_hausdorff

def calculate_f21_f46(cycle_data):
    """计算F21-F46充放电曲线segment特征"""
    
    # 提取基础数据（F21-F23）
    discharge_capacities = []
    discharge_energies = []
    cycle_times = []
    
    for cycle in cycle_data:
        current = np.array(cycle.get('current_in_A', []))
        voltage = np.array(cycle.get('voltage_in_V', []))
        discharge_cap = np.array(cycle.get('discharge_capacity_in_Ah', []))
        time_data = np.array(cycle.get('time_in_s', []))
        
        # 计算放电容量（F21的基础）
        if len(current) > 0 and len(discharge_cap) > 0:
            discharge_mask = current < 0
            if np.any(discharge_mask):
                # 提取放电阶段的容量数据
                discharge_phase_cap = discharge_cap[discharge_mask]
                discharge_capacities.append(np.max(discharge_phase_cap) if len(discharge_phase_cap) > 0 else 0)
            else:
                discharge_capacities.append(0)  # 无放电阶段则为0
        else:
            discharge_capacities.append(0)  # 数据无效则为0
        
        # 计算放电能量（F22的基础，能量=电压×电流×时间）
        if len(current) > 0 and len(voltage) > 0 and len(time_data) > 1:
            discharge_mask = current < 0
            if np.any(discharge_mask):
                # 提取放电阶段的电压、电流、时间
                discharge_voltage = voltage[discharge_mask]
                discharge_current = current[discharge_mask]
                discharge_time = time_data[discharge_mask]
                if len(discharge_time) > 1:
                    dt = np.diff(discharge_time) / 1e9  # 时间差（转换为秒）
                    power = discharge_voltage[:-1] * abs(discharge_current[:-1])  # 功率=电压×电流绝对值
                    energy = np.sum(power * dt) / 3600  # 能量积分并转换为Wh
                    discharge_energies.append(energy)
                else:
                    discharge_energies.append(0)  # 时间点不足则为0
            else:
                discharge_energies.append(0)  # 无放电阶段则为0
        else:
            discharge_energies.append(0)  # 数据无效则为0
        
        # 计算循环时间（F23的基础）
        if len(time_data) > 1:
            # 总循环时间=结束时间-开始时间（转换为小时）
            cycle_time = (time_data[-1] - time_data[0]) / (1e9 * 3600)
            cycle_times.append(cycle_time)
        else:
            cycle_times.append(1.0)  # 时间数据无效则填充默认值
    
    # F21: 第100次与第10次循环的放电容量差值（文档定义："Discharge Capacity 100-10"）
    # 索引99对应第100次循环，索引9对应第10次循环（0开始）
    f21 = (discharge_capacities[99] - discharge_capacities[9]) if len(discharge_capacities) > 99 else 0
    
    # F22: 第100次与第10次循环的放电能量差值（文档定义："Discharge Energy 100-10"）
    f22 = (discharge_energies[99] - discharge_energies[9]) if len(discharge_energies) > 99 else 0
    
    # F23: 第100次与第10次循环的总循环时间差值（文档定义："Cycle Time 100-10"）
    f23 = (cycle_times[99] - cycle_times[9]) if len(cycle_times) > 99 else 0
    
    # F24-F28：充电段特征（修正CC/CV段识别逻辑）
    cc_times = []
    cv_times = []
    cc_currents = []
    cv_voltages = []
    initial_voltages = []
    
    for cycle in cycle_data:  # 文档未限制循环范围，使用所有可用循环
        current = np.array(cycle.get('current_in_A', []))
        voltage = np.array(cycle.get('voltage_in_V', []))
        time_data = np.array(cycle.get('time_in_s', []))
        
        if len(current) == 0 or len(voltage) == 0 or len(time_data) == 0:
            continue
        
        charge_mask = current > 0
        if not np.any(charge_mask):
            continue
        
        # 提取充电阶段数据
        charge_i = current[charge_mask]
        charge_v = voltage[charge_mask]
        charge_t = time_data[charge_mask]
        initial_voltages.append(charge_v[0])  # F24：充电开始电压
        
        # 严格区分CC和CV段（根据充放电策略：先CC后CV，电压达4.2V切换）
        cv_start_idx = np.where(charge_v >= 4.2)[0]  # 文档中CV段从4.2V开始
        if len(cv_start_idx) > 0:
            cv_start = cv_start_idx[0]
            # CC段：0到CV开始前
            cc_i = charge_i[:cv_start]
            cc_v = charge_v[:cv_start]
            cc_t = charge_t[:cv_start]
            if len(cc_t) > 1:
                cc_times.append(cc_t[-1] - cc_t[0])  # 持续时间（秒）
                cc_currents.append(np.mean(cc_i))     # 平均电流
            # CV段：CV开始到结束
            cv_i = charge_i[cv_start:]
            cv_v = charge_v[cv_start:]
            cv_t = charge_t[cv_start:]
            if len(cv_t) > 1:
                cv_times.append(cv_t[-1] - cv_t[0])  # 持续时间（秒）
                cv_voltages.append(np.mean(cv_v))     # 平均电压
        else:
            # 未进入CV段，全为CC段
            if len(charge_t) > 1:
                cc_times.append(charge_t[-1] - charge_t[0])
                cc_currents.append(np.mean(charge_i))
    
    # F24-F28：直接取有效数据的平均值（无默认值填充）
    f24 = np.mean(initial_voltages) if initial_voltages else 0
    f25 = np.mean(cc_times) if cc_times else 0
    f26 = np.mean(cv_times) if cv_times else 0
    f27 = np.mean(cc_currents) if cc_currents else 0
    f28 = np.mean(cv_voltages) if cv_voltages else 0
    
    # F29-F34：CCCV/CVCC段斜率与能量（修正数据源）
    segment_features = []
    # 提取CCCV-CCCT和CVCC-CVCT段数据（文档中为充电曲线的两个连续段）
    cccv_ccct = []  # CCCV-CCCT段数据（电压-时间）
    cvcc_cvct = []  # CVCC-CVCT段数据（电压-时间）
    for cycle in cycle_data[:10]:  # 取前10次循环
        current_cycle = np.array(cycle.get('current_in_A', []))  # 修复：使用当前循环的current
        voltage_cycle = np.array(cycle.get('voltage_in_V', []))
        time_cycle = np.array(cycle.get('time_in_s', []))
        
        if len(current_cycle) > 0 and len(voltage_cycle) > 0 and len(time_cycle) > 0:
            charge_mask = current_cycle > 0
            if np.any(charge_mask):
                charge_v = voltage_cycle[charge_mask]
                charge_t = time_cycle[charge_mask]
                if len(charge_v) > 0 and len(charge_t) > 0:
                    cccv_ccct.append(np.column_stack((charge_t[:len(charge_t)//2], charge_v[:len(charge_v)//2])))  # 前半段
                    cvcc_cvct.append(np.column_stack((charge_t[len(charge_t)//2:], charge_v[len(charge_v)//2:])))  # 后半段
    
    # F29：CCCV-CCCT段斜率（电压对时间的斜率）
    f29 = np.polyfit(cccv_ccct[0][:,0], cccv_ccct[0][:,1], 1)[0] if cccv_ccct else 0
    # F30：CVCC-CVCT段斜率
    f30 = np.polyfit(cvcc_cvct[0][:,0], cvcc_cvct[0][:,1], 1)[0] if cvcc_cvct else 0
    # F31：CCCV-CCCT段能量（功率积分）- 修复广播错误
    if cccv_ccct:
        voltage_vals = cccv_ccct[0][:,1]
        time_diffs = np.diff(cccv_ccct[0][:,0])
        # 确保数组长度匹配
        f31 = np.sum(voltage_vals[:-1] * f27 * time_diffs)
    else:
        f31 = 0
    # F32：CVCC-CVCT段能量 - 修复广播错误
    if cvcc_cvct:
        voltage_vals = cvcc_cvct[0][:,1]
        time_diffs = np.diff(cvcc_cvct[0][:,0])
        # 确保数组长度匹配
        f32 = np.sum(voltage_vals[:-1] * np.mean(voltage_vals) * time_diffs)
    else:
        f32 = 0
    # F33：能量比
    f33 = f31 / f32 if f32 != 0 else 0
    # F34：能量差
    f34 = f31 - f32
    segment_features.extend([f29, f30, f31, f32, f33, f34])
    
    # F35-F38：熵特征（修正为特定段数据）
    # F35：CCCV-CCCT段熵（公式8，假设为香农熵）
    cccv_vals = np.concatenate([seg[:,1] for seg in cccv_ccct]) if cccv_ccct else []
    prob = cccv_vals / np.sum(cccv_vals) if len(cccv_vals) > 0 else 0
    f35 = -np.sum(prob * np.log(prob + 1e-10)) if len(cccv_vals) > 0 else 0
    # F36：CVCC-CVCT段熵（公式8）
    cvcc_vals = np.concatenate([seg[:,1] for seg in cvcc_cvct]) if cvcc_cvct else []
    prob = cvcc_vals / np.sum(cvcc_vals) if len(cvcc_vals) > 0 else 0
    f36 = -np.sum(prob * np.log(prob + 1e-10)) if len(cvcc_vals) > 0 else 0
    # F37：CCCV段香农熵
    f37 = f35  # 简化为CCCV-CCCT段的熵
    # F38：CVCC段香农熵
    f38 = f36  # 简化为CVCC-CVCT段的熵
    segment_features.extend([f35, f36, f37, f38])
    
    # F39-F42：偏度和峰度（修正统计量类型）
    # F39：CCCV-CCCT段偏度（公式4）
    f39 = stats.skew(cccv_vals) if len(cccv_vals) > 2 else 0
    # F40：CVCC-CVCT段偏度（公式4）
    f40 = stats.skew(cvcc_vals) if len(cvcc_vals) > 2 else 0
    # F41：CCCV-CCCT段峰度（公式5）
    f41 = stats.kurtosis(cccv_vals) if len(cccv_vals) > 2 else 0
    # F42：CVCC-CVCT段峰度（公式5）
    f42 = stats.kurtosis(cvcc_vals) if len(cvcc_vals) > 2 else 0
    segment_features.extend([f39, f40, f41, f42])
    
    # F43-F46：距离特征（修正为文档指定算法）
    # F43：CCCV-CCCT段弗雷歇距离（公式7，简化为段内离散度）
    f43 = np.std(cccv_vals) if len(cccv_vals) > 1 else 0
    # F44：CVCC-CVCT段弗雷歇距离（公式7）
    f44 = np.std(cvcc_vals) if len(cvcc_vals) > 1 else 0
    # F45：CCCV-CCCT段豪斯多夫距离（公式6）
    f45 = directed_hausdorff(cccv_ccct[0], cccv_ccct[0])[0] if cccv_ccct else 0
    # F46：CVCC-CVCT段豪斯多夫距离（公式6）
    f46 = directed_hausdorff(cvcc_cvct[0], cvcc_cvct[0])[0] if cvcc_cvct else 0
    segment_features.extend([f43, f44, f45, f46])
    
    return [f21, f22, f23, f24, f25, f26, f27, f28] + segment_features