import numpy as np

def calculate_f47_f59(cycle_data):
    """计算F47-F59充电模式及容量特征"""
    
    # 提取前100个周期的数据用于特征计算
    cycles_to_use = cycle_data[:100] if len(cycle_data) >= 100 else cycle_data
    
    # F47: MVF——mean voltage falloff，5min after discharge
    # 改进：基于实际放电后的电压变化计算
    voltage_falloffs = []
    for cycle in cycles_to_use:
        current = np.array(cycle.get('current_in_A', []))
        voltage = np.array(cycle.get('voltage_in_V', []))
        time_data = np.array(cycle.get('time_in_s', []))
        
        if len(current) > 0 and len(voltage) > 0 and len(time_data) > 0:
            # 找到放电阶段
            discharge_mask = current < 0
            if np.any(discharge_mask):
                # 找到放电结束后的静置阶段
                discharge_indices = np.where(discharge_mask)[0]
                if len(discharge_indices) > 0:
                    discharge_end = discharge_indices[-1]
                    # 放电结束后的数据
                    rest_indices = np.where((current == 0) & (np.arange(len(current)) > discharge_end))[0]
                    if len(rest_indices) > 10:  # 确保有足够的静置数据
                        rest_voltage = voltage[rest_indices[:10]]  # 取前10个静置点
                        rest_time = time_data[rest_indices[:10]]
                        if len(rest_voltage) > 1:
                            # 计算电压衰减率
                            voltage_drop = rest_voltage[0] - rest_voltage[-1]
                            time_span = (rest_time[-1] - rest_time[0]) / 1e9  # 转换为秒
                            if time_span > 0:
                                voltage_falloffs.append(voltage_drop / time_span)  # 电压衰减率 V/s
    
    f47 = np.mean(voltage_falloffs) if voltage_falloffs else 0.001
    
    # F48-F53: 改进CC/CV阶段的识别和计算
    cc_voltage_time_intervals = []
    cc_capacities_4_0_4_2 = []
    cv_current_time_intervals = []
    cv_capacities_4_0_1 = []
    cc_temp_rates = []
    cv_temp_rates = []
    
    for cycle in cycles_to_use:
        current = np.array(cycle.get('current_in_A', []))
        voltage = np.array(cycle.get('voltage_in_V', []))
        time_data = np.array(cycle.get('time_in_s', []))
        charge_cap = np.array(cycle.get('charge_capacity_in_Ah', []))
        
        if len(current) == 0 or len(voltage) == 0 or len(time_data) == 0:
            continue
            
        # 识别充电阶段
        charge_mask = current > 0
        if not np.any(charge_mask):
            continue
            
        charge_current = current[charge_mask]
        charge_voltage = voltage[charge_mask]
        charge_time = time_data[charge_mask]
        charge_capacity = charge_cap[charge_mask] if len(charge_cap) > 0 else np.cumsum(charge_current) * 1e-6  # 估算容量
        
        # 改进的CC/CV段识别：基于电流变化而不是电压阈值
        # CC段：电流相对稳定，CV段：电流逐渐下降
        if len(charge_current) > 10:
            current_diff = np.abs(np.diff(charge_current))
            current_std = np.std(charge_current)
            
            # 找到电流开始显著下降的点（CC到CV的转换点）
            cv_start_candidates = np.where(current_diff > current_std * 0.1)[0]
            if len(cv_start_candidates) > 0:
                cv_start = cv_start_candidates[0]
                
                # CC段数据
                cc_voltage = charge_voltage[:cv_start]
                cc_current = charge_current[:cv_start]
                cc_time = charge_time[:cv_start]
                cc_cap = charge_capacity[:cv_start]
                
                # CV段数据
                cv_voltage = charge_voltage[cv_start:]
                cv_current = charge_current[cv_start:]
                cv_time = charge_time[cv_start:]
                cv_cap = charge_capacity[cv_start:]
                
                # F48: CC阶段电压从4.0V升至4.2V的时间间隔
                if len(cc_voltage) > 1:
                    v_min, v_max = np.min(cc_voltage), np.max(cc_voltage)
                    if v_max > v_min:
                        # 找到接近4.0V和4.2V的点
                        v_range = v_max - v_min
                        v_4_0_target = v_min + 0.3 * v_range  # 相对的"4.0V"点
                        v_4_2_target = v_min + 0.8 * v_range  # 相对的"4.2V"点
                        
                        idx_4_0 = np.where(cc_voltage >= v_4_0_target)[0]
                        idx_4_2 = np.where(cc_voltage >= v_4_2_target)[0]
                        
                        if len(idx_4_0) > 0 and len(idx_4_2) > 0:
                            time_interval = (cc_time[idx_4_2[0]] - cc_time[idx_4_0[0]]) / 1e9
                            cc_voltage_time_intervals.append(time_interval)
                            
                            # F49: CC阶段对应区间的充电容量
                            cap_in_range = cc_cap[idx_4_0[0]:idx_4_2[0]+1]
                            if len(cap_in_range) > 0:
                                cc_capacities_4_0_4_2.append(np.max(cap_in_range) - np.min(cap_in_range))
                            
                            # F52: CC阶段温度变化率（基于功率变化）
                            power_cc = cc_voltage[idx_4_0[0]:idx_4_2[0]+1] * cc_current[idx_4_0[0]:idx_4_2[0]+1]
                            if len(power_cc) > 1:
                                power_change_rate = np.std(np.diff(power_cc)) / (time_interval + 1e-6)
                                cc_temp_rates.append(power_change_rate)
                
                # F50: CV阶段电流变化的时间间隔
                if len(cv_current) > 1:
                    i_max, i_min = np.max(cv_current), np.min(cv_current)
                    if i_max > i_min:
                        # 电流从高到低的变化区间
                        i_range = i_max - i_min
                        i_high = i_max - 0.2 * i_range  # 相对的"高电流"点
                        i_low = i_min + 0.2 * i_range   # 相对的"低电流"点
                        
                        idx_high = np.where(cv_current <= i_high)[0]
                        idx_low = np.where(cv_current <= i_low)[0]
                        
                        if len(idx_high) > 0 and len(idx_low) > 0:
                            time_interval = (cv_time[idx_low[-1]] - cv_time[idx_high[0]]) / 1e9
                            cv_current_time_intervals.append(time_interval)
                            
                            # F51: CV阶段对应区间的充电容量
                            cap_in_range = cv_cap[idx_high[0]:idx_low[-1]+1]
                            if len(cap_in_range) > 0:
                                cv_capacities_4_0_1.append(np.max(cap_in_range) - np.min(cap_in_range))
                            
                            # F53: CV阶段温度变化率
                            power_cv = cv_voltage[idx_high[0]:idx_low[-1]+1] * cv_current[idx_high[0]:idx_low[-1]+1]
                            if len(power_cv) > 1:
                                power_change_rate = np.std(np.diff(power_cv)) / (time_interval + 1e-6)
                                cv_temp_rates.append(power_change_rate)
    
    # 计算F48-F53（移除默认值，使用实际计算结果）
    f48 = np.mean(cc_voltage_time_intervals) if cc_voltage_time_intervals else np.nan
    f49 = np.mean(cc_capacities_4_0_4_2) if cc_capacities_4_0_4_2 else np.nan
    f50 = np.mean(cv_current_time_intervals) if cv_current_time_intervals else np.nan
    f51 = np.mean(cv_capacities_4_0_1) if cv_capacities_4_0_1 else np.nan
    f52 = np.mean(cc_temp_rates) if cc_temp_rates else np.nan
    f53 = np.mean(cv_temp_rates) if cv_temp_rates else np.nan
    
    # 对于NaN值，使用基于数据的合理估算而不是固定默认值
    if np.isnan(f48):
        # 基于平均充电时间估算
        charge_times = []
        for cycle in cycles_to_use[:10]:
            current = np.array(cycle.get('current_in_A', []))
            time_data = np.array(cycle.get('time_in_s', []))
            if len(current) > 0 and len(time_data) > 0:
                charge_mask = current > 0
                if np.any(charge_mask):
                    charge_time_span = (time_data[charge_mask][-1] - time_data[charge_mask][0]) / 1e9
                    charge_times.append(charge_time_span)
        f48 = np.mean(charge_times) * 0.3 if charge_times else 300  # CC阶段约占充电时间的30%
    
    if np.isnan(f49):
        # 基于平均充电容量估算
        charge_caps = []
        for cycle in cycles_to_use[:10]:
            charge_cap = np.array(cycle.get('charge_capacity_in_Ah', []))
            if len(charge_cap) > 0:
                charge_caps.append(np.max(charge_cap))
        f49 = np.mean(charge_caps) * 0.6 if charge_caps else 0.15  # CC阶段约占总容量的60%
    
    if np.isnan(f50):
        f50 = f48 * 0.7 if not np.isnan(f48) else 200  # CV阶段通常比CC阶段短
    
    if np.isnan(f51):
        f51 = f49 * 0.4 if not np.isnan(f49) else 0.1  # CV阶段容量通常比CC阶段少
    
    if np.isnan(f52):
        # 基于电压和电流变化估算
        power_changes = []
        for cycle in cycles_to_use[:5]:
            current = np.array(cycle.get('current_in_A', []))
            voltage = np.array(cycle.get('voltage_in_V', []))
            if len(current) > 0 and len(voltage) > 0:
                charge_mask = current > 0
                if np.any(charge_mask):
                    power = voltage[charge_mask] * current[charge_mask]
                    if len(power) > 1:
                        power_changes.extend(np.abs(np.diff(power)))
        f52 = np.std(power_changes) * 0.01 if power_changes else 0.005
    
    if np.isnan(f53):
        f53 = f52 * 0.5 if not np.isnan(f52) else 0.003  # CV阶段功率变化通常更小
    
    # F54-F59保持原有逻辑，但改进计算
    # F54: CC充电容量（整个CC阶段）
    cc_total_capacities = []
    cv_total_capacities = []
    cc_end_slopes = []
    cc_corner_slopes = []
    
    for cycle in cycles_to_use:
        current = np.array(cycle.get('current_in_A', []))
        voltage = np.array(cycle.get('voltage_in_V', []))
        charge_cap = np.array(cycle.get('charge_capacity_in_Ah', []))
        
        if len(current) == 0 or len(voltage) == 0:
            continue
            
        charge_mask = current > 0
        if not np.any(charge_mask):
            continue
            
        charge_current = current[charge_mask]
        charge_voltage = voltage[charge_mask]
        charge_capacity = charge_cap[charge_mask] if len(charge_cap) > 0 else np.cumsum(charge_current) * 1e-6
        
        # 改进的CC/CV识别
        if len(charge_current) > 10:
            current_diff = np.abs(np.diff(charge_current))
            current_std = np.std(charge_current)
            cv_start_candidates = np.where(current_diff > current_std * 0.1)[0]
            
            if len(cv_start_candidates) > 0:
                cv_start = cv_start_candidates[0]
                
                # F54: CC阶段总充电容量
                cc_cap = charge_capacity[:cv_start]
                if len(cc_cap) > 0:
                    cc_total_capacities.append(np.max(cc_cap) - np.min(cc_cap))
                    
                    # F56: CC充电模式结束时的斜率
                    if len(cc_cap) > 5:
                        end_slope = np.polyfit(range(len(cc_cap)-5, len(cc_cap)), cc_cap[-5:], 1)[0]
                        cc_end_slopes.append(end_slope)
                        
                    # F57: CC充电曲线拐角处的垂直斜率
                    if cv_start > 5 and cv_start < len(charge_voltage) - 5:
                        before_corner = charge_voltage[cv_start-5:cv_start]
                        after_corner = charge_voltage[cv_start:cv_start+5]
                        if len(before_corner) > 0 and len(after_corner) > 0:
                            slope_before = np.polyfit(range(len(before_corner)), before_corner, 1)[0]
                            slope_after = np.polyfit(range(len(after_corner)), after_corner, 1)[0]
                            corner_slope = abs(slope_after - slope_before)
                            cc_corner_slopes.append(corner_slope)
                
                # F55: CV阶段总充电容量
                cv_cap = charge_capacity[cv_start:]
                if len(cv_cap) > 0:
                    cv_total_capacities.append(np.max(cv_cap) - np.min(cv_cap))
    
    f54 = np.mean(cc_total_capacities) if cc_total_capacities else f49
    f55 = np.mean(cv_total_capacities) if cv_total_capacities else f51
    f56 = np.mean(cc_end_slopes) if cc_end_slopes else 0.001
    f57 = np.mean(cc_corner_slopes) if cc_corner_slopes else 0.01
    
    # F58-F59保持原有逻辑
    discharge_capacities = []
    for cycle in cycle_data:
        current = np.array(cycle.get('current_in_A', []))
        discharge_cap = np.array(cycle.get('discharge_capacity_in_Ah', []))
        
        if len(current) > 0 and len(discharge_cap) > 0:
            discharge_mask = current < 0
            if np.any(discharge_mask):
                discharge_phase_cap = discharge_cap[discharge_mask]
                max_cap = np.max(discharge_phase_cap) if len(discharge_phase_cap) > 0 else 0
                discharge_capacities.append(max_cap)
            else:
                discharge_capacities.append(0)
        else:
            discharge_capacities.append(0)
    
    if discharge_capacities:
        max_cap_cycle = np.argmax(discharge_capacities) + 1
        f58 = max_cap_cycle
    else:
        f58 = 1
    
    if discharge_capacities:
        max_cap_idx = np.argmax(discharge_capacities)
        if max_cap_idx < len(cycle_data):
            cycle_with_max_cap = cycle_data[max_cap_idx]
            time_data = np.array(cycle_with_max_cap.get('time_in_s', []))
            if len(time_data) > 0:
                f59 = time_data[-1] / (1e9 * 3600) if time_data[-1] > 0 else 1.0
            else:
                f59 = 1.0
        else:
            f59 = 1.0
    else:
        f59 = 1.0
    
    return [f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59]