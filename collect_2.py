import pickle  # 用于读取pickle格式的电池数据文件
import numpy as np  # 用于数值计算和数组操作
import math  # 用于数学运算

from utils import *  # 导入自定义工具函数（包含熵计算、曲线长度等方法）
from scipy.spatial.distance import directed_hausdorff  # 用于计算有向豪斯多夫距离
from collect_base import BaseDataset  # 导入基础数据集类


# 针对单颗电池的数据集处理类（提取F21-F46特征）
class DatasetTwo(BaseDataset):
    def __init__(self, battery, battery_index):
        # 调用父类构造方法初始化电池数据和索引
        super().__init__(battery, battery_index)


    def extract(self):
        """提取电池的26个特征（F21-F46），返回特征字典"""
        
        # F21：第100周期与第10周期的累积放电容量差值（单位：Ah）
        Cumulated_DQ_10_100 = self.battery['summary']['QD'][99] - self.battery['summary']['QD'][9]

        # F22：第100周期与第10周期的累积放电能量差值（单位：Wh）
        Cumulated_DC_Energy_10_100 = self.get_stage_energe(100, 'Discharge') - self.get_stage_energe(10, 'Discharge')

        # F23：第100周期与第2周期的循环时间差值（单位：s）
        cycle_time_100_2 = self.get_cycle_attr(100, 't')[-1] - self.get_cycle_attr(2, 't')[-1]

        # F24：第100周期与第10周期的充电起始端电压差值（单位：V）
        V_s_of_charge = self.get_cycle_attr(100, 'V')[0] - self.get_cycle_attr(10, 'V')[0]

        # F25：第100周期与第10周期的恒流（CC）充电时间差值（单位：s）
        cc_t_100_10 = self.get_charge_time(100, 'CC-0') - self.get_charge_time(10, 'CC-0')

        # F26：第100周期与第10周期的恒压（CV）充电时间差值（单位：s）
        cv_t_100_10 = self.get_charge_time(100, 'CV') - self.get_charge_time(10, 'CV')

        # F27：第100周期与第10周期的CC阶段平均电压差值（单位：V）
        V_mean_cc_100_10 = self.get_mean_stage(100, 'CC', 'V') - self.get_mean_stage(10, 'CC', 'V')

        # F28：第100周期与第10周期的CV阶段平均电流差值（单位：A）
        I_mean_cv_100_10 = self.get_mean_stage(100, 'CV', 'I') - self.get_mean_stage(10, 'CV', 'I')

        # F29：第100周期与第10周期的CC阶段电压-时间曲线斜率差值
        slope_v_t_100_10 = self.get_slope(100, 'CC', 'VT') - self.get_slope(10, 'CC', 'VT')

        # F30：第100周期与第10周期的CV阶段电流-时间曲线斜率差值
        slope_c_t_100_10 = self.get_slope(100, 'CV', 'CT') - self.get_slope(10, 'CV', 'CT')

        # F31：第100周期与第10周期的CC阶段能量差值（单位：Wh）
        energy_cc = self.get_stage_energe(100, 'CC') - self.get_stage_energe(10, 'CC')

        # F32：第100周期与第10周期的CV阶段能量差值（单位：Wh）
        energy_cv = self.get_stage_energe(100, 'CV') - self.get_stage_energe(10, 'CV')

        # F33：第100周期与第10周期的CC/CV能量比差值
        energy_ratio_100 = self.get_stage_energe(100, 'CC') / (self.get_stage_energe(100, 'CV') + 1e-6)  # 加小值避免除零
        energy_ratio_10 = self.get_stage_energe(10, 'CC') / (self.get_stage_energe(10, 'CV') + 1e-6)
        energy_ratio = energy_ratio_100 - energy_ratio_10

        # F34：第100周期与第10周期的CC-CV能量差的差值（单位：Wh）
        energy_diff_100 = self.get_stage_energe(100, 'CC-0') - self.get_stage_energe(100, 'CV')
        energy_diff_10 = self.get_stage_energe(10, 'CC-0') - self.get_stage_energe(10, 'CV')
        energy_diff = energy_diff_100 - energy_diff_10

        # F37：第100周期与第10周期的CC阶段电压的香农熵差值
        s_entropy_v_100_10 = self.get_shannon_entropy(100, 'CC-0', 'V') - self.get_shannon_entropy(10, 'CC-0', 'V')

        # F38：第100周期与第10周期的CV阶段电流的香农熵差值
        s_entropy_c_100_10 = self.get_shannon_entropy(100, 'CV', 'I') - self.get_shannon_entropy(10, 'CV', 'I')

        # F39：第100周期与第10周期的CC阶段电压的偏度差值
        skewness_cc = self.get_skewness_stage(100, 'CC-0', 'V') - self.get_skewness_stage(10, 'CC-0', 'V')

        # F40：第100周期与第10周期的CV阶段电流的偏度差值
        skewness_cv = self.get_skewness_stage(100, 'CV', 'I') - self.get_skewness_stage(10, 'CV', 'I')

        # F41：第100周期与第10周期的CC阶段电压的峰度差值
        kurtosis_cc = self.get_kurtosis_stage(100, 'CC-0', 'V') - self.get_kurtosis_stage(10, 'CC-0', 'V')

        # F42：第100周期与第10周期的CV阶段电流的峰度差值
        kurtosis_cv = self.get_kurtosis_stage(100, 'CV', 'I') - self.get_kurtosis_stage(10, 'CV', 'I')

        # F43-F44：弗雷歇距离（Frechet Distance）差值
        # F45-F46：豪斯多夫距离（Hausdorff Distance）差值
        # 计算CC阶段的距离差值
        FD_cc_10, HD_cc_10 = self.get_FD_HD_stage(10, 'CC-0')  # 第10周期
        FD_cc_100, HD_cc_100 = self.get_FD_HD_stage(100, 'CC-0')  # 第100周期
        FD_cc = FD_cc_100 - FD_cc_10  # F43
        HD_cc = HD_cc_100 - HD_cc_10  # F45

        # 计算CV阶段的距离差值
        FD_cv_10, HD_cv_10 = self.get_FD_HD_stage(10, 'CV')  # 第10周期
        FD_cv_100, HD_cv_100 = self.get_FD_HD_stage(100, 'CV')  # 第100周期
        FD_cv = FD_cv_100 - FD_cv_10  # F44
        HD_cv = HD_cv_100 - HD_cv_10  # F46

        # F35：第100周期与第10周期的CC阶段电压曲线熵差值
        cc_curve_entropy = self.get_curve_entropy(100, 'CC-0', 'V') - self.get_curve_entropy(10, 'CC-0', 'V')

        # F36：第100周期与第10周期的CV阶段电流曲线熵差值
        cv_curve_entropy = self.get_curve_entropy(100, 'CV', 'I') - self.get_curve_entropy(10, 'CV', 'I')


        # 将所有特征整合到字典并返回
        result_dict = {
            'F21': Cumulated_DQ_10_100,
            'F22': Cumulated_DC_Energy_10_100,
            'F23': cycle_time_100_2,
            'F24': V_s_of_charge,
            'F25': cc_t_100_10,
            'F26': cv_t_100_10,
            'F27': V_mean_cc_100_10,
            'F28': I_mean_cv_100_10,
            'F29': slope_v_t_100_10,
            'F30': slope_c_t_100_10,
            'F31': energy_cc,
            'F32': energy_cv,
            'F33': energy_ratio,
            'F34': energy_diff,
            'F35': cc_curve_entropy,
            'F36': cv_curve_entropy,
            'F37': s_entropy_v_100_10,
            'F38': s_entropy_c_100_10,
            'F39': skewness_cc,
            'F40': skewness_cv,
            'F41': kurtosis_cc,
            'F42': kurtosis_cv,
            'F43': FD_cc,
            'F44': FD_cv,
            'F45': HD_cc,
            'F46': HD_cv,
        }

        return result_dict


    def get_stage_energe(self, cycle_id, stage):
        """计算指定周期、指定阶段（CC/CV/放电）的能量（单位：Wh）"""
        # 获取阶段的起始和结束索引（st=start, ed=end）
        st, ed = self.get_cycle_stages(cycle_id)[stage]
        # 获取该阶段的电压、电流、时间数据
        Voltage = self.get_cycle_attr(cycle_id, 'V')  # 电压（V）
        Current = self.get_cycle_attr(cycle_id, 'I')  # 电流（A）
        Time = self.get_cycle_attr(cycle_id, 't')  # 时间（s）

        # 截取该阶段的数据
        Vd = Voltage[st: ed]
        Id = Current[st: ed]
        Td_minute = Time[st:ed]
        # 时间转换为小时（1小时=60分钟）
        Td_h = [g / 60 for g in Td_minute]

        # 能量计算：平均电压 × 平均电流 × 时间（小时）（取绝对值避免方向影响）
        Energy_dc = abs(np.mean(Vd)) * abs(np.mean(Id)) * abs(Td_h[-1] - Td_h[0])
        return Energy_dc


    def get_charge_time(self, cycle_id, charge_type):
        """计算指定周期、指定充电类型（CC/CV）的持续时间（单位：s）"""
        # 获取阶段的起始和结束索引
        st, ed = self.get_cycle_stages(cycle_id)[charge_type]
        # 获取时间数据
        t = self.get_cycle_attr(cycle_id, 't')
        # 结束时间 - 起始时间 = 持续时间
        return t[ed] - t[st]


    def get_mean_stage(self, cycle_id, stage, attr):
        """计算指定周期、指定阶段、指定属性（电压/电流）的平均值"""
        # 获取阶段的起始和结束索引
        st, ed = self.get_cycle_stages(cycle_id)[stage]
        # 获取属性数据（如电压'V'或电流'I'）
        attr_value = self.get_cycle_attr(cycle_id, attr)
        # 返回该阶段的属性平均值
        return np.mean(attr_value[st: ed])


    def get_slope(self, cycle_id, stage, combine):
        """计算指定周期、指定阶段的曲线斜率（电压-时间或电流-时间）"""
        # 获取阶段的起始和结束索引
        st, ed = self.get_cycle_stages(cycle_id)[stage]
        # 获取时间数据（x轴）
        t = self.get_cycle_attr(cycle_id, 't')[st:ed]
        
        # 根据combine参数选择y轴数据（电压或电流）
        if combine == 'VT':
            # 电压-时间曲线（V-t）
            y = self.get_cycle_attr(cycle_id, 'V')[st:ed]
        elif combine == 'CT':
            # 电流-时间曲线（I-t）
            y = self.get_cycle_attr(cycle_id, 'I')[st:ed]
        else:
            raise ValueError(f"无效的combine参数：{combine}（应为'VT'或'CT'）")

        # 一阶多项式拟合（线性回归），返回斜率
        coefficients = np.polyfit(t, y, deg=1)
        slope, intercept = coefficients  # intercept未使用
        return slope


    def get_shannon_entropy(self, cycle_id, stage, attr):
        """计算指定周期、指定阶段、指定属性的香农熵（微分熵）"""
        # 获取阶段的起始和结束索引
        st, ed = self.get_cycle_stages(cycle_id)[stage]
        # 获取属性曲线数据（如电压或电流）
        curve_v = self.get_cycle_attr(cycle_id, attr)[st:ed]
        # 调用工具函数计算微分熵（香农熵的连续形式）
        entropy = differential_entropy(curve_v)
        return entropy


    def get_skewness_stage(self, cycle_id, stage, attr):
        """计算指定周期、指定阶段、指定属性的偏度（衡量分布对称性）"""
        # 获取阶段的起始和结束索引
        st, ed = self.get_cycle_stages(cycle_id)[stage]
        # 获取属性数据
        data = self.get_cycle_attr(cycle_id, attr)[st:ed]
        
        mean = np.mean(data)  # 平均值
        std = np.std(data)    # 标准差
        # 偏度公式：E[(X-μ)/σ]^3（使用样本估计）
        return np.sum((data - mean) ** 3) / ((len(data) - 1) * (std ** 3))


    def get_kurtosis_stage(self, cycle_id, stage, attr):
        """计算指定周期、指定阶段、指定属性的峰度（衡量分布陡峭程度）"""
        # 获取阶段的起始和结束索引
        st, ed = self.get_cycle_stages(cycle_id)[stage]
        # 获取属性数据
        data = self.get_cycle_attr(cycle_id, attr)[st:ed]
        
        mean = np.mean(data)  # 平均值
        std = np.std(data)    # 标准差
        # 峰度公式：E[(X-μ)/σ]^4（使用样本估计）
        return np.sum((data - mean) ** 4) / ((len(data) - 1) * (std ** 4))


    def get_FD_HD_stage(self, cycle_id, stage):
        """计算指定周期、指定阶段的弗雷歇距离（FD）和豪斯多夫距离（HD）"""
        # 获取阶段的起始和结束索引
        st, ed = self.get_cycle_stages(cycle_id)[stage]

        # 提取电流和时间数据（按步长2采样，减少计算量）
        current = self.get_cycle_attr(cycle_id, 'I')[st: ed]
        current_time = self.get_cycle_attr(cycle_id, 't')[st: ed]
        index = [st + i for i in range(ed - st)]  # 索引序列（x轴）
        # 构建电流曲线和时间曲线的点集（(索引, 值)）
        current_curve = [(index[i], current[i]) for i in range(0, len(index), 2)]
        time_curve = [(index[i], current_time[i]) for i in range(0, len(index), 2)]

        # 转换为numpy数组（用于豪斯多夫距离计算）
        current_curve_H = np.array(current_curve)
        time_curve_H = np.array(time_curve)

        # 计算弗雷歇距离（调用工具函数）和有向豪斯多夫距离
        FD = frechet_distance(current_curve, time_curve)
        HD = directed_hausdorff(current_curve_H, time_curve_H)[0]  # 返回距离值（元组第一个元素）

        return FD, HD


    def get_curve_entropy(self, cycle_id, stage, attr):
        """计算指定周期、指定阶段、指定属性的曲线熵（基于几何特征）"""
        # 获取阶段的起始和结束索引
        st, ed = self.get_cycle_stages(cycle_id)[stage]

        # 提取x（时间）和y（属性值）数据
        y = self.get_cycle_attr(cycle_id, attr)[st:ed]
        x = self.get_cycle_attr(cycle_id, 't')[st:ed]
        # 构建曲线点集
        points = [(x1, y1) for x1, y1 in zip(x, y)]
        
        # 计算曲线长度、最小直径和点数量（用于熵计算）
        L = calculate_curve_length(points)  # 曲线长度
        D = calculate_min_diameter(points)  # 最小直径
        N = len(points)                     # 点数量

        # 调用工具函数计算曲线熵
        ec_value = calculate_ec(L, D, N)
        return ec_value


# 主程序：测试特征提取功能
if __name__ == "__main__":
    # 读取电池数据文件（pickle格式）
    file_path = r"./data/merged_batch.pkl"
    batch1 = pickle.load(open(file_path, 'rb'))
    
    battery_index = 0  # 电池索引计数器
    # 遍历数据集中的每颗电池
    for k, battery in batch1.items():
        # 初始化数据集实例
        d = DatasetTwo(battery, battery_index)
        # 提取特征
        result = d.extract()
        # 打印特征值
        for k, v in result.items():
            print(f"{k}: {v}")
        battery_index += 1
        break  # 只处理第一颗电池（调试用）
        # 如需处理前10颗电池，可取消下面注释
        # if battery_index == 10:
        #     break