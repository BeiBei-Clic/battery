import pickle  # 用于读取pickle格式的电池数据文件
import numpy as np  # 用于数值计算和数组操作
import math  # 用于数学运算
import bisect  # 用于二分查找，定位电压/电流关键点

from utils import *  # 导入自定义工具函数（包含曲线平滑、拐点计算等）
from scipy.spatial.distance import directed_hausdorff  # 用于计算有向豪斯多夫距离
from collect_base import BaseDataset  # 导入基础数据集类
from kneed import KneeLocator  # 用于检测曲线拐点（膝盖点）

import matplotlib.pyplot as plt  # 用于数据可视化（调试用）


# 针对单颗电池的数据集处理类（提取F47-F59特征）
class DatasetThree(BaseDataset):
    def __init__(self, battery, battery_index):
        # 调用父类构造方法初始化电池数据和索引
        super().__init__(battery, battery_index)


    def extract(self):
        """提取电池的13个特征（F47-F59），返回特征字典"""
        
        # F47：第100周期与第10周期的平均电压波动（MVF）差值
        mvf_100_10 = self.get_MVF(100) - self.get_MVF(10)

        # F48：第100周期与第10周期的CC3阶段（3.4-3.6V）时间差值
        t_cc3_100_10 = self.get_cc3_diff(100, 't') - self.get_cc3_diff(10, 't')

        # F49：第100周期与第10周期的CC3阶段（3.4-3.6V）充电容量差值
        Q_cc3_100_10 = self.get_cc3_diff(100, 'Qc') - self.get_cc3_diff(10, 'Qc')

        # F52：第100周期与第10周期的CC3阶段（3.4-3.6V）温度变化率差值
        T_cc3_100_10 = self.get_cc3_diff(100, 'T') - self.get_cc3_diff(10, 'T')

        # F50：第100周期与第10周期的CV阶段（1A-0.1A）时间差值
        t_cv_100_10 = self.get_cv_diff(100, 't') - self.get_cv_diff(10, 't')

        # F51：第100周期与第10周期的CV阶段（1A-0.1A）充电容量差值
        Q_cv_100_10 = self.get_cv_diff(100, 'Qc') - self.get_cv_diff(10, 'Qc')

        # F53：第100周期与第10周期的CV阶段（1A-0.1A）温度变化率差值
        T_cv_100_10 = self.get_cv_diff(100, 'T') - self.get_cv_diff(10, 'T')

        # F54：第100周期与第10周期的CC阶段总充电容量差值
        cc_charge_capacity_100_10 = self.get_charge_capicity(100, 'CC-0') - self.get_charge_capicity(10, 'CC-0')

        # F55：第100周期与第10周期的CV阶段总充电容量差值
        cv_charge_capacity_100_10 = self.get_charge_capicity(100, 'CV') - self.get_charge_capicity(10, 'CV')

        # F56：第100周期与第10周期的CC阶段结束时电压-时间曲线斜率差值
        cc_last_vt_slope = self.get_last_slope(100) - self.get_last_slope(10)

        # F57：第100周期与第10周期的CC阶段起始拐点处垂直斜率差值
        cc_begin_elbows_slope = self.get_elbows_slope(100) - self.get_elbows_slope(10)

        # F58：最大放电容量对应的周期索引（第2-100周期）
        qdischarge = self.battery['summary']['QD'][1:100]  # 提取第2-100周期的放电容量
        qdischarge[qdischarge > 1.3] = 0  # 过滤异常值（大于1.3的视为无效）
        max_qd_cycle = np.argmax(qdischarge) + 2  # 计算最大容量对应的周期（+2是因为索引从0开始）
        diff_qd_index = max_qd_cycle.astype(np.float64)  # 转换为浮点数作为特征

        # F59：从第1周期到最大容量周期的总充放电时间
        charge_and_dis_time = self.get_c_dc_time()


        # 整合所有特征到字典并返回
        result_dict = {
            'F47': mvf_100_10,
            'F48': t_cc3_100_10,
            'F49': Q_cc3_100_10,
            'F50': t_cv_100_10,
            'F51': Q_cv_100_10,
            'F52': T_cc3_100_10,
            'F53': T_cv_100_10,
            'F54': cc_charge_capacity_100_10,
            'F55': cv_charge_capacity_100_10,
            'F56': cc_last_vt_slope,
            'F57': cc_begin_elbows_slope,
            'F58': diff_qd_index,
            'F59': charge_and_dis_time,
        }

        return result_dict


    def get_MVF(self, cycle_id):
        """计算指定周期的平均电压波动（Mean Voltage Fluctuation, MVF）"""
        # 获取该周期的电压和时间数据
        V = self.get_cycle_attr(cycle_id, 'V')
        t = self.get_cycle_attr(cycle_id, 't')
        # 获取放电阶段的起始索引
        discharge_begin = self.get_cycle_stages(cycle_id)['discharge_begin']
        
        # 找到放电开始后5分钟的索引（用于截取稳定放电阶段）
        for i in range(discharge_begin, len(t)):
            if t[i] - t[discharge_begin] > 5:  # 时间差超过5秒（此处单位可能需确认）
                dsc_5_min = i
                break
        
        # 截取放电5分钟后100个点的电压数据，计算与3.6V的平均偏差
        V_100 = V[dsc_5_min:dsc_5_min+100]
        mvf = np.mean(abs(V_100 - 3.6))  # 平均绝对偏差
        return mvf


    def get_cc_3_beign_V(self, cycle_id=100):
        """获取指定周期CC3阶段开始时的电压（辅助调试用）"""
        st = self.get_cycle_stages(cycle_id)['cc3_begin']  # CC3阶段起始索引
        V = self.get_cycle_attr(cycle_id, 'V')  # 电压数据
        return V[st]


    def get_cc3_diff(self, cycle_id, attr):
        """计算CC阶段中电压从3.4V到3.6V的属性（时间/容量/温度）差值"""
        # 获取CC阶段的起始和结束索引
        st, ed = self.get_cycle_stages(cycle_id)['CC']
        # 截取CC阶段的电压数据
        V = self.get_cycle_attr(cycle_id, 'V')
        V_bt = V[st:ed]  # bt=between
        
        # 用二分查找定位3.4V和3.6V在电压序列中的位置
        id_1 = bisect.bisect_left(V_bt, 3.4)  # 3.4V的索引
        id_2 = bisect.bisect_left(V_bt, 3.6)  # 3.6V的索引
        
        # 获取目标属性（t时间/Qc容量/T温度）并计算差值
        a = self.get_cycle_attr(cycle_id, attr)
        diff_a = a[id_2 + st] - a[id_1 + st]  # 加上st偏移到全局索引
        
        # 温度属性特殊处理：计算相对变化率（除以室温30℃）
        if attr == 'T':
            diff_a = diff_a / 30
        return diff_a


    def get_cv_diff(self, cycle_id, attr):
        """计算CV阶段中电流从1A到0.1A的属性（时间/容量/温度）差值"""
        # 获取CV阶段的起始和结束索引
        st, ed = self.get_cycle_stages(cycle_id)['CV']
        # 截取CV阶段的电流数据（取负值以便二分查找）
        I = self.get_cycle_attr(cycle_id, 'I')
        I_bt = I[st:ed]
        inverse_I_bt = [-item for item in I_bt]  # 转为负值，使电流从大到小变为从小到大
        
        # 用二分查找定位1A和0.1A在电流序列中的位置（通过负值转换）
        id_1 = bisect.bisect_left(inverse_I_bt, -1)  # 对应1A的索引
        id_2 = bisect.bisect_left(inverse_I_bt, -0.1)  # 对应0.1A的索引
        
        # 获取目标属性并计算差值
        a = self.get_cycle_attr(cycle_id, attr)
        diff_a = a[id_2 + st] - a[id_1 + st]  # 加上st偏移到全局索引
        
        # 温度属性特殊处理：计算相对变化率
        if attr == 'T':
            diff_a = diff_a / 30
        return diff_a


    def get_last_slope(self, cycle_id, no_points=5):
        """计算CC阶段结束前最后n个点的电压-时间曲线斜率"""
        # 获取CC阶段的起始和结束索引
        st, ed = self.get_cycle_stages(cycle_id)['CC']
        # 截取结束前no_points个点的电压和时间数据
        V = self.get_cycle_attr(cycle_id, 'V')
        t = self.get_cycle_attr(cycle_id, 't')
        x_last = t[ed - no_points: ed + 1]  # 时间（x轴）
        y_last = V[ed - no_points: ed + 1]  # 电压（y轴）
        
        # 线性拟合计算斜率
        slope, intercept = np.polyfit(x_last, y_last, 1)
        return slope


    def get_charge_capicity(self, cycle_id, stage):
        """计算指定周期、指定阶段（CC/CV）的充电容量（单位：mAh）"""
        # 获取阶段的起始和结束索引
        st, ed = self.get_cycle_stages(cycle_id)[stage]
        # 获取充电容量数据并计算差值（结束值 - 起始值）
        Qc = self.get_cycle_attr(cycle_id, 'Qc')
        diff = Qc[ed] - Qc[st]
        return diff


    def get_elbows_slope(self, cycle_id):
        """计算CC阶段起始处拐点的垂直斜率（曲线突变处的斜率）"""
        stages = self.get_cycle_stages(cycle_id)  # 获取该周期的阶段划分
        
        # 截取CC阶段起始部分的数据（前200个点）
        st = 0
        ed = 200  # 固定取前200点，覆盖CC阶段起始部分
        # 稀疏采样（步长5）减少噪声和计算量
        V_bt = self.get_cycle_attr(cycle_id, 'V')[st:ed:5]
        t_bt = self.get_cycle_attr(cycle_id, 't')[st:ed:5]
        
        # 平滑电压曲线以减少噪声干扰
        V_smooth = smooth_curve(V_bt)
        # 用KneeLocator检测曲线拐点（ concave表示 concave曲线，increasing表示上升趋势）
        kneedle = KneeLocator(t_bt, V_smooth, curve='concave', direction='increasing')
        elbow_point = kneedle.elbow  # 拐点对应的时间值
        
        # 找到拐点在原始数据中的索引
        index = find_closest_index(t_bt, elbow_point)
        
        # 计算拐点处的垂直斜率（调用工具函数）
        slope = perpendicular_slope_at_inflection(t_bt, V_bt, index)
        return slope


    def get_c_dc_time(self):
        """计算从第1周期到最大放电容量周期的总充放电时间"""
        # 提取第2-100周期的放电容量数据
        qdischarge = self.battery['summary']['QD'][1:100]
        qdischarge[qdischarge > 1.3] = 0  # 过滤异常值
        
        # 找到最大放电容量对应的周期
        max_qd_index = np.argmax(qdischarge) + 2  # +2是因为索引从0开始，对应第2周期起
        
        # 累计充放电时间
        all_discharge_time = 0  # 总放电时间
        all_charge_time = 0     # 总充电时间
        charge_time_list = self.battery['summary']['chargetime']  # 各周期充电时间列表
        
        # 遍历到最大容量周期，累计时间
        for cycle_idx in range(max_qd_index):
            # 计算当前周期的放电时间
            Current_now = self.get_cycle_attr(cycle_idx + 1, 'I')  # 电流数据（+1是因为周期从1开始）
            dis_begin, dis_end = self.get_discharge_time(Current_now)  # 获取放电阶段的起止索引
            timeline_now = self.get_cycle_attr(cycle_idx + 1, 't')  # 时间数据
            discharge_time = timeline_now[dis_end] - timeline_now[dis_begin]  # 放电持续时间
            all_discharge_time += discharge_time
            
            # 累加充电时间（跳过异常长的充电时间）
            temp_idx = cycle_idx
            while charge_time_list[temp_idx] > 100 and temp_idx < 100:  # 过滤>100秒的异常值
                temp_idx += 1
            all_charge_time += charge_time_list[temp_idx]
        
        # 总时间 = 总充电时间 + 总放电时间
        charge_and_dis_time = all_charge_time + all_discharge_time
        return charge_and_dis_time


    def get_discharge_time(self, Current):
        """根据电流数据定位放电阶段的起止索引"""
        discharge_begin = 0  # 放电开始索引
        discharge_end = 0    # 放电结束索引
        
        # 定位放电开始：电流从0变为-1（假设-1表示放电开始）
        for i in range(len(Current) - 1):
            if [int(Current[i]), int(float(Current[i + 1]))] == [0, -1]:
                discharge_begin = i
        
        # 定位放电结束：电流从-4变为大于-4（假设-4表示深度放电）
        for i in range(len(Current) - 1):
            if (int(Current[i]) == -4) & (float(Current[i + 1]) > -4):
                discharge_end = i
        
        return discharge_begin, discharge_end


# 主程序：测试特征提取功能
if __name__ == "__main__":
    # 读取电池数据文件（pickle格式）
    file_path = r"./data/merged_batch.pkl"
    batch1 = pickle.load(open(file_path, 'rb'))
    
    battery_index = 0  # 电池索引计数器
    all_cc3_st_V = []  # 存储CC3阶段起始电压（调试用）
    
    # 遍历数据集中的每颗电池
    for k, battery in batch1.items():
        # 初始化数据集实例
        d = DatasetThree(battery, battery_index)
        
        # 调试用：收集CC3阶段起始电压
        # temp_V = d.get_cc_3_beign_V()
        # all_cc3_st_V.append(temp_V)
        
        # 提取特征
        result = d.extract()
        # 打印特征值
        for k, v in result.items():
            print(f"{k}: {v}")
        
        battery_index += 1
        break  # 只处理第一颗电池（调试用）
        # 如需处理更多电池，可修改条件
        # if battery_index == 10:
        #     break
    
    # 调试用：打印最小CC3起始电压
    # min_v = min(all_cc3_st_V)
    # print("min V is ", min_v)