import pickle  # 用于读取pickle格式的数据文件
import matplotlib.pyplot as plt  # 用于数据可视化（此处未实际绘图）
import numpy as np  # 用于数值计算和数组操作
import math  # 用于数学运算（如对数计算）

from MIT.collect_base import BaseDataset  # 导入基础数据集类
from scipy import integrate  # 用于积分计算

# 针对单颗电池的数据集处理类
class DatasetOne(BaseDataset):
    def __init__(self, battery, battery_index):
        # 调用父类BaseDataset的初始化方法
        super().__init__(battery, battery_index)

    def test_main(self):
        # 测试方法：打印电池数据的长度（周期数）
        print(len(self.battery))

    def extract(self):
        """提取电池的20个特征，返回特征字典"""
        # 计算第100个周期与第10个周期的放电容量曲线差值（△Q(V)）
        Diff_100_10 = self.get_cycle_attr(100, 'Qdlin') - self.get_cycle_attr(10, 'Qdlin')
        # 计算△Q(V)的方差（自由度为1）
        var = np.var(Diff_100_10, ddof=1)
        
        # F1：△Q(V)最小值的绝对值的对数（以10为底）
        min_diff = math.log(np.abs(np.min(Diff_100_10)), 10)
        # F2：△Q(V)平均值的绝对值的对数
        mean_diff = math.log(np.abs(np.mean(Diff_100_10)), 10)
        # F3：△Q(V)方差的绝对值的对数
        vardQ = math.log(np.abs(var), 10)

        # F4：△Q(V)的偏度（Skewness）的对数
        Skewness = self.get_Skewness(Diff_100_10)
        # F5：△Q(V)的峰度（Kurtosis）的对数
        Kurtosis = self.get_Kurtosis(Diff_100_10)

        # F6：2V电压处△Q(V)值的绝对值的对数（假设Qdlin最后一个元素对应2V）
        v2v = math.log(abs(Diff_100_10[-1]), 10)

        # 提取放电容量曲线的斜率和截距（用于特征F7-F10）
        # F7-F8：第2-100周期放电容量曲线的斜率和截距
        slope_2_100, intercept_2_100 = self.get_QD_slope(2, 100)
        # F9-F10：第91-100周期放电容量曲线的斜率和截距
        slope_91_100, intercept_91_100 = self.get_QD_slope(91, 100)

        # F11：第2周期的放电容量（QD）
        QD_2 = self.battery['summary']['QD'][1]  # 索引从0开始，故第2周期为索引1

        # F12：第2-100周期中放电容量与第2周期的最大差值
        DiffM2 = max((self.battery['summary']['QD'][1:100]) - self.battery['summary']['QD'][1])

        # F13：第100周期的放电容量
        QD_100 = self.battery['summary']['QD'][99]  # 索引99对应第100周期

        # F14：第2-6周期的平均充电时间
        chargetime_F5 = self.battery['summary']['chargetime'][1:6]  # 索引1-5对应第2-6周期
        Avg_ChargeTime = np.mean(chargetime_F5)

        # F15：第2-100周期的最高温度
        Tmax_2_100 = np.max(self.battery['summary']['Tmax'][1:100])
        # F16：第2-100周期的最低温度
        Tmin_2_100 = np.min(self.battery['summary']['Tmin'][1:100])

        # F17：第2-100周期的温度-时间积分值
        T_integal_2_100 = self.get_integal_T_t(2, 100)

        # F18：第2周期的内阻（IR）
        IR_2 = self.battery['summary']['IR'][1]
        # F19：第2-100周期的最小内阻
        IR_min_2_100 = np.min(self.battery['summary']['IR'][1:100])
        # F20：第100周期与第2周期的内阻差值
        IR_100_2 = self.battery['summary']['IR'][99] - self.battery['summary']['IR'][1]

        # 将所有特征整合到字典中返回
        result_dict = {
            'F1': min_diff, 'F2': mean_diff, 'F3': vardQ,
            'F4': Skewness, 'F5': Kurtosis, 'F6': v2v,
            'F7': slope_2_100, 'F8': intercept_2_100,
            'F9': slope_91_100, 'F10': intercept_91_100,
            'F11': QD_2, 'F12': DiffM2, 'F13': QD_100,
            'F14': Avg_ChargeTime, 'F15': Tmax_2_100,
            'F16': Tmin_2_100, 'F17': T_integal_2_100,
            'F18': IR_2, 'F19': IR_min_2_100, 'F20': IR_100_2
        }
        return result_dict

    def get_integal_T_t(self, start, end):
        """计算指定周期范围内的温度-时间积分值"""
        # 存储每个周期的平均温度
        T_list = []
        for idx in range(start, end + 1):
            # 获取当前周期的温度数据，计算平均值并添加到列表
            T_list.append(np.mean(np.array(self.get_cycle_attr(idx, 'T'))))
        
        # 生成x轴（周期数）：从start到end的线性空间（共end-start个点）
        X = np.linspace(start, end, end - start)
        # 计算温度对周期数的积分（梯形法）
        sum_integal_2 = integrate.trapz(T_list, X)
        return sum_integal_2

    def get_Skewness(self, data):
        """计算数据的偏度（Skewness）并取对数"""
        mean = np.mean(data)  # 计算平均值
        # 分子：(数据-均值)立方的平均值
        numerator = np.mean((data - mean) **3)
        # 分母：(数据-均值)平方和的平方根的三次方
        denominator = (np.sqrt(np.sum((data - mean)** 2))) **3
        
        if denominator == 0:
            raise ValueError("分母为零，无法计算偏度。")
        
        fraction = numerator / denominator  # 偏度核心计算
        return np.log(np.abs(fraction))  # 取绝对值的对数

    def get_QD_slope(self, start, end):
        """计算指定周期范围内放电容量（QD）随周期变化的斜率和截距"""
        # 提取start到end周期的放电容量数据（索引从start-1到end-1）
        QDis = self.battery['summary']['QD'][start-1: end]
        
        # 修正异常值：若某周期放电容量>1.3，则用前一周期值替代
        for idx in range(1, len(QDis)):
            if QDis[idx] > 1.3:
                QDis[idx] = QDis[idx-1]
        
        # 生成x轴（周期数）
        cycle_x = np.arange(start, end + 1)
        # 一阶多项式拟合（线性回归），返回斜率和截距
        coefficients = np.polyfit(cycle_x, QDis, deg=1)
        slope, intercept = coefficients
        return slope, intercept

    def get_Kurtosis(self, data):
        """计算数据的峰度（Kurtosis）并取对数"""
        mean = np.mean(data)  # 计算平均值
        # 分子：(数据-均值)四次方的平均值
        numerator = np.mean((data - mean) **4)
        # 分母：(数据-均值)平方的平均值的平方
        denominator = (np.mean((data - mean)** 2)) **2
        
        if denominator == 0:
            raise ValueError("分母为零，无法计算峰度。")
        
        fraction = numerator / denominator  # 峰度核心计算
        return np.log(np.abs(fraction))  # 取绝对值的对数


if __name__ == "__main__":
    # 读取pickle格式的电池数据文件
    file_path = r"./data/MATR/MATR_b1c1.pkl"
    batch1 = pickle.load(open(file_path, 'rb'))
    
    battery_index = 0  # 电池索引计数器
    # 遍历数据集中的每颗电池
    for k, battery in batch1.items():
        # 初始化单颗电池的数据集处理实例
        d = DatasetOne(battery, battery_index)
        # 提取特征并获取结果字典
        result_dict = d.extract()
        battery_index += 1  # 更新电池索引
        # 若需要调试，可取消下面注释查看前10颗电池的特征
        # if battery_index == 10:
        #     break