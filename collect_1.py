import pickle

import matplotlib.pyplot as plt
import numpy as np
import math

from MIT.collect_base import BaseDataset
from scipy import integrate

# 针对一颗电池
class DatasetOne(BaseDataset):
    def __init__(self, battery, battery_index):
        super().__init__(battery, battery_index)

    def test_main(self):
        print(len(self.battery))

    def extract(self):
        Diff_100_10 = self.get_cycle_attr(100,'Qdlin') - self.get_cycle_attr(10,'Qdlin')
        var = np.var(Diff_100_10, ddof=1)
        min_diff = math.log(np.abs(np.min(Diff_100_10)), 10) # F1
        mean_diff = math.log(np.abs(np.mean(Diff_100_10)), 10) # F2
        vardQ = math.log(np.abs(var), 10)  # F3


        Skewness = self.get_Skewness(Diff_100_10) # F4

        Kurtosis = self.get_Kurtosis(Diff_100_10) # F5

        # Value at 2 V
        v2v = math.log(abs(Diff_100_10[-1]), 10)



        # 提取斜率和截距
        slope_2_100, intercept_2_100 = self.get_QD_slope(2,100)
        slope_91_100, intercept_91_100 = self.get_QD_slope(91,100)

        # F11 ~ F13
        QD_2 = self.battery['summary']['QD'][1]

        DiffM2 = max((self.battery['summary']['QD'][1:100]) -
                     self.battery['summary']['QD'][1])

        QD_100 = self.battery['summary']['QD'][99]

        # F14
        chargetime_F5 = self.battery['summary']['chargetime'][1:6]
        Avg_ChargeTime = np.mean(chargetime_F5)

        # max min T
        Tmax_2_100 = np.max(self.battery['summary']['Tmax'][1:100])
        Tmin_2_100 = np.min(self.battery['summary']['Tmin'][1:100])

        # integal T-t

        T_integal_2_100 = self.get_integal_T_t(2,100)

        # IR
        IR_2 = self.battery['summary']['IR'][1]

        IR_min_2_100 = np.min(self.battery['summary']['IR'][1:100])

        IR_100_2 = self.battery['summary']['IR'][99] - self.battery['summary']['IR'][1]

        result_dict = {
            'F1': min_diff,
            'F2': mean_diff,
            'F3': vardQ,
            'F4': Skewness,
            'F5': Kurtosis,
            'F6': v2v,
            'F7': slope_2_100,
            'F8': intercept_2_100,
            'F9': slope_91_100,
            'F10': intercept_91_100,
            'F11': QD_2,
            'F12': DiffM2,
            'F13': QD_100,
            'F14': Avg_ChargeTime,
            'F15': Tmax_2_100,
            'F16': Tmin_2_100,
            'F17': T_integal_2_100,
            'F18': IR_2,
            'F19': IR_min_2_100,
            'F20': IR_100_2,

        }
        # print(result_dict)

        return result_dict
    def get_integal_T_t(self, start, end):

        sum_integal = 0

        # ours
        # for idx in range(start, end+1):
        #     T = self.get_cycle_attr(idx, 'T')
        #     t = self.get_cycle_attr(idx, 't') / 60 # hour
        #
        #     sum_integal += np.trapz(T, x=t, axis=-1)
        # print('sum_integal: ', sum_integal)
        # 师姐
        T_list = []
        for idx in range(start, end+1):
            T_list.append(np.mean(np.array(self.get_cycle_attr(idx, 'T'))))

        X = np.linspace(2, 100, 99)
        sum_integal_2 = integrate.trapz(T_list, X)
        #
        # print('sum_integal_2: ', sum_integal_2)

        return sum_integal_2

    def get_Skewness(self, data):

        """
        计算给定的公式
        :param data: 列表或数组，表示 △Q(V)
        :return: 公式的计算结果
        """
        # 计算平均值
        mean = np.mean(data)

        # 分子：△Q(V) 减去其平均值的立方
        numerator = np.mean((data - mean) ** 3)

        # 分母：△Q(V) 的平方和的平方根的三次方
        denominator = (np.sqrt(np.sum((data - mean) ** 2))) ** 3

        # 避免除以零
        if denominator == 0:
            raise ValueError("分母为零，无法计算公式。")

        # 计算分数
        fraction = numerator / denominator

        # 计算对数
        result = np.log(np.abs(fraction))

        return result

    def get_QD_slope(self, start, end):
        # Slope of discharge curve, cycles 2 to 100
        QDis = self.battery['summary']['QD'][start-1: end]  # mAh

        for idx in range(1, len(QDis)):
            if QDis[idx] > 1.3:
                QDis[idx] = QDis[idx-1]

        cycle_x = np.arange(start, end+1)
        # 最小二乘回归（一阶多项式）
        coefficients = np.polyfit(cycle_x, QDis, deg=1)
        slope, intercept = coefficients
        # if start == 2:
        #     plt.plot(cycle_x, QDis)
        #     plt.show()
        return slope, intercept
    def get_Kurtosis(self, data):

        """
        计算给定的公式
        :param data: 列表或数组，表示 △Q(V)
        :return: 公式的计算结果
        """
        # 计算平均值
        mean = np.mean(data)

        numerator = np.mean((data - mean) ** 4)

        denominator = (np.mean((data - mean) ** 2)) ** 2

        # 避免除以零
        if denominator == 0:
            raise ValueError("分母为零，无法计算公式。")

        # 计算分数
        fraction = numerator / denominator

        # 计算对数
        result = np.log(np.abs(fraction))

        return result



if __name__ == "__main__":
    file_path = r"./data/merged_batch.pkl"
    batch1 = pickle.load(open(file_path, 'rb'))
    battery_index = 0
    for k, battery in batch1.items():

        d = DatasetOne(battery, battery_index)
        result_dict = d.extract()
        battery_index += 1

        # break
        # if battery_index == 10:
            # a = 0