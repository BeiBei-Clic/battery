import pickle
import numpy as np
import math
import bisect

from utils import *
from scipy.spatial.distance import directed_hausdorff

from collect_base import BaseDataset
from kneed import KneeLocator

import matplotlib.pyplot as plt

# 针对一颗电池
class DatasetThree(BaseDataset):
    def __init__(self, battery, battery_index):
        super().__init__(battery, battery_index)

    def extract(self):
        # MVF at 100
        mvf_100_10 = self.get_MVF(100)-self.get_MVF(10)

        # The time interval of an equal charging voltage difference（cc3阶段 3.4-3.6v）
        t_cc3_100_10 = self.get_cc3_diff(100, 't') - self.get_cc3_diff(10, 't')

        # Charging capacity in CC phase（cc3阶段 3.4-3.6v）
        Q_cc3_100_10 = self.get_cc3_diff(100, 'Qc') - self.get_cc3_diff(10, 'Qc')

        # The temperature changing rate of an equal charging voltage（cc阶段 3.4-3.6v）
        T_cc3_100_10 = self.get_cc3_diff(100, 'T') - self.get_cc3_diff(10, 'T')

        # The time interval of an equal charging current difference（cv阶段 1A-0.1A）
        t_cv_100_10 = self.get_cv_diff(100, 't') - self.get_cv_diff(10, 't')

        # Charging capacity in CV phase（cv阶段 1A-0.1A）
        Q_cv_100_10 = self.get_cv_diff(100, 'Qc') - self.get_cv_diff(10, 'Qc')

        # The temperature changing rate of an equal charging voltage（cv阶段 1A-0.1A）

        T_cv_100_10 = self.get_cv_diff(100, 'T') - self.get_cv_diff(10, 'T')

        # CC charge capacity # 54
        cc_charge_capacity_100_10 = self.get_charge_capicity(100, 'CC-0') - self.get_charge_capicity(10, 'CC-0')

        # CV charge capacity # F55
        cv_charge_capacity_100_10 = self.get_charge_capicity(100, 'CV') - self.get_charge_capicity(10, 'CV')

        # The slope of the curve at the end of CC charging mode # F56
        cc_last_vt_slope = self.get_last_slope(100) - self.get_last_slope(10)

        # The vertical slope at the corner of the CC charging curve # F57
        cc_begin_elbows_slope = self.get_elbows_slope(100) - self.get_elbows_slope(10)

        # cycle with maximum capacity
        qdischarge = self.battery['summary']['QD'][1:100]

        qdischarge[qdischarge > 1.3] = 0
        max_qd_cycle = np.argmax(qdischarge) + 2

        diff_qd_index = max_qd_cycle.astype(np.float64)  # F59
        # time with maximum capacity
        # diff_max_time = 0  # F60
        # for cycle_idx in range(3, max_qd_cycle + 1):
        #     diff_max_time += self.get_cycle_attr(cycle_idx, 't')[-1]
        #
        # diff_max_time = diff_max_time / 60.0  # hours

        charge_and_dis_time = self.get_c_dc_time()

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

        ## 1
        # Diff_100_10 = self.get_cycle_attr(100,'Qdlin') - self.get_cycle_attr(10,'Qdlin')
        # var = np.var(Diff_100_10, ddof=1)
        # min_diff = np.min(Diff_100_10) # F1
        # mean_diff = np.mean(Diff_100_10) # F2
        # vardQ = math.log(abs(var), 10)  # F3
        #
        # Skewness = self.get_Skewness(Diff_100_10) # F4
        #
        # Kurtosis = self.get_Kurtosis(Diff_100_10) # F5
        #
        # # Value at 2 V
        # v2v = math.log(abs(Diff_100_10[-1]), 10)
        #
        #
        #
        # # 提取斜率和截距
        # slope_2_100, intercept_2_100 = self.get_QD_slope(2,100)
        # slope_91_100, intercept_91_100 = self.get_QD_slope(91,100)
        #
        # # F11 ~ F13
        # QD_2 = self.battery['summary']['QD'][1]
        #
        # DiffM2 = max((self.battery['summary']['QD'][1:100]) -
        #              self.battery['summary']['QD'][1])
        #
        # QD_100 = self.battery['summary']['QD'][99]
        #
        # # F14
        # chargetime_F5 = self.battery['summary']['chargetime'][1:6]
        # Avg_ChargeTime = np.mean(chargetime_F5)
        #
        # # max min T
        # Tmax_2_100 = np.max(self.battery['summary']['Tmax'][1:101])
        # Tmin_2_100 = np.min(self.battery['summary']['Tmin'][1:101])
        #
        # # integal T-t
        #
        # T_integal_2_100 = self.get_integal_T_t(2,100)
        #
        # # IR
        # IR_2 = self.battery['summary']['IR'][1]
        #
        # IR_min_2_100 = np.min(self.battery['summary']['IR'][1:100])
        #
        # IR_100_2 = self.battery['summary']['IR'][99] - self.battery['summary']['IR'][1]
        #

        # print(result_dict)

        return result_dict

    def get_MVF(self, cycle_id):
        V = self.get_cycle_attr(cycle_id, 'V')
        t = self.get_cycle_attr(cycle_id, 't')
        discharge_begin = self.get_cycle_stages(cycle_id)['discharge_begin']
        # 放电五分钟
        for i in range(discharge_begin, len(t)):
            if t[i] - t[discharge_begin] > 5:
                dsc_5_min = i
                break

        V_100 = V[dsc_5_min:dsc_5_min+100]
        mvf = np.mean(abs(V_100-3.6))

        return mvf

    def get_cc_3_beign_V(self, cycle_id=100):
        st = self.get_cycle_stages(cycle_id)['cc3_begin']
        V = self.get_cycle_attr(cycle_id,'V')
        return V[st]

    def get_cc3_diff(self, cycle_id, attr):
        st, ed = self.get_cycle_stages(cycle_id)['CC']
        V = self.get_cycle_attr(cycle_id, 'V')
        V_bt = V[st:ed]
        id_1 = bisect.bisect_left(V_bt, 3.4)
        id_2 = bisect.bisect_left(V_bt, 3.6)
        a = self.get_cycle_attr(cycle_id, attr)
        diff_a = a[id_2+st] - a[id_1+st]

        if attr == 'T':
            # 变化率 除以室温 30℃
            diff_a = diff_a / 30
        return diff_a

    def get_cv_diff(self, cycle_id, attr):
        st, ed = self.get_cycle_stages(cycle_id)['CV']
        I = self.get_cycle_attr(cycle_id, 'I')
        I_bt = I[st:ed]

        inverse_I_bt = [-item for item in I_bt]
        id_1 = bisect.bisect_left(inverse_I_bt, -1)
        id_2 = bisect.bisect_left(inverse_I_bt, -0.1)
        a = self.get_cycle_attr(cycle_id, attr)
        diff_a = a[id_2 + st] - a[id_1 + st]

        if attr == 'T':
            # 变化率 除以室温 30℃
            diff_a = diff_a / 30
        return diff_a

    def get_last_slope(self, cycle_id, no_points=5):
        st, ed = self.get_cycle_stages(cycle_id)['CC']
        V = self.get_cycle_attr(cycle_id,'V')
        t = self.get_cycle_attr(cycle_id,'t')

        x_last = t[ed-no_points: ed + 1]
        y_last = V[ed-no_points: ed + 1]

        slope, intercept = np.polyfit(x_last, y_last, 1)

        return slope


    def get_charge_capicity(self, cycle_id, stage):

        st, ed = self.get_cycle_stages(cycle_id)[stage]


        Qc = self.get_cycle_attr(cycle_id, 'Qc')



        diff = Qc[ed] - Qc[st]

        return diff

    def get_elbows_slope(self, cycle_id):
        stages = self.get_cycle_stages(cycle_id)

        st = 0
        ed = 200
        # if 'cc1_end' in stages.keys():
        #     ed = stages['cc1_end']
        # else:
        #     if 'cc2_end' in stages.keys():
        #         ed = stages['cc2_end']
        #     else:
        #         ed = stages['cc3_begin']



        V_bt = self.get_cycle_attr(cycle_id, 'V')[st:ed:5] # 稀疏采样避免噪声
        t_bt = self.get_cycle_attr(cycle_id, 't')[st:ed:5]

        # 先将曲线平滑 再找拐点
        V_smooth = smooth_curve(V_bt)
        kneedle = KneeLocator(t_bt, V_smooth, curve='concave', direction='increasing')
        elbow_point = kneedle.elbow
        # print(elbow_point)

        index = find_closest_index(t_bt, elbow_point)

        # print(f"拐点的索引: {inflection_indices}")
        # if len(inflection_indices) != 1:
        #     plt.plot(t_bt, V_smooth)
        #     for index in inflection_indices:
        #         plt.plot(t_bt[index], V_smooth[index], 'r+')
        #     plt.show()

        # plt.plot(t_bt, V_smooth)
        # plt.plot(t_bt[index], V_smooth[index], 'r+')
        # plt.show()
        slope = perpendicular_slope_at_inflection(t_bt, V_bt, index)

        return slope


    def get_c_dc_time(self):
        qdischarge = self.battery['summary']['QD'][1:100]

        # 大于1.3的置零
        qdischarge[qdischarge > 1.3] = 0

        max_qd_index = np.argmax(qdischarge) + 2
        # diff_qd_index = max_qd_index.astype(np.float64) - 2.0 # F59
        diff_qd_index = max_qd_index.astype(np.float64)  # F59

        all_discharge_time = 0
        all_charge_time = 0
        charge_time_list = self.battery['summary']['chargetime']

        # for cycle_idx in range(2, max_qd_index):
        for cycle_idx in range(max_qd_index):
            Current_now = self.get_cycle_attr(cycle_idx+1, 'I')
            dis_begin, dis_end = self.get_discharge_time(Current_now)
            # dis_begin, dis_end = self.get_cycle_stages(cycle_idx+1)['Discharge']
            timeline_now = self.get_cycle_attr(cycle_idx+1,'t')
            discharge_time = timeline_now[dis_end] - timeline_now[dis_begin]
            all_discharge_time += discharge_time

            temp_idx = cycle_idx
            while charge_time_list[temp_idx] > 100 and temp_idx < 100:
                temp_idx += 1
            all_charge_time += charge_time_list[temp_idx]

        charge_and_dis_time = all_charge_time + all_discharge_time
        return charge_and_dis_time

    def get_discharge_time(self, Current):
        discharge_begin = 0
        discharge_end = 0
        for i in range(len(Current) - 1):
            if [int(Current[i]), int(float(Current[i + 1]))] == [0, -1]:
                discharge_begin = i

        for i in range(len(Current) - 1):
            if (int(Current[i]) == -4) & (float(Current[i + 1]) > -4):
                discharge_end = i

        return discharge_begin, discharge_end


if __name__ == "__main__":
    file_path = r"./data/merged_batch.pkl"
    batch1 = pickle.load(open(file_path, 'rb'))
    battery_index = 0
    all_cc3_st_V = []
    for k, battery in batch1.items():
        d = DatasetThree(battery, battery_index)

        # temp_V = d.get_cc_3_beign_V()
        # all_cc3_st_V.append(temp_V)
        # d.get_stage_energe(15, 'Discharge')
        result = d.extract()

        for k,v in result.items():
            print(f"{k}: {v}")
        battery_index += 1
        break
        # # if battery_index == 10:
            # a = 0

    # min_v = min(all_cc3_st_V)
    # print("min V is ", min_v)