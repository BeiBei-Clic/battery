# 导入必要的库
import re  # 用于正则表达式处理
import h5py  # 用于读取HDF5格式文件
import numpy as np  # 用于数值计算

from tqdm import tqdm  # 用于显示进度条
from typing import List  # 用于类型提示
from scipy.io import loadmat  # 用于读取MATLAB格式文件
from batteryml.builders import PREPROCESSORS  # 导入预处理器注册器
from batteryml.preprocess.base import BasePreprocessor  # 导入基础预处理器类
from batteryml import BatteryData, CycleData, CyclingProtocol  # 导入电池数据相关类


# 将该预处理器注册到系统中，使其可被调用
@PREPROCESSORS.register()
class MATRPreprocessor(BasePreprocessor):
    """
    MATR数据集专用预处理器，继承自基础预处理器类
    用于加载、清洗和转换MATR电池循环数据为标准化的BatteryData格式
    """
    def process(self, parentdir, **kwargs) -> List[BatteryData]:
        """
        处理入口函数，加载并预处理MATR数据集
        
        参数:
            parentdir: 数据根目录路径
            **kwargs: 其他可选参数
        
        返回:
            处理后的BatteryData对象列表
        """
        # 定义需要加载的原始数据文件路径
        raw_files = [
            parentdir / 'MATR_batch_20170512.mat',
            parentdir / 'MATR_batch_20170630.mat',
            parentdir / 'MATR_batch_20180412.mat',
            parentdir / 'MATR_batch_20190124.mat',
        ]

        data_batches = []  # 用于存储多个批次的数据
        # 如果不启用静默模式，为文件列表添加进度条
        if not self.silent:
            raw_files = tqdm(raw_files)

        # 遍历每个原始数据文件
        for indx, f in enumerate(raw_files):
            # 如果是tqdm进度条对象，设置当前处理的文件名
            if hasattr(raw_files, 'set_description'):
                raw_files.set_description(f'Loading {f.stem}')

            # 检查文件是否存在
            if not f.exists():
                raise FileNotFoundError(f'Batch file not found: {str(f)}')

            # 加载当前批次数据并添加到列表
            data_batches.append(load_batch(f, indx+1))

        # 清洗数据批次并返回处理后的电池数量
        batteries_num = clean_batches(
            data_batches, self.dump_single_file, self.silent)

        return batteries_num


def load_batch(file, k):
    """
    加载单个批次的MATR数据文件
    
    参数:
        file: 批次文件路径
        k: 批次编号
    
    返回:
        包含该批次所有电池数据的字典
    """
    # 打开HDF5格式文件
    with h5py.File(file, 'r') as f:
        batch = f['batch']  # 获取批次数据主节点
        num_cells = batch['summary'].shape[0]  # 获取电池数量
        bat_dict = {}  # 存储该批次所有电池数据的字典

        # 遍历每个电池
        for i in tqdm(range(num_cells), desc='Processing cells', leave=False):
            # 读取循环寿命数据
            cl = f[batch['cycle_life'][i, 0]][:]
            # 读取充电策略（解码为字符串）
            policy = f[batch['policy_readable'][i, 0]][:].tobytes()[::2].decode()
            
            # 读取摘要数据（多个性能指标的时间序列）
            summary_IR = np.hstack(f[batch['summary'][i, 0]]['IR'][0, :].tolist())  # 内阻
            summary_QC = np.hstack(f[batch['summary'][i, 0]]['QCharge'][0, :].tolist())  # 充电容量
            summary_QD = np.hstack(f[batch['summary'][i, 0]]['QDischarge'][0, :].tolist())  # 放电容量
            summary_TA = np.hstack(f[batch['summary'][i, 0]]['Tavg'][0, :].tolist())  # 平均温度
            summary_TM = np.hstack(f[batch['summary'][i, 0]]['Tmin'][0, :].tolist())  # 最低温度
            summary_TX = np.hstack(f[batch['summary'][i, 0]]['Tmax'][0, :].tolist())  # 最高温度
            summary_CT = np.hstack(f[batch['summary'][i, 0]]['chargetime'][0, :].tolist())  # 充电时间
            summary_CY = np.hstack(f[batch['summary'][i, 0]]['cycle'][0, :].tolist())  # 循环编号
            
            # 汇总摘要数据
            summary = {
                'IR': summary_IR,
                'QC': summary_QC,
                'QD': summary_QD,
                'Tavg': summary_TA,
                'Tmin': summary_TM,
                'Tmax': summary_TX,
                'chargetime': summary_CT,
                'cycle': summary_CY
            }
            
            # 读取循环数据
            cycles = f[batch['cycles'][i, 0]]
            cycle_dict = {}  # 存储每个循环的数据
            
            # 遍历每个循环
            for j in range(cycles['I'].shape[0]):
                I = np.hstack((f[cycles['I'][j, 0]][:]))  # 电流 (A)
                Qc = np.hstack((f[cycles['Qc'][j, 0]][:]))  # 充电容量 (Ah)
                Qd = np.hstack((f[cycles['Qd'][j, 0]][:]))  # 放电容量 (Ah)
                Qdlin = np.hstack((f[cycles['Qdlin'][j, 0]][:]))  # 线性放电容量
                T = np.hstack((f[cycles['T'][j, 0]][:]))  # 温度 (°C)
                Tdlin = np.hstack((f[cycles['Tdlin'][j, 0]][:]))  # 线性温度
                V = np.hstack((f[cycles['V'][j, 0]][:]))  # 电压 (V)
                dQdV = np.hstack((f[cycles['discharge_dQdV'][j, 0]][:]))  # 放电dQ/dV
                t = np.hstack((f[cycles['t'][j, 0]][:]))  # 时间 (分钟)
                
                # 存储当前循环数据
                cd = {
                    'I': I,
                    'Qc': Qc,
                    'Qd': Qd,
                    'Qdlin': Qdlin,
                    'T': T,
                    'Tdlin': Tdlin,
                    'V': V,
                    'dQdV': dQdV,
                    't': t
                }
                cycle_dict[str(j)] = cd  # 以循环编号为键存储
                
            # 整合当前电池的所有数据
            cell_dict = {
                'cycle_life': cl,  # 循环寿命
                'charge_policy': policy,  # 充电策略
                'summary': summary,  # 摘要数据
                'cycles': cycle_dict  # 循环数据
            }
            
            # 生成电池唯一标识（如b1c0表示第1批次第0个电池）
            key = f'b{k}c' + str(i)
            bat_dict[key] = cell_dict  # 存储到批次字典中
            
    return bat_dict


def clean_batches(data_batches, dump_single_file, silent):
    """
    清洗数据批次，处理异常电池数据并合并跨批次的电池数据
    
    参数:
        data_batches: 多个批次的数据
        dump_single_file: 用于保存单个电池数据的函数
        silent: 是否启用静默模式（不输出日志）
    
    返回:
        处理和跳过的电池数量
    """
    # 移除第1批次中未达到80%容量的异常电池
    del data_batches[0]['b1c8']
    del data_batches[0]['b1c10']
    del data_batches[0]['b1c12']
    del data_batches[0]['b1c13']
    del data_batches[0]['b1c22']

    # 第1批次的部分电池数据延续到第2批次，需要合并
    batch2_keys = ['b2c7', 'b2c8', 'b2c9', 'b2c15', 'b2c16']  # 第2批次中对应的键
    batch1_keys = ['b1c0', 'b1c1', 'b1c2', 'b1c3', 'b1c4']    # 第1批次中对应的键
    add_len = [662, 981, 1060, 208, 482]  # 第2批次延续的循环数

    # 合并跨批次的电池数据
    for i, bk in enumerate(batch1_keys):
        # 更新循环寿命（累加两个批次的循环数）
        data_batches[0][bk]['cycle_life'] = \
            data_batches[0][bk]['cycle_life'] + add_len[i]
        
        # 合并摘要数据
        for j in data_batches[0][bk]['summary'].keys():
            if j == 'cycle':  # 循环编号需要偏移（累加前一批次的循环数）
                data_batches[0][bk]['summary'][j] = np.hstack((
                    data_batches[0][bk]['summary'][j],
                    data_batches[1][batch2_keys[i]]['summary'][j]
                    + len(data_batches[0][bk]['summary'][j])
                ))
            else:  # 其他指标直接拼接
                data_batches[0][bk]['summary'][j] = np.hstack((
                    data_batches[0][bk]['summary'][j],
                    data_batches[1][batch2_keys[i]]['summary'][j]
                ))
        
        # 合并循环数据（偏移循环编号）
        last_cycle = len(data_batches[0][bk]['cycles'].keys())
        for j, jk in enumerate(data_batches[1][batch2_keys[i]]['cycles'].keys()):
            data_batches[0][bk]['cycles'][str(last_cycle + j)] = \
                data_batches[1][batch2_keys[i]]['cycles'][jk]

    # 统计处理和跳过的电池数量
    process_batteries_num = 0
    skip_batteries_num = 0
    
    # 处理所有批次的电池数据
    for batch in data_batches:
        for cell in batch:
            # 跳过已合并到第1批次的第2批次电池
            if cell not in batch2_keys:
                # 跳过需要丢弃的异常电池
                if cell in TO_DROP:
                    continue
                # 整理电池数据为标准化格式
                battery = organize_cell(batch[cell], cell)
                # 保存处理后的电池数据
                dump_single_file(battery)
                # 输出处理日志（非静默模式）
                if not silent:
                    tqdm.write(f'File: {battery.cell_id} dumped to pkl file')
                process_batteries_num += 1

    return process_batteries_num, skip_batteries_num


def organize_cell(data, name):
    """
    将单个电池的原始数据整理为BatteryData对象
    
    参数:
        data: 单个电池的原始数据
        name: 电池标识
    
    返回:
        标准化的BatteryData对象
    """
    cycle_data = []  # 存储循环数据的列表
    
    # 遍历每个循环
    for cycle in range(len(data['cycles'])):
        # 获取当前循环数据
        cur_data = data['cycles'][str(cycle)]
        
        # 跳过第0个循环（通常为初始化循环）
        if cycle == 0:
            continue
        
        # 处理特定电池的异常循环（跳过损坏的数据点）
        if name == 'b1c0' and cycle >= 11:
            if cycle + 1 == len(data['cycles']):
                continue
            else:
                cur_data = data['cycles'][str(cycle + 1)]
        elif name == 'b2c12' and cycle >= 252:
            if cycle + 1 == len(data['cycles']):
                continue
            cur_data = data['cycles'][str(cycle + 1)]
        elif name == 'b2c44' and cycle >= 247:
            if cycle + 1 == len(data['cycles']):
                continue
            cur_data = data['cycles'][str(cycle + 1)]
        elif name == 'b1c18' and cycle >= 39:
            if cycle + 1 == len(data['cycles']):
                continue
            cur_data = data['cycles'][str(cycle + 1)]
        
        # 转换时间单位（分钟→秒）
        time = cur_data['t'].tolist()
        time_in_s = [i * 60 for i in time]
        
        # 创建CycleData对象并添加到列表
        cycle_data.append(CycleData(
            cycle_number=cycle,  # 循环编号
            voltage_in_V=cur_data['V'].tolist(),  # 电压 (V)
            current_in_A=cur_data['I'].tolist(),  # 电流 (A)
            temperature_in_C=cur_data['T'].tolist(),  # 温度 (°C)
            discharge_capacity_in_Ah=cur_data['Qd'].tolist(),  # 放电容量 (Ah)
            charge_capacity_in_Ah=cur_data['Qc'].tolist(),  # 充电容量 (Ah)
            time_in_s=time_in_s,  # 时间 (s)
            internal_resistance_in_ohm=data['summary']['IR'][cycle],  # 内阻 (Ω)
            Qdlin=cur_data['Qdlin'].tolist()  # 线性放电容量
        ))

    # 定义放电协议（固定为4C倍率，从100% SOC到0% SOC）
    discharge_protocol = CyclingProtocol(
        rate_in_C=4.0, start_soc=1.0, end_soc=0.0
    )
    
    # 解析充电策略（从字符串提取多阶段充电参数）
    stages = [x for x in data['charge_policy'].split('-') if 'new' not in x]
    
    # 处理两阶段充电策略
    if len(stages) == 2:
        pattern = r'(.*?)C\((.*?)%\)'  # 匹配如"3C(50%)"格式的阶段
        rate1, end_soc = re.findall(pattern, stages[0])[0]  # 第一阶段参数
        # 第二阶段参数（提取倍率）
        rate2 = float(stages[1][:-1] if 'C' in stages[1] else stages[1])
        # 构建两阶段充电协议
        charge_protocol = [
            CyclingProtocol(
                rate_in_C=float(rate1),
                start_soc=0.0,
                end_soc=float(end_soc)),
            CyclingProtocol(
                rate_in_C=float(rate2),
                start_soc=float(end_soc),
                end_soc=1.0)
        ]
    else:
        # 处理多阶段充电策略（默认均匀分配SOC区间）
        charge_protocol = [
            CyclingProtocol(
                rate_in_C=float(x),
                start_soc=i*0.2,  # 假设5个阶段，每个阶段20% SOC
                end_soc=(i+1)*0.2
            ) for i, x in enumerate(stages)
        ]

    # SOC区间（0到1表示0%到100%）
    soc_interval = [0, 1]

    # 创建并返回标准化的BatteryData对象
    return BatteryData(
        cell_id=f'MATR_{name}',  # 电池唯一标识
        cycle_data=cycle_data,  # 所有循环数据
        form_factor='cylindrical_18650',  # 电池外形（18650圆柱型）
        anode_material='graphite',  # 负极材料（石墨）
        cathode_material='LiFePO4',  # 正极材料（磷酸铁锂）
        discharge_protocol=discharge_protocol,  # 放电协议
        charge_protocol=charge_protocol,  # 充电协议
        nominal_capacity_in_Ah=1.1,  # 标称容量（1.1 Ah）
        min_voltage_limit_in_V=2.0,  # 最低电压限制（2.0 V）
        max_voltage_limit_in_V=3.5,  # 最高电压限制（3.5 V）
        SOC_interval=soc_interval  # SOC区间
    )

# 需要丢弃的异常电池列表
TO_DROP = [
    'b3c2',
    'b3c23',
    'b3c32',
    'b3c37',
    'b3c42',
    'b3c43',
]