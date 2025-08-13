import pickle
import os

# 输入输出文件路径
ipk_path = r"data\MATR\MATR_b1c0.pkl"
txt_path = r"result\MATR_b1c0.txt"

# 检查输入文件是否存在
if not os.path.exists(ipk_path):
    print(f"错误：文件 {ipk_path} 不存在")
    exit(1)

# 确保输出目录存在
os.makedirs(os.path.dirname(txt_path), exist_ok=True)

# 读取PKL文件并写入TXT
with open(ipk_path, 'rb') as pkl_file, open(txt_path, 'w', encoding='utf-8') as txt:
    data = pickle.load(pkl_file)
    
    # 写入数据类型和基本信息
    txt.write(f"数据类型: {type(data)}\n")
    txt.write(f"数据内容:\n")
    txt.write("=" * 50 + "\n")
    
    # 根据数据类型写入内容
    if isinstance(data, dict):
        for key, value in data.items():
            txt.write(f"{key}: {value}\n")
    elif isinstance(data, (list, tuple)):
        for i, item in enumerate(data):
            txt.write(f"[{i}]: {item}\n")
    else:
        txt.write(str(data))

print(f"转换完成：{ipk_path} -> {txt_path}")
    