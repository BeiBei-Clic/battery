# 提取指定特征列：F1, F5, F11, F14, F59, Cycle_Life
input_file = "matr_all_features.txt"
output_file = "matr_selected_features.txt"

with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

with open(output_file, 'w', encoding='utf-8') as f:
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) >= 61:  # 确保有足够的列（包括Cycle_Life）
            # 提取指定列：Battery_Name(0), F1(1), F5(5), F11(11), F14(14), F59(59), Cycle_Life(60)
            selected = [parts[0], parts[1], parts[5], parts[11], parts[14], parts[59], parts[60]]
            f.write('\t'.join(selected) + '\n')

print(f"已提取特征并保存到 {output_file}")