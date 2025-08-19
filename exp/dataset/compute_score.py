import csv
import pandas as pd
import json
from json_repair import repair_json

def get_first_n_rows(csv_file, n):
    """读取CSV文件前n行完整数据
    
    Args:
        csv_file (str): CSV文件路径
        n (int): 要读取的行数
    
    Returns:
        list: 包含前n行数据的列表，每个元素是一个字典(列名:值)
    """
    rows = []
    
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            if i >= n:
                break
            rows.append(row)  # 保留原始数据
    
    return rows

def clean_csv_columns(input_file, output_file=None, additional_columns_to_drop=None):
    """
    清理CSV文件中的无意义列
    
    参数:
        input_file (str): 输入CSV文件路径
        output_file (str, optional): 输出CSV文件路径，默认为None(会修改原始文件)
        additional_columns_to_drop (list, optional): 需要额外删除的列名列表
        
    返回:
        pd.DataFrame: 清理后的DataFrame
    """
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    # 默认要删除的列
    default_columns_to_drop = [
        col for col in df.columns 
        if 'Unnamed' in str(col)  # 删除所有Unnamed列
    ]
    
    # 合并用户指定的额外列
    if additional_columns_to_drop:
        default_columns_to_drop.extend(additional_columns_to_drop)
    
    # 去重
    columns_to_drop = list(set(default_columns_to_drop))
    
    # 只删除实际存在的列
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    
    # 删除列
    df = df.drop(columns=existing_columns_to_drop, errors='ignore')
    
    # 保存或返回结果
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"清理完成，结果已保存到 {output_file}")
    else:
        print("清理完成，返回清理后的DataFrame")
    
    return df


def calculate_score_stats(csv_file):
    """
    计算CSV文件中score列的统计信息
    
    参数:
        csv_file (str): CSV文件路径
        
    返回:
        dict: 包含平均分和各分数段数量的字典
    """
    scores = []
    score_counts = {1: 0, 0.5: 0, 0.25: 0, 0.75: 0, 0: 0}  # 初始化分数计数器
    
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            try:
                score = float(row['score'])
                # score = repair_json(row['score'])
                
                # score = float(json.loads(score)['score'])
                # score = float(row['score'])
                scores.append(score)
                
                # 统计各分数段数量
                if score == 1:
                    score_counts[1] += 1
                elif score == 0.25:
                    score_counts[0.25] += 1
                elif score == 0.5:
                    score_counts[0.5] += 1
                elif score == 0.75:
                    score_counts[0.75] += 1
                elif score == 0:
                    score_counts[0] += 1
                    
            except (ValueError, KeyError):
                continue
    
    if not scores:
        print("警告: 没有找到有效的score数据")
        return None
    
    # 计算统计结果
    result = {
        '平均得分': sum(scores) / len(scores),
        '1': score_counts[1],
        '0.25': score_counts[0.25],
        '0.5': score_counts[0.5],
        '0.75': score_counts[0.75],
        '0': score_counts[0],
        '部分正确': score_counts[0.25] + score_counts[0.5],
        '正确': score_counts[0.75] + score_counts[1],
        '不正确': score_counts[0]
    }
    
    return result


# 使用示例
csv_file_path = '/rt-vepfs/xjl/Search-R1/exp/eval/eval_a800_think_rag_llama3-chinese-sft.csv'  # 替换为你的CSV文件路径
# print(get_first_n_rows(csv_file_path, 3))
scores = calculate_score_stats(csv_file_path)
from pprint import pprint
pprint(scores)

# if average_score is not None:
#     print(f"score的平均值是: {average_score:.2f}")

# clean_csv_columns("/rt-vepfs/xjl/Search-R1/exp/eval/eval_a800_think_rag_32b.csv", "/rt-vepfs/xjl/Search-R1/exp/eval/eval_a800_think_rag_32b-new.csv")