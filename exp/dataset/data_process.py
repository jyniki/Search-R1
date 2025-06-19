"""
Author: JiangYu
Email: 1067087283@qq.com
Date: 2025-05-30 09:25:27
FileName: data_process
Description: 数据处理函数，将CSV转换为parquet格式，支持随机划分train/test
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def convert_csv_to_parquet_with_split(
    csv_file_path, output_dir, test_size=0.2, random_state=42
):
    os.makedirs(output_dir, exist_ok=True)

    df_csv = pd.read_csv(csv_file_path)
    print(f"读取到 {len(df_csv)} 行数据")

    train_df, test_df = train_test_split(
        df_csv, test_size=test_size, random_state=random_state, shuffle=True
    )

    print(f"训练集: {len(train_df)} 行 ({len(train_df)/len(df_csv)*100:.1f}%)")
    print(f"测试集: {len(test_df)} 行 ({len(test_df)/len(df_csv)*100:.1f}%)")

    train_data_list = process_dataframe(train_df, split_name="train")
    train_parquet = pd.DataFrame(train_data_list)
    train_output_path = os.path.join(output_dir, "test_train.parquet")
    train_parquet.to_parquet(train_output_path, engine="pyarrow", index=False)
    print(f"训练集已保存到: {train_output_path}")

    test_data_list = process_dataframe(test_df, split_name="test")
    test_parquet = pd.DataFrame(test_data_list)
    test_output_path = os.path.join(output_dir, "test_test.parquet")
    test_parquet.to_parquet(test_output_path, engine="pyarrow", index=False)
    print(f"测试集已保存到: {test_output_path}")

    return train_parquet, test_parquet


def process_dataframe(df, split_name="train"):
    """
    处理DataFrame，转换为指定格式

    Args:
        df: 输入的DataFrame
        split_name: 数据集名称（train或test）
    """
    data_list = []

    for idx, row in df.iterrows():
        prompt_content = f"""回答给定的问题。\
每次获得新信息时，你必须先在<think>推理过程</think>之间进行推理。\
推理过程中的数据应该权威可靠，不要编造数据。\
推理后，如果你发现缺乏某些知识，你可以通过<search>中文query</search>调用搜索引擎，在<information>搜索结果</information>之间返回最相关的搜索结果。\
你可以根据需要搜索多次。\
如果你发现不需要更多外部知识，你可以直接在<answer>和</answer>之间提供答案，无需详细说明。问题：{row['问题']}\n"""

        prompt_array = np.array(
            [{"content": prompt_content, "role": "user"}], dtype=object
        )

        golden_answers = np.array([row["答案"]], dtype=object)

        reward_model = {
            "ground_truth": {"target": np.array([row["答案"]], dtype=object)},
            "style": "rule",
        }

        extra_info = {"index": idx, "split": split_name}

        data_row = {
            "id": f"{split_name}_{idx}",
            "question": row["问题"],
            "golden_answers": golden_answers,
            "data_source": "a800",
            "prompt": prompt_array,
            "ability": "fact-reasoning",
            "reward_model": reward_model,
            "extra_info": extra_info,
            "metadata": None,
        }

        data_list.append(data_row)

    return data_list


def convert_csv_to_parquet(csv_file_path, output_parquet_path):
    """
    保留原有的单文件转换功能（向后兼容）
    """
    df_csv = pd.read_csv(csv_file_path)
    data_list = process_dataframe(df_csv, split_name="train")
    df_parquet = pd.DataFrame(data_list)
    df_parquet.to_parquet(output_parquet_path, engine="pyarrow", index=False)
    print(f"成功转换 {len(df_csv)} 行数据")
    print(f"已保存到: {output_parquet_path}")
    return df_parquet


def main():
    csv_file = "exp/dataset/test_a800.csv"
    output_dir = "exp/dataset"

    try:
        train_df, test_df = convert_csv_to_parquet_with_split(
            csv_file_path=csv_file,
            output_dir=output_dir,
            test_size=0.2,
            random_state=42,
        )

        print("\n数据划分完成！")
        print(f"训练集大小: {len(train_df)}")
        print(f"测试集大小: {len(test_df)}")

    except Exception as e:
        print(f"转换过程中出现错误: {e}")


if __name__ == "__main__":
    main()
