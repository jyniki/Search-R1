#!/usr/bin/env python3
import sys
import pandas as pd

data_path = "/rt-vepfs/jy/dataset/nq_data/nq_hotpotqa/train.parquet"
df = pd.read_parquet(data_path)
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
print("\nFirst row:")
print(df.iloc[0])
print("\nFirst prompt:")
print(df.iloc[0]["prompt"])
print("\nSecond prompt:")
print(df.iloc[1]["prompt"])
