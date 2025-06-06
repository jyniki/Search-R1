"""
Author: JiangYu
Email: 1067087283@qq.com
Date: 2025-05-30 08:02:09
FileName: eval
Description:
"""

import pandas as pd
from tqdm import tqdm
from exp.think_rag import ThinkRAG
from verl.utils.reward_score.qa_em_llm import compute_score_em, extract_solution

model_id = "/rt-vepfs/public_model/Qwen/Qwen2.5-32B"
think_rag = ThinkRAG(model_id)

df = pd.read_csv("exp/dataset/test_a800.csv")

df["llm_answer"] = None
df["score"] = None

for index, row in tqdm(df.iterrows(), total=len(df), desc="Think RAG"):
    question = row["问题"]
    solution_str = think_rag.get_search_results(question)
    score = compute_score_em(solution_str, ground_truth={"target": row["答案"]})
    df.at[index, "llm_answer"] = solution_str
    df.at[index, "score"] = score

df.to_csv("exp/eval/eval_a800_think_rag_32b.csv", index=False)
