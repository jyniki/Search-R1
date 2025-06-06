"""
Author: JiangYu
Email: 1067087283@qq.com
Date: 2025-05-30 08:56:19
FileName: eval_pure_rag
Description:
"""

import requests
import pandas as pd
from tqdm import tqdm
from exp.milvus_search import search
from exp.settings import LLM_URL
from verl.utils.reward_score.qa_em_llm import compute_score_em, extract_solution

df = pd.read_csv("exp/dataset/test_a800.csv")
RAG_PROMPT = """
你是一个智能问答助手，请基于以下检索到的相关信息回答用户的问题。
如果检索到的信息不足以回答问题，请如实告知用户你不知道答案。
不要编造信息，只使用提供的上下文来回答。

用户问题: {query}

相关信息:
{ref}

请根据以上信息提供准确、有帮助的回答。
返回格式为: <answer>回答内容</answer>
"""


def get_llm_answer(prompt):
    payload = {
        "model": "gpt-4.1",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    response = requests.post(
        LLM_URL,
        json=payload,
        headers={"Authorization": "Basic cnQtdXNlcjoxa3NaUjkzWg=="},
    )
    return response.json()["choices"][0]["message"]["content"]


df["llm_answer"] = None
df["score"] = None

for index, row in tqdm(df.iterrows(), total=len(df), desc="Pure RAG"):
    question = row["问题"]
    search_results = search(question, topk=3)
    prompt = RAG_PROMPT.format(query=question, ref=search_results)
    solution_str = prompt + "\n" + get_llm_answer(prompt)
    llm_answer = extract_solution(solution_str=solution_str)
    score = compute_score_em(solution_str, ground_truth={"target": row["答案"]})
    df.at[index, "llm_answer"] = llm_answer
    df.at[index, "score"] = score

df.to_csv("exp/eval/eval_a800_pure_rag.csv", index=False)
