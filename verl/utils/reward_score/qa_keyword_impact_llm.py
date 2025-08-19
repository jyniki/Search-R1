
import json
import re
import random
import time
import datetime
import requests
from typing import Optional
from exp.settings import LLM_URL, LLM_API_KEY
call_cnt = 0

def keyword_impact_llm_score(question, keyword):
    if keyword == "" or keyword == "中文query" or len(keyword) == 1 or len(keyword) > 20:
        return 0.0

    max_retries = 3
    retry_delay = 1
    prompt = f"""
问题: {question}
关键词：{keyword}

请仔细判断关键词对问题回答的贡献度。判断标准如下：
1. 如果关键词对问题回答贡献非常大，返回1.0；
2. 如果关键词对问题回答有一定贡献，返回0.5；
3. 如果关键词对问题回答完全无贡献或没有实质信息，返回0.0（例如：只有“请搜索以下内容”，而没有搜索的内容，或者只有“请搜索以下内容”，而没有要搜索的内容）；
返回格式为json，格式如下：
{{
    "score": 0 or 0.5 or 1.0
}}
"""
    payload = {
        "model": "gpt-4.1",
        "messages": [
            {"role": "system", "content": "你是一个评分专家，请根据评分标准判断关键词对问题的贡献度。"},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                LLM_URL,
                json=payload,
                headers={"Authorization": f"Basic {LLM_API_KEY}"},
            )
            score = float(json.loads(response.json()["choices"][0]["message"]["content"])["score"])
            do_print = random.randint(1, 64) == 1
            if do_print:
                print(f"--------------------------------")
                print(f"Question: {question}")
                print(f"Keyword: {keyword}")
                print(f"Score: {score}")
            return score
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")

            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"All {max_retries} attempts failed. Returning score: 0.0")
    return 0.0
