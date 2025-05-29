"""
Author: JiangYu
Email: 1067087283@qq.com
Date: 2025-05-29 07:38:28
FileName: qa_em copy
Description:
"""

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re
import random
import time
from typing import Optional
from openai import OpenAI

deepseek_api_key = "sk-9b455fee69344119b6c759ea4fd8dc38"
client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")


def llm_score(prediction, golden_answer):
    prompt = f"""
标准答案：{golden_answer}
模型回答：{prediction}

请仔细判断模型回答是否正确。判断标准如下：
1. 如果答案包含数字，请注意：
    - 对于百分比或小数，只要数值在四舍五入后是一致的，就视为正确（例如：-10.4086% 和 -10.41% 应该被视为相同答案）
2. 如果单位不同，但数值转换后是一致的，视为正确
3. 考虑回答的完整性

如果模型回答正确，返回1.0；如果模型回答部分正确，返回0.5；如果模型回答错误，返回0.0。
返回格式为json，格式如下：
{{
    "score": 0 or 0.5 or 1
}}
"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False,
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)


def em_check(prediction, golden_answer):
    max_retries = 3
    retry_delay = 1
    start_time = time.time()
    for attempt in range(max_retries):
        try:
            score = float(llm_score(prediction, golden_answer)["score"])
            end_time = time.time()
            print(f"LLM Server Time: {end_time - start_time} s")
            return score
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")

            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"All {max_retries} attempts failed. Returning score: 0.0")

    return 0.0


def extract_solution(solution_str) -> Optional[str]:
    """Extract the equation from the solution string."""
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    if len(matches) <= 1:
        return None

    return matches[-1].group(1).strip()


def compute_score_em(
    solution_str, ground_truth, method="strict", format_score=0.0, score=1.0
):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        score = 0.0
    else:
        score = em_check(answer, ground_truth["target"])
    return score
