import requests
import os
import re
import json
from exp.milvus_search import search
from exp.settings import LLM_URL, LLM_API_KEY

LOG_FILE = "/rt-vepfs/xjl/Search-R1/exp/eval/log.txt"



def extract_query(solution_str):
    """Extract the equation from the solution string."""
    query_pattern = r"<search>(.*?)</search>"
    match = re.finditer(query_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    if len(matches) < 1:
        return None

    return "<search>" + matches[-1].group(1).strip() + "</search>"


class ThinkRAGOpenAI:
    def __init__(self, model_name="gpt-4.1"):
        self.instruction = """回答给定的问题。\
你必须先在<think>推理过程</think>之间进行推理。\
推理过程中的数据应该权威可靠，不要编造数据。\
推理后，如果你发现缺乏某些知识，你可以通过<search>中文query</search>调用搜索引擎，我将在<information>搜索结果</information>之间返回最相关的搜索结果。不要编造information中的内容。\
如果你发现不需要更多外部知识，你可以直接在<answer>和</answer>之间提供答案，无需详细说明。\
你的回答应该有且只有两种形式，要么是<think>推理过程</think><search>中文query</search>，要么是<think>推理过程</think><answer>答案</answer>"""
        self.curr_search_template = ("<information>{search_results}</information>")
        self.model_name = model_name
        self.api_key = "sk-Tk5jgkqVP7LN0TF8z5KxWw"
        self.api_url = LLM_URL

    @staticmethod
    def get_query(text):
        pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
        matches = pattern.findall(text)
        if matches:
            return matches[-1]
        else:
            return None

    def get_search_results(self, question, max_turn=2):
        question = question.strip()
        cnt = 0
        history = [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": "问题：" + question}
        ]
        print("\n\n################# [Start Reasoning + Searching] ##################\n\n")
        print("prompt: " + self.instruction + question + "\n")
        with open(LOG_FILE, "a") as f:
            f.write("\n\n################# [Start Reasoning + Searching] ##################\n\n")
            f.write("prompt: " + self.instruction + question + "\n")
        while cnt < max_turn:
            payload = {
                "model": self.model_name,
                "messages": history
            }
            headers = {
                "Authorization": f"Basic {self.api_key}"
            }
            response = requests.post(self.api_url, json=payload, headers=headers)
            if response.status_code != 200:
                raise RuntimeError(f"OpenAI API 请求失败: {response.status_code}, {response.text}")
            data = response.json()
            output_text = data["choices"][0]["message"]["content"]
            reason_text = data["choices"][0]["message"]["reasoning_content"]

            print("output: " + output_text)
            with open(LOG_FILE, "a") as f:
                f.write("output: " + output_text + "\n")
            # 判断是否已经给出最终答案
            if "<answer>" in output_text and "</answer>" in output_text:
                history.append({"role": "assistant", "content": f"<think>{reason_text}</think>{output_text}"})
                return json.dumps(history)
                # return "\n".join([msg["content"] for msg in history if msg["role"]=="user"]) + "\n" + output_text
            search_query = self.get_query(output_text)
            if search_query:
                print(f'searching "{search_query}"..."')
                with open(LOG_FILE, "a") as f:
                    f.write(f'searching "{search_query}"...\n')
                search_results = search(search_query)
            else: 
                search_results = ""
            search_text = self.curr_search_template.format(
                search_results=search_results
            )
            print(search_text)
            with open(LOG_FILE, "a") as f:
                f.write(search_text + "\n")
            # 将搜索结果加入对话历史
            history.append({"role": "assistant", "content": f"<think>{reason_text}</think>{extract_query(output_text)}"})
            history.append({"role": "user", "content": search_text})
            
            cnt += 1
        
        # history.append({"role": "assistant", "content": output_text})
        
        return json.dumps(history)
        # return "\n".join([msg["content"] for msg in history if msg["role"]=="user"]) 