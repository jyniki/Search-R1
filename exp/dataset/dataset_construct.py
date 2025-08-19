import aiohttp
import asyncio
import json
import csv
import sys
from tqdm.asyncio import tqdm_asyncio
import random
from verl.utils.reward_score.qa_em_llm import llm_score, extract_solution
from exp.settings import LLM_URL, LLM_API_KEY
class OpenaiAPI():
    def __init__(self, url, api_key, max_concurrent_requests=8):
        self.max_retries = 2
        self.delay = 2
        self.url = LLM_URL
        self.api_key = LLM_API_KEY
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Basic {self.api_key}"
        }
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def send_request(self, session, payload=None):
        async with self.semaphore:  # 使用信号量限制并发
            try:
                async with session.post(self.url, headers=self.headers, data=json.dumps(payload)) as response:
                    response.raise_for_status()
                    return await response.json()
            except aiohttp.ClientError as e:
                print(f"请求失败，错误信息：{e}")
                return None
    
    async def get_response(self, session, user_input, model='gpt-4.1', format="json_object"):
        if format:
            payload = {
                "model": model,
                "response_format": {"type": format},
                "messages": [
                    {"role": "system", "content": r""""""},
                    {"role": "user", "content": user_input}
                ]
            }
        else:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": r""""""},
                    {"role": "user", "content": user_input}
                ]
            }

        for i in range(self.max_retries):
            response = await self.send_request(session, payload)
            if response is not None:
                assistant_output = response['choices'][0]['message']['content']
                return assistant_output
            else:
                print(f"Error for input: {user_input}")
                print(f"等待{self.delay}秒后重试...")
                await asyncio.sleep(self.delay)
        return ''
    
    async def send(self, user_inputs, model='gpt-4.1', format='json_object'):
        async with aiohttp.ClientSession() as session:
            tasks = [self.get_response(session, user_input, model=model, format=format) for user_input in user_inputs]
            if len(tasks) >= 50:
                responses = await tqdm_asyncio.gather(*tasks)
            else:
                responses = await asyncio.gather(*tasks)
        return responses


async def get_model_res(user_inputs, model, format="json_object"):
    api = OpenaiAPI(url="http://106.75.245.178:4000/v1/chat/completions", api_key="sk-z6gZt-213cdrE1eCJnW6og")
    try:
        return await api.send(user_inputs, model=model, format=format)
    except:
        print("请输入正确的模型。")



def read_csv_with_headers(file_path: str):
    """
    读取包含特定header的CSV文件
    
    参数:
        file_path: CSV文件路径
        
    返回:
        包含字典的列表，每个字典代表一行数据，键为header名
    """
    data = []
    with open(file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        # 检查header是否符合预期
        for row in reader:
            data.append(dict(row))
    return data


def construct_prompts(qa_pairs):
    prompt_template = """我正在构建大模型微调数据集。每条数据包含是一个问答对和参考文本。请你帮我：
（1）根据参考文本检查答案的准确性，如果不能明确回答问题，或者问题不合适，请设置flag为False，并说明原因。否则设置为True。
（2）如果答案中包含与问题无关信息，请精简。
（3）以json格式返回调整后的问答对。

例子1：
输入：
{
	"问题": "招商蛇口、保利发展等企业被建议关注的理由是什么？",
	"答案": "建议关注招商蛇口、保利发展、中国金茂、中国海外发展、越秀地产、华发股份、滨江集团，因为这些房企具备片区综合开发能力，有望参与超大特大城市的城市更新，市占率有望提升，是稳健龙头房企。"
	"参考": "建议关注两条主线：1）看好具备片区综合开发能力，有机会参与超大特大城市的城市更新，市占率有望提升的稳健龙头房企，建议关注招商蛇口、保利发展、中国金茂、中国海外发展、越秀地产、华发股份、滨江集团。"
}
输出：
{
	"flag": True,
	"问题": "招商蛇口、保利发展等企业被建议关注的理由是什么？",
	"答案": "这些房企具备片区综合开发能力，有望参与超大特大城市的城市更新，市占率有望提升，是稳健龙头房企。"
}

例子2:
输入：
{
	"问题": "海大集团被列为投资建议的原因是什么？",
	"答案": "光大证券认为猪价反转将提振饲料、动保需求，估值修复板块将开启上行，建议关注海大集团。"，
	"参考": "（2）后周期板块，猪价反转将提振饲料、动保需求，估值修复板块将开启上行，建议关注海大集团。"
}
输出：
{
	"flag": False,
	"原因": "答案不能明确回答问题"
}

你的任务："""
    
    prompts = []
    for qa_pair in qa_pairs:
        prompt = prompt_template + f"""
输入：
    {{
        "问题": {qa_pair['问题']},
        "答案": {qa_pair['答案']},
        "参考": {qa_pair['参考']}
    }}
输出：
"""
        prompts.append(prompt)
    return prompts

def construct_prompts_for_sft_think(qa_pairs):
    prompt_template = """我将给你一个问答对，以及回答的参考资料，请生成中间的推理过程，并用json格式输出。
输出格式如下：
{
	"think": "推理过程"
}
"""
    prompts = []
    for qa_pair in qa_pairs:
        prompt = prompt_template + f"""
输入：
    {{
        "问题": {qa_pair['问题']},
        "答案": {qa_pair['答案']},
        "参考资料": {qa_pair['参考']}
    }}
输出：
"""
        prompts.append(prompt)
    return prompts

def construct_prompts_for_sft_search(qa_pairs):
    prompt_template = """我将给你一个问题，你需要调用搜索引擎来获得相关信息，以帮助问题的解答。你只需要输出搜索的关键词，用json格式输出。
	输出格式如下：
    {
        "query": "搜索关键词"
    }
"""
    prompts = []
    for qa_pair in qa_pairs:
        prompt = prompt_template + f"""
问题：{qa_pair['问题']}"""
        prompts.append(prompt)
    return prompts

    


def construct_prompts_for_gen_ref(qa_pairs):
    prompt_template = """我将给你一个问答对，请据此生成一些有助于回答问题的参考资料，用json格式输出。
例子：
输入：
{
	"问题"："2024年10月三美股份主要制冷剂品种的价格同比涨幅是多少？",
	"答案"："2024年10月，主要制冷剂品种价格同比涨幅如下：R32价格同比上涨139.4%，R125价格同比上涨41.5%，R134a价格同比上涨43.4%，R410a价格同比上涨78.6%，R22价格同比上涨64.1%。"
}
输出：
{
	"参考": "2024年10月，三美股份主要制冷剂品种的价格同比涨幅如下：

- 氟制冷剂：同比上涨28.17%
- 氟发泡剂：同比下降18.40%
- 氟化氢：同比上涨4.30%

其中，氟制冷剂的涨幅最为显著[1][11]。"
你的任务:
"""
    prompts = []
    for qa_pair in qa_pairs:
        prompt = prompt_template + f"""
输入：
输入：
    {{
        "问题": {qa_pair['问题']},
        "答案": {qa_pair['答案']}
    }}
输出：
"""
        prompts.append(prompt)
    return prompts

def construct_prompts_for_decompose_question(qa_pairs):
    prompt_template = """我将给你一个问答对，请判断其中包含几个问题，如果只包含一个，设置flag为True，如果包含多个问题，设置flag为False，请你将它们分解为多个问题，用json格式输出。
例子：
输入：
{
	"问题": "截至2024年第三季度末，大全能源的现金及资产负债率情况如何？",
	"答案": "截至2024年第三季度末，大全能源各类现金及现金等价物合计约为58.3亿元，还有约100亿元定期存款，资产负债率仅为11%。"
}
输出：
{
    "flag": False,
    "result": [
        {
            "问题": "截至2024年第三季度末，大全能源的现金情况如何？",
            "答案": "截至2024年第三季度末，大全能源各类现金及现金等价物合计约为58.3亿元。"
        },
        {
            "问题": "截至2024年第三季度末，大全能源的资产负债率情况如何？",
            "答案": "截至2024年第三季度末，大全能源资产负债率为11%。"
        }
    ]
}

你的任务：
"""
    prompts = []
    for qa_pair in qa_pairs:
        prompt = prompt_template + f"""
输入：
    {{
        "问题": {qa_pair['问题']},
        "答案": {qa_pair['答案']}
    }}
输出：
"""
        prompts.append(prompt)
    return prompts





async def main_rl():
    raw_content = read_csv_with_headers("/rt-vepfs/xjl/Search-R1/exp/dataset/a800_2.csv")
    user_inputs = construct_prompts(raw_content)

    resps = await get_model_res(user_inputs, "gpt-4.1")
    del_ques = []
    for content, resp in zip(raw_content, resps):
        resp = json.loads(resp)
        try:
            if resp['flag']:
                if content['问题'] == resp['问题']:
                    content['答案'] = resp['答案']
            else:
                del_ques.append(content['问题'])
            continue
        except:
            del_ques.append(content['问题'])
            continue
    raw_content = [content for content in raw_content if content['问题'] not in del_ques]
    
    with open("/rt-vepfs/xjl/Search-R1/exp/dataset/a800_2_processed.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['问题', '答案', '参考', 'us3路径'])
        writer.writeheader()
        writer.writerows(raw_content)


async def main_sft():
    raw_content = read_csv_with_headers("/rt-vepfs/xjl/Search-R1/exp/dataset/a800_2_processed.csv")
    user_inputs = construct_prompts_for_sft(raw_content)
    resps = await get_model_res(user_inputs, "gpt-4.1")
    results = []
    for content, resp in zip(raw_content, resps):
        try:
            content['think'] = json.loads(resp)['think']
            results.append({
                "instruction": "回答给定的问题。每次获得新信息时，你必须先在<think>推理过程</think>之间进行推理。推理过程中的数据应该权威可靠，不要编造数据。我将在<information>搜索结果</information>之间给出相关信息，你应该在<answer>和</answer>之间提供答案，无需详细说明。\n",
                "input": f"问题：{content['问题']}\n<information>{content['参考']}</information>",
                "output": f"<think>{content['think']}</think>\n<answer>{content['答案']}</answer>"
            })
        except:
            continue
    with open("/rt-vepfs/xjl/Search-R1/exp/dataset/a800_sft_processed2.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

async def main_sft_search():
    raw_content = read_csv_with_headers("/rt-vepfs/xjl/Search-R1/exp/dataset/a800_2_processed.csv")
    user_inputs = construct_prompts_for_sft_search(raw_content)
    resps = await get_model_res(user_inputs, "gpt-4.1")
    results = []
    for content, resp in zip(raw_content, resps):
        try:
            query = json.loads(resp)["query"]
            results.append({
                "instruction": """回答给定的问题。\
每次获得新信息时，你必须先在<think>推理过程</think>之间进行推理。\
推理过程中的数据应该权威可靠，不要编造数据。\
推理后，如果你发现缺乏某些知识，你可以通过<search>中文query</search>调用搜索引擎，在<information>搜索结果</information>之间返回最相关的搜索结果。\
你可以根据需要搜索多次。\
如果你发现不需要更多外部知识，你可以直接在<answer>和</answer>之间提供答案，无需详细说明。""",
                "input": f"问题：{content['问题']}",
                "output": f"<search>{query}</search>"
            })
        except:
            continue
    with open("/rt-vepfs/xjl/Search-R1/exp/dataset/a800_2_sft_processed_search.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

async def main_gen_ref():
    raw_content = read_csv_with_headers("/rt-vepfs/xjl/Search-R1/exp/dataset/test_a800.csv")
    user_inputs = construct_prompts_for_gen_ref(raw_content)
    resps = await get_model_res(user_inputs, "gpt-4.1")
    results = []
    for content, resp in zip(raw_content, resps):
        try:
            ref = json.loads(resp)["参考"]
            results.append({
                "问题": content['问题'],
                "答案": content['答案'],
                "参考": ref
            })
        except Exception as e:
            continue
    with open("/rt-vepfs/xjl/Search-R1/exp/dataset/a800_gen_ref_processed.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=['问题', '答案', '参考'])
        writer.writeheader()
        writer.writerows(results)


async def main_decompose_question():
    raw_content = read_csv_with_headers("/rt-vepfs/xjl/Search-R1/exp/dataset/a800_2_processed.csv")
    user_inputs = construct_prompts_for_decompose_question(raw_content)
    resps = await get_model_res(user_inputs, "gpt-4.1")
    results = []
    for content, resp in zip(raw_content, resps):
        try:
            res = json.loads(resp)
            if res['flag']:
                results.append(content)
            else:
                new_pairs = res['result']
                for new_pair in new_pairs:
                    results.append({
                        "问题": new_pair['问题'],
                        "答案": new_pair['答案'],
                        "参考": content['参考'],
                        "us3路径": content['us3路径']
                    })
        except:
            continue
    with open("/rt-vepfs/xjl/Search-R1/exp/dataset/a800_2_processed_decomposed.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


def concat_data():
    with open("/rt-vepfs/xjl/Search-R1/exp/dataset/a800_sft_processed1.json", "r") as f:
        data1 = json.load(f)
    with open("/rt-vepfs/xjl/Search-R1/exp/dataset/a800_sft_processed2.json", "r") as f:
        data2 = json.load(f)
    data = data1 + data2
    with open("/rt-vepfs/xjl/Search-R1/exp/dataset/a800_sft_processed.json", "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def shuffle_data():
    with open("/rt-vepfs/xjl/Search-R1/exp/dataset/a800_sft_processed.json", "r") as f:
        data = json.load(f)
    random.shuffle(data)
    train_data_think = data[:int(len(data) * 0.6)]
    train_data_search = data[int(len(data) * 0.6):]
    with open("/rt-vepfs/xjl/Search-R1/exp/dataset/a800_sft_processed_think.json", "w") as f:
        json.dump(train_data_think, f, ensure_ascii=False, indent=4)
    with open("/rt-vepfs/xjl/Search-R1/exp/dataset/a800_sft_processed_search.json", "w") as f:
        json.dump(train_data_search, f, ensure_ascii=False, indent=4)


def filter_wrong_data(file_path):
    filtered_data = []
    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['score'] == None or "1" not in row['score']:
                filtered_data.append({
                    "问题": row['问题'],
                    "答案": row['答案'],
                    "llm_answer": row['llm_answer'],
                    "score": row['score'],
                    "query": row['query'],
                    "info": row['info']
                })
    return filtered_data

def filter_right_data(file_path):
    filtered_data = []
    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "1" in row['score'] or "0.75" in row['score']:
                filtered_data.append({
                    "问题": row['问题'],
                    "答案": row['答案'],
                    "llm_answer": row['llm_answer'],
                    "score": row['score'],
                    "query": row['query'],
                    "info": row['info']
                })
    return filtered_data

def rescore(data):
    for row in data:
        if "0.75" in row['score']:
            score = llm_score(row['问题'], extract_solution(row['llm_answer']), row['答案'])
            row['score'] = score
    return data


def construct_sft_data(csv_path):
    search_data = []
    think_data = []
    instruction = """回答给定的问题。\
你必须先在<think>推理过程</think>之间进行推理。\
推理过程中的数据应该权威可靠，不要编造数据。\
推理后，如果你发现缺乏某些知识，你可以通过<search>中文query</search>调用搜索引擎，我将在<information>搜索结果</information>之间返回最相关的搜索结果。不要编造information中的内容。\
如果你发现不需要更多外部知识，你可以直接在<answer>和</answer>之间提供答案，无需详细说明。\
你的回答应该有且只有两种形式，要么是<think>推理过程</think><search>中文query</search>，要么是<think>推理过程</think><answer>答案</answer>"""
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            search_data.append({
                "instruction": instruction,
                "input": f"问题：{row['问题']}",
                "output": row['query']
            })
            think_data.append({
                "instruction": instruction,
                "input": f"问题：{row['问题']}\n<information>{row['info']}</information>",
                "output": row['llm_answer']
            })
    with open("/rt-vepfs/xjl/Search-R1/exp/dataset/a800_sft_search.json", "w") as f:
        json.dump(search_data, f, ensure_ascii=False, indent=4)
    with open("/rt-vepfs/xjl/Search-R1/exp/dataset/a800_sft_think.json", "w") as f:
        json.dump(think_data, f, ensure_ascii=False, indent=4)


def split_test():
    with open("/rt-vepfs/xjl/Search-R1/exp/dataset/a800_sft_search.json", "r") as f:
        data1 = json.load(f)
    with open("/rt-vepfs/xjl/Search-R1/exp/dataset/a800_sft_reason.json", "r") as f:
        data2 = json.load(f)
    random.shuffle(data1)
    random.shuffle(data2)
    test_data1 = data1[:int(len(data1) * 0.1)]
    train_data1 = data1[int(len(data1) * 0.1):]
    test_data2 = data2[:int(len(data2) * 0.1)]
    train_data2 = data2[int(len(data2) * 0.1):]
    test_data = test_data1 + test_data2
    with open("/rt-vepfs/xjl/Search-R1/exp/dataset/a800_sft_test.json", "w") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)
    with open("/rt-vepfs/xjl/Search-R1/exp/dataset/a800_sft_train_search.json", "w") as f:
        json.dump(train_data1, f, ensure_ascii=False, indent=4)
    with open("/rt-vepfs/xjl/Search-R1/exp/dataset/a800_sft_train_reason.json", "w") as f:
        json.dump(train_data2, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # construct_sft_data("/rt-vepfs/xjl/Search-R1/exp/dataset/sft_data.csv")  
    split_test()

    # data = filter_right_data("/rt-vepfs/xjl/Search-R1/exp/dataset/a800_2_processed_decomposed_generated.csv")
    # origin_data = filter_right_data("/rt-vepfs/xjl/Search-R1/exp/dataset/right_data_2.csv")
    # data = filter_right_data("/rt-vepfs/xjl/Search-R1/exp/dataset/a800_simple_gpt41_generated_filtered_wrong_output.csv")
    # data_wrong = filter_wrong_data("/rt-vepfs/xjl/Search-R1/exp/dataset/a800_simple_gpt41_generated_filtered_wrong_output.csv")
    # with open("/rt-vepfs/xjl/Search-R1/exp/dataset/right_data_3.csv", "w") as f:
    #     writer = csv.DictWriter(f, fieldnames=data[0].keys())
    #     writer.writeheader()
    #     writer.writerows(data+origin_data)
    # with open("/rt-vepfs/xjl/Search-R1/exp/dataset/wrong_data_2.csv", "w") as f:
    #     writer = csv.DictWriter(f, fieldnames=data_wrong[0].keys())
    #     writer.writeheader()
    #     writer.writerows(data_wrong)

    # asyncio.run(main_rl())
    # asyncio.run(main_sft())
    # asyncio.run(main_decompose_question())
    # concat_data()

    # asyncio.run(main_sft_search())

