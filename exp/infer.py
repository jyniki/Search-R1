"""
Author: JiangYu
Email: 1067087283@qq.com
Date: 2025-05-28 09:57:47
FileName: infer
Description:
"""

import transformers
import torch
import random
import requests
import re
from .milvus_search import search

question = "寒武纪2024年的业绩预期是多少"

model_id = "/rt-vepfs/public_model/Qwen/Qwen2.5-7B"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

question = question.strip()
curr_eos = [151645, 151643]  # for Qwen2.5 series models
curr_search_template = (
    "\n\n{output_text}<information>{search_results}</information>\n\n"
)

# Prepare the message
prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

# Initialize the tokenizer and model
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)


# Define the custom stopping criterion
class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_pattern, tokenizer):
        self.target_pattern = re.compile(target_pattern)
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Decode the current generated text
        generated_text = self._tokenizer.decode(input_ids[0], skip_special_tokens=True)

        # Check if the pattern matches
        if self.target_pattern.search(generated_text):
            return True

        return False


def get_query(text):
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None


# Initialize the stopping criteria - simplified to just detect </search>
stopping_criteria = transformers.StoppingCriteriaList(
    [StopOnSequence(r"</search>", tokenizer)]
)

cnt = 0

if tokenizer.chat_template:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )

print("\n\n################# [Start Reasoning + Searching] ##################\n\n")
print(prompt)
# Encode the chat-formatted prompt and move it to the correct device
while True:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)

    # Generate text with the stopping criteria
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1024,
        stopping_criteria=stopping_criteria,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
    )

    if outputs[0][-1].item() in curr_eos:
        generated_tokens = outputs[0][input_ids.shape[1] :]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(output_text)
        break

    generated_tokens = outputs[0][input_ids.shape[1] :]
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    search_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
    if search_query:
        print(f'searching "{search_query}"...')
        search_results = search(search_query)
    else:
        search_results = ""

    search_text = curr_search_template.format(
        output_text=output_text, search_results=search_results
    )
    print(search_text)
    prompt += search_text
    cnt += 1
