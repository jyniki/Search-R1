"""
Author: JiangYu
Email: 1067087283@qq.com
Date: 2025-05-28 09:57:47
FileName: infer
Description:
"""

import transformers
import torch
import re
from exp.milvus_search import search


class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        # Encode the string so we have the exact token-IDs pattern
        self.target_ids = [
            tokenizer.encode(target_sequence, add_special_tokens=False)
            for target_sequence in target_sequences
        ]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Make sure the target IDs are on the same device
        targets = [
            torch.as_tensor(target_id, device=input_ids.device)
            for target_id in self.target_ids
        ]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        # Compare the tail of input_ids with our target_ids
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i] :], target):
                return True

        return False


class ThinkRAG:
    def __init__(self, model_id):
        self.prompt_template = """回答给定的问题。\
每次获得新信息时，你必须先在<think>推理过程</think>之间进行推理。\
推理过程中的数据应该权威可靠，不要编造数据。\
推理后，如果你发现缺乏某些知识，你可以通过<search>中文query</search>调用搜索引擎，在<information>搜索结果</information>之间返回最相关的搜索结果。\
你可以根据需要搜索多次。\
如果你发现不需要更多外部知识，你可以直接在<answer>和</answer>之间提供答案，无需详细说明。问题：{question}\n"""

        self.curr_eos = [151645, 151643]  # for Qwen2.5 series models
        self.curr_search_template = (
            "\n\n{output_text}<information>{search_results}</information>\n\n"
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.stopping_criteria = self.get_stopping_criteria()

    def get_stopping_criteria(self):
        target_sequences = [
            "</search>",
            " </search>",
            "</search>\n",
            " </search>\n",
            "</search>\n\n",
            " </search>\n\n",
        ]
        stopping_criteria = transformers.StoppingCriteriaList(
            [StopOnSequence(target_sequences, self.tokenizer)]
        )
        return stopping_criteria

    @staticmethod
    def get_query(text):
        pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
        matches = pattern.findall(text)
        if matches:
            return matches[-1]
        else:
            return None

    def get_search_results(self, question, max_turn=3):
        question = question.strip()
        cnt = 0

        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": self.prompt_template.format(question=question),
                    }
                ],
                add_generation_prompt=True,
                tokenize=False,
            )

        print(
            "\n\n################# [Start Reasoning + Searching] ##################\n\n"
        )
        print(prompt)
        # Encode the chat-formatted prompt and move it to the correct device
        while cnt < max_turn:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(
                self.device
            )
            attention_mask = torch.ones_like(input_ids)

            # Generate text with the stopping criteria
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1024,
                stopping_criteria=self.stopping_criteria,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
            )
            generated_tokens = outputs[0][input_ids.shape[1] :]
            output_text = self.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )
            print(output_text)
            if outputs[0][-1].item() in self.curr_eos:
                return prompt + output_text

            search_query = self.get_query(
                self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            )
            if search_query:
                print(f'searching "{search_query}"...')
                search_results = search(search_query)
            else:
                search_results = ""

            search_text = self.curr_search_template.format(
                output_text=output_text, search_results=search_results
            )
            print(search_text)
            prompt += search_text
            cnt += 1

        return prompt
