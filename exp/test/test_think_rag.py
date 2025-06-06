"""
Author: JiangYu
Email: 1067087283@qq.com
Date: 2025-06-06 05:44:16
FileName: test_think_rag
Description:
"""

from exp.think_rag import ThinkRAG

model_id = "/rt-vepfs/jy/project/Search-R1/outputs/verl_checkpoints/search-r1-grpo-qwen2.5-7b-em-a800/global_step_90/actor/huggingface"
think_rag = ThinkRAG(model_id)
think_rag.get_search_results("2022年，古井贡酒账上交易性金融资产是多少？")
