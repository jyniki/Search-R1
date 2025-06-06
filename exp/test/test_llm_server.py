"""
Author: JiangYu
Email: 1067087283@qq.com
Date: 2025-05-30 08:07:42
FileName: test_llm_server
Description:
"""

LLM_URL = "http://106.75.245.178:8600/v1/chat/completions"

import requests

payload = {
    "model": "gpt-4.1",
    "messages": [{"role": "user", "content": "你好"}],
}
try:
    response = requests.post(
        LLM_URL,
        json=payload,
        headers={"Authorization": "Basic cnQtdXNlcjoxa3NaUjkzWg=="},
    )
    print(response.json())
except Exception as e:
    print(f"Error: {e}")
