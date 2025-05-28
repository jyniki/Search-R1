import requests

SEARCH_URL = "http://106.75.245.178:7654/mix_search"

payload = {
    "query": "test",
    "topk": 3,
    "search_params": {
        "translate": False,
        "milvus": {"name": ["ann_cninfo_csi800"], "num": 5, "expand_num": 1},
        "es": {"name": ["ann_cninfo_csi800"], "num": 5, "expand_num": 1},
    },
}

response = requests.post(
    SEARCH_URL,
    json=payload,
    headers={"Authorization": "ruitian_search_agent_123qweasd"},
)
print(response.json())
