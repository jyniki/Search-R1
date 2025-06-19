import requests
from exp.settings import SEARCH_URL, SEARCH_ERROR_LOG


def search(query: str, topk: int = 3):
    payload = {
        "query": query,
        "search_params": {
            "translate": False,
            "milvus": {"name": ["ann_cninfo_csi800"], "num": 5, "expand_num": 0},
            "es": {"name": ["ann_cninfo_csi800"], "num": 5, "expand_num": 0},
        },
    }
    try:
        response = requests.post(
            SEARCH_URL,
            json=payload,
            headers={"Authorization": "ruitian_search_agent_123qweasd"},
        )
        result = response.json()["result"]
    except Exception as e:
        print(f"Error: {e}")
        with open(SEARCH_ERROR_LOG, "a") as f:
            f.write(f"查询失败: {query}\n")
        result = []

    return _passages2string(result[:topk])


def _passages2string(retrieval_result):
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["abs"]
        title = content.split("\n")[0]
        text = "".join(content.split("\n")[1:])
        format_reference += f"Doc {idx+1}:\nTitle: ({title})\n{text}\n"

    return format_reference


if __name__ == "__main__":
    print(search("五粮液2024年的业绩预期是多少"))
