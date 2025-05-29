import requests


def search(query: str, topk: int = 3):
    SEARCH_URL = "http://106.75.245.178:7654/mix_search"

    payload = {
        "query": query,
        "topk": topk,
        "search_params": {
            "translate": False,
            "milvus": {"name": ["ann_cninfo_csi800"], "num": 5, "expand_num": 1},
            "es": {"name": ["ann_cninfo_csi800"], "num": 5, "expand_num": 1},
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
        result = []

    return _passages2string(result)


def _passages2string(retrieval_result):
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["expand_abs"]
        title = content.split("\n")[0]
        text = "".join(content.split("\n")[1:]).replace(f"{title}\n", "")
        format_reference += f"Doc {idx+1}:\nTitle: ({title})\n{text}\n"

    return format_reference


if __name__ == "__main__":
    print(search("寒武纪2024年的业绩预期是多少"))
