curl -L -X POST "http://106.75.245.178:7654/mix_search" \
  -H "Content-Type: application/json" \
  -H "Authorization: ruitian_search_agent_123qweasd" \
  -d '{
    "query": "test",
    "topk": 3,
    "search_params": {
      "translate": false,
      "milvus": {
        "name": ["ann_cninfo_csi800"],
        "num": 5,
        "expand_num": 1
      },
      "es": {
        "name": ["ann_cninfo_csi800"],
        "num": 5,
        "expand_num": 1
      }
    }
  }'