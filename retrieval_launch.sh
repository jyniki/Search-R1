
index_file=/rt-vepfs/jy/dataset/wiki-18-e5-index/e5_Flat.index
corpus_file=/rt-vepfs/jy/dataset/wiki-18-corpus/wiki-18.jsonl
retriever_name=e5
retriever_path=/rt-vepfs/jy/model/embedding/e5-base-v2

python3 search_r1/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --faiss_gpu
