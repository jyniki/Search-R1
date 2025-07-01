"""
Author: JiangYu
Email: 1067087283@qq.com
Date: 2025-05-30 08:02:09
FileName: eval
Description:
"""

import pandas as pd
from tqdm import tqdm
from exp.think_rag import ThinkRAG
from verl.utils.reward_score.qa_em_llm import compute_score_em, extract_solution
import argparse


# model_id = "/rt-vepfs/xjl/backup0624/search-r1-grpo-qwen2.5-32b-em-a800-202506200033/search-r1-grpo-qwen2.5-32b-em-a800-202506200033/global_step_320/actor/huggingface"
# think_rag = ThinkRAG(model_id)

# df = pd.read_csv("exp/dataset/a800_test_0625.csv")

# df["llm_answer"] = None
# df["score"] = None

# print(model_id)

# for index, row in tqdm(df.iterrows(), total=len(df), desc="Think RAG"):
#     question = row["问题"]
#     solution_str = think_rag.get_search_results(question)
#     score = compute_score_em(solution_str, ground_truth={"target": row["答案"]})
#     df.at[index, "llm_answer"] = solution_str
#     df.at[index, "score"] = score

# df.to_csv("exp/eval/eval_a800_think_rag_32b_320-steps.csv", index=False)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Evaluate ThinkRAG model on a test dataset")
    parser.add_argument("--model_id", type=str, required=True,
                       help="Path to the model directory")
    parser.add_argument("--input_csv", type=str, required=True,
                       help="Path to the input CSV file")
    parser.add_argument("--output_csv", type=str, required=True,
                       help="Path to save the output CSV file")
    parser.add_argument("--question_col", type=str, default="问题",
                       help="Column name for questions in the input CSV")
    parser.add_argument("--answer_col", type=str, default="答案",
                       help="Column name for answers in the input CSV")
    
    args = parser.parse_args()

    # Initialize ThinkRAG
    think_rag = ThinkRAG(args.model_id)

    # Load data
    df = pd.read_csv(args.input_csv)

    # Initialize result columns
    df["llm_answer"] = None
    df["score"] = None

    print(f"Using model: {args.model_id}")

    # Process each row
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Think RAG"):
        question = row[args.question_col]
        solution_str = think_rag.get_search_results(question)
        score = compute_score_em(solution_str, ground_truth={"target": row[args.answer_col]})
        df.at[index, "llm_answer"] = solution_str
        df.at[index, "score"] = score

    # Save results
    df.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")

if __name__ == "__main__":
    main()