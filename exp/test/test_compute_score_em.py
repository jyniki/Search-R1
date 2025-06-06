#!/usr/bin/env python3
"""
测试 compute_score_em 函数的脚本
"""


from verl.utils.reward_score.qa_em_llm import compute_score_em


def test_compute_score_em():
    """测试 compute_score_em 函数"""
    print("=== 测试 compute_score_em 函数 ===")

    # print("测试1: 没有答案标签")
    # solution1 = "这是一个没有答案标签的解答"
    # ground_truth1 = {"target": "42"}
    # score1 = compute_score_em(solution1, ground_truth1)
    # print(f"ground_truth: {ground_truth1}\nanswer: {solution1}")
    # print(f"结果: {score1}")

    # print("\n测试2: 只有一个答案标签")
    # solution2 = "<answer></answer>任务的回答是<answer>42</answer>"
    # ground_truth2 = {"target": "42"}
    # score2 = compute_score_em(solution2, ground_truth2)
    # print(f"ground_truth: {ground_truth2}\nanswer: {solution2}")
    # print(f"结果: {score2}")

    # print("\n测试3: 多个答案标签，答案错误")
    # solution3 = "<answer></answer>任务的回答是<answer>32</answer>"
    # ground_truth3 = {"target": "42"}
    # score3 = compute_score_em(solution3, ground_truth3, format_score=0.1)
    # print(f"ground_truth: {ground_truth3}\nanswer: {solution3}")
    # print(f"结果: {score3}")

    # print("\n测试4: 数值相近的答案")
    # solution4 = "<answer></answer>任务的回答是<answer>-10.4086%</answer>"
    # ground_truth4 = {"target": "-10.41%"}
    # score4 = compute_score_em(solution4, ground_truth4)
    # print(f"ground_truth: {ground_truth4}\nanswer: {solution4}")
    # print(f"结果: {score4}")

    print("\n测试5: 部分正确")
    solution5 = "<answer></answer>任务的回答是<answer>-10.4086%和0.32</answer>"
    ground_truth5 = {"target": "-10.41%和0.2"}
    score5 = compute_score_em(solution5, ground_truth5)
    print(f"ground_truth: {ground_truth5}\nanswer: {solution5}")
    print(f"结果: {score5}")

    print("compute_score_em 测试完成！")


if __name__ == "__main__":
    print("开始测试 compute_score_em 函数\n")
    test_compute_score_em()
