def has_high_char_repetition(text, threshold=0.1):
    """
    检查字符串中是否有大量重复字符。
    :param text: 输入字符串
    :param threshold: 重复字符占比阈值（默认0.1）
    :return: True（超过阈值） / False（正常）
    """
    if not text:
        return False
    # 统计每个字符的出现次数
    char_counts = {}
    for char in text:
        char_counts[char] = char_counts.get(char, 0) + 1
    
    # 计算最高频字符的占比
    max_count = max(char_counts.values())
    repetition_ratio = max_count / len(text)
    
    return repetition_ratio > threshold

def error_response(text):
    try:
        return has_high_char_repetition(text)
    except:
        return False

def contains_answer(text):
    try:
        return "</answer>" in text
    except:
        return False

