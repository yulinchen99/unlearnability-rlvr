import re

def extract_last_boxed(text):
    """
    提取 LaTeX 文本中最后一个 \boxed 命令中的内容
    
    返回:
    - str: 最后一个 \boxed 中的内容。如果没有找到则返回 None
    """
    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
    
    # 找到所有匹配
    matches = list(re.finditer(pattern, text))
    
    # 如果找到匹配，返回最后一个的内容
    if matches:
        return matches[-1].group(0)
    return None



def compute_score(solution_str, ground_truth, method='strict'):
    solution_str = extract_last_boxed(solution_str).strip().lower()
    ground_truth = ground_truth.strip().lower()
    if method == 'strict':
        if solution_str == ground_truth:
            return {"score": 1.0, "correctness": True}
        else:
            return {"score": 0.0, "correctness": False}
    elif method == 'flexible':
        if solution_str in ground_truth:
            return {"score": 1.0, "correctness": True}
        else:
            return {"score": 0.0, "correctness": False}
    else:
        raise ValueError(f"Invalid method: {method}")