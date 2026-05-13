import pandas as pd
import numpy as np
from collections import defaultdict
import json
import re

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"
ANSWER_TRIGGER = "The answer is"

def calculate_pass_at_k(n_samples: int, problem_results, avg=True) -> float:
    """
    Calculate probability of solving a problem with k attempts.
    
    Args:
        n_samples: Total number of samples
        n_correct: Number of correct samples
        k: Number of attempts allowed
    
    Returns:
        Probability of solving the problem within k attempts
    """
    k_values = [1]
    power = 1
    while 2**power <= n_samples:  # Add powers of 2 up to max_len
        k_values.append(2**power)
        power += 1


    def estimate_pass_at_k(n_samples: int, problem_results, k: int) -> float:
        if n_samples < k:
            raise ValueError(f"n_samples ({n_samples}) must be greater than or equal to k ({k})")
        assert len(problem_results) == n_samples
        n_correct = sum(problem_results)
        
        # Calculate probability using combination formula
        n = n_samples
        c = n_correct
        if c == 0:
            return 0.0
        
        # Calculate 1 - P(all k samples are wrong) and the std of the probability
        pass_at_k = 1.0 - np.prod([(n - c - i)/(n - i) for i in range(k)])
        return pass_at_k
    
    qid2score = {}
    for q_id, results in problem_results.items():
        for k in k_values:
            score = estimate_pass_at_k(n_samples, results, k)
            if q_id not in qid2score:
                qid2score[q_id] = {}
            qid2score[q_id][f"pass@{k}"] = score
            
    if avg:
        for k in k_values:
            avg = np.mean([qid2score[q_id][f"pass@{k}"] for q_id in qid2score])
            std = np.std([qid2score[q_id][f"pass@{k}"] for q_id in qid2score])
            return {f"pass@{k}": avg, f"pass@{k}-std": std}
    return qid2score

def convert_to_df(qid2score):
    """
    convert {
        "qid1": {
            "pass@1": 0.5,
            "pass@2": 0.6,
            "pass@4": 0.7,
        },
        "qid2": {
            "pass@1": 0.6,
            "pass@2": 0.7,
            "pass@4": 0.8,
        },
    }
    to a dataframe with columns: qid, pass@1, pass@2, ..., pass@k
    """
    df_dict = {"q_id": [], "score": [], "pass_k": []}
    for q_id, scores in qid2score.items():
        for k, score in scores.items():
            # insert new row
            df_dict["q_id"].append(q_id)
            df_dict["score"].append(score)
            df_dict["pass_k"].append(k.split("@")[1])
    df = pd.DataFrame(df_dict)
    return df

def load_results(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    id2correct = defaultdict(list)
    for item in data:
        id2correct[item["question_id"]].append(int(item["verification"]["is_correct"]))
    return id2correct

def evaluate_mc(completions=None, answers=None):
    """
    Evaluate multiple-choice completions.
    Args:
        completions (List[str]): List of completions.
        answers (List[str]): List of answers.
    Returns:
        float: Accuracy.
    """

    # extract answer from the completion
    patterns = [r'answer is \((.)\)', r'Answer: \((.)\)', r'answer: \((.)\)', r'answer \((.)\)', r'\((.)\)']
    pred_list = []
    format_error = 0
    for pred in completions:
        matched_pred = last_boxed_only_string(pred)

        if matched_pred is None:
            last_match = None
            last_index = -1  # Track the last occurrence index

            for pattern in patterns:
                for match in re.finditer(pattern, pred):  # Iterate over all matches for the pattern
                    if match and match.group(1):
                        start_index = match.start()  # Get the starting position of the match
                        if start_index > last_index:  # Update if it's the latest match
                            last_index = start_index
                            last_match = match.group(1)

            if last_match:
                matched_pred = last_match
            else:
                format_error += 1  # No match found

        if matched_pred:
            letter_match = re.search(r'(?i)\b[A-D]\b', matched_pred)
            if letter_match:
                matched_pred = letter_match.group(0).upper()
        
        pred_list.append(matched_pred)
    
    assert len(pred_list) == len(answers)
    
    correct = 0
    total = 0
    eval_results = []
    for pred, ans in zip(pred_list, answers):
        if pred == ans:
            correct += 1
        total += 1
        eval_results.append(int(pred == ans))
    return eval_results, pred_list, {
        "accuracy": correct / len(pred_list),
        "correct_num": correct,
        "total": len(pred_list),
        "format_error": format_error
    }


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    left_brace_idx = None
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
            if left_brace_idx is None:
                left_brace_idx = i
        elif string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break

        i += 1
    
    if left_brace_idx is None or right_brace_idx is None:
        return None

    return string[left_brace_idx + 1: right_brace_idx].strip()