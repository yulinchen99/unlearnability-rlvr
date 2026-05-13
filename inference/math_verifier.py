from typing import List, Dict, Any
import json
import os
import sys
# add the repo root (parent of inference/) to the python path so common.* resolves
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from math_verify import parse, verify


def last_boxed_only_string(string: str):
    """Return the contents of the last \\boxed{...} (or \\fbox{...}) in `string`,
    or None if no balanced boxed expression is found."""
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


def clean_answer(model_pred: str) -> str:
    """Extract the contents of the last \\boxed{} in the model output."""
    match_str = last_boxed_only_string(model_pred)
    if match_str is not None:
        return match_str
    return "[invalid]"


def _verify_with_math_verify(gold: str, target: str) -> bool:
    if "\\boxed" not in target:
        target = f"\\boxed{{{target}}}"
    if "\\boxed" not in gold:
        gold = f"\\boxed{{{gold}}}"
    try:
        parsed_gold = parse(gold)
        parsed_target = parse(target)
        return bool(verify(gold=parsed_gold, target=parsed_target))
    except Exception:
        return False


def evaluate(filepath=None, answers=None, completions=None, output_file=None):
    eval_results = []
    model_answers = []
    for answer, model_completion in zip(answers, completions):
        model_answer = clean_answer(model_completion)
        is_cor = _verify_with_math_verify(answer, model_completion)
        eval_results.append(is_cor)
        model_answers.append(model_answer)
    total = len(eval_results)
    correct_num = sum(eval_results)
    accuracy = float(correct_num) / total if total else 0.0
    return eval_results, model_answers, {"total": total, "correct_num": correct_num, "accuracy": accuracy}


class MathResponseVerifier:
    # default evaluate function is the math_verify-based evaluate above
    # can pass in evaluate_mc from utils
    def __init__(self, evaluate_func=evaluate):
        """Initialize the math response verifier."""
        self.evaluate_func = evaluate_func

    def verify_responses(self, responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Verify the quality of math responses using rule-based verification."""
        verified_responses = []

        # Group responses by question for batch evaluation
        question_groups = {}
        for response in responses:
            qid = response["question_id"]
            if qid not in question_groups:
                question_groups[qid] = {
                    "question": response["question"],
                    "ground_truth": response.get("ground_truth", None),
                    "responses": [],
                    "cum_logprobs": [],
                    "total_length": []
                }
            question_groups[qid]["responses"].append(response["response"])
            if "cum_logprobs" in response:
                question_groups[qid]["cum_logprobs"].append(response["cum_logprobs"])
            if "total_length" in response:
                question_groups[qid]["total_length"].append(response["total_length"])

        # Process each question group
        for qid, group in question_groups.items():
            ground_truth = group["ground_truth"]
            if not ground_truth:
                print(f"Warning: No ground truth provided for question {qid}")
                continue

            # Evaluate all responses for this question
            eval_results, model_answers, stats = self.evaluate_func(
                answers=[ground_truth] * len(group["responses"]),
                completions=group["responses"]
            )

            # Add verification data to each response
            for idx, (response, is_correct, model_answer) in enumerate(zip(group["responses"], eval_results, model_answers)):
                response_data = {
                    "question_id": qid,
                    "question": group["question"],
                    "response": response,
                    "verification": {
                        "score": 1.0 if is_correct else 0.0,
                        "extracted_answer": model_answer,
                        "is_correct": is_correct,
                        "is_accepted": is_correct
                    }
                }
                if group["cum_logprobs"]:
                    response_data["cum_logprobs"] = group["cum_logprobs"][idx]
                if group["total_length"]:
                    response_data["total_length"] = group["total_length"][idx]
                verified_responses.append(response_data)

        return verified_responses

    def save_verified_responses(self, responses: List[Dict[str, Any]], output_file: str):
        """Save verified responses to a file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(responses, f, indent=2)

    def filter_responses(self, responses: List[Dict[str, Any]],
                        min_score: float = 1.0) -> List[Dict[str, Any]]:
        """Filter responses based on verification scores.
        For math problems, we typically want only correct answers (score = 1.0)."""
        return [r for r in responses if r["verification"]["score"] >= min_score]


if __name__ == "__main__":
    # smoke test
    responses = [
        {"question_id": "1", "question": "2+2?", "response": "The answer is \\boxed{4}.", "ground_truth": "4"},
        {"question_id": "1", "question": "2+2?", "response": "I get \\boxed{5}.",         "ground_truth": "4"},
    ]
    v = MathResponseVerifier()
    print(v.verify_responses(responses))
