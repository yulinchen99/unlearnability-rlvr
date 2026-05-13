"""
Extract a final boxed answer from each Gemini-generated solution. If the
solution does not contain a parseable \\boxed{...}, fall back to asking
gpt-5-nano (or whichever model is given via --openai-model) to extract it.

Reads {input_dir}/{mode}_data_{setting}_{data_type}_{idx}_with_gemini_answers.json
and writes alongside it as ..._with_extracted_gemini_answers.json.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_filter import last_boxed_only_string
from common.openai_utils import OpenaiClient
from tqdm import tqdm

EXTRACT_PROMPT = (
    "Please extract the answer for the problem from the following solution. "
    "Only extract the answer, do not include any other text. The answer should "
    "be a single number or a single expression. The answer should be placed "
    "within \\boxed{{}}.\n\n"
    "Problem: {problem}\n\nSolution: {gemini_answer}"
)


def extract_answers(data, problem_list_key, problem_text_key, model):
    for item in tqdm(data):
        for subproblem in item[problem_list_key]:
            answer = last_boxed_only_string(subproblem["gemini_answer"])
            if answer is None:
                response = model.query(
                    EXTRACT_PROMPT.format(
                        problem=subproblem[problem_text_key],
                        gemini_answer=subproblem["gemini_answer"],
                    ),
                    temperature=1.0,
                    max_tokens=50000,
                )
                answer = last_boxed_only_string(response)
            subproblem["extracted_gemini_answer"] = answer
    return data


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--setting", required=True,
                        help="Identifier matching the augment/decompose run, e.g. qwen_0.5b_math_level1to4.")
    parser.add_argument("--data-type", required=True,
                        help="learnable / unlearnable / etc.")
    parser.add_argument("--mode", required=True, choices=["augmented", "decomposed"])
    parser.add_argument("--idx", type=int, default=0,
                        help="Shard index produced by augment/decompose.")
    parser.add_argument("--input-dir", default=None,
                        help="Default: ./{mode}_data_gemini_answer")
    parser.add_argument("--openai-model", default="gpt-5-nano",
                        help="Fallback OpenAI model when no \\boxed{} is parseable.")
    args = parser.parse_args()

    if args.mode == "augmented":
        problem_list_key, problem_text_key = "augmented_problems", "problem"
    else:
        problem_list_key, problem_text_key = "subproblems", "subproblem"

    input_dir = args.input_dir or f"./{args.mode}_data_gemini_answer"
    file_path = os.path.join(
        input_dir,
        f"{args.mode}_data_{args.setting}_{args.data_type}_{args.idx}_with_gemini_answers.json",
    )
    with open(file_path) as f:
        data = json.load(f)

    model = OpenaiClient(model=args.openai_model)
    data = extract_answers(data, problem_list_key, problem_text_key, model)

    save_file = file_path.replace(
        "_with_gemini_answers.json", "_with_extracted_gemini_answers.json"
    )
    with open(save_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"saved to {save_file}")


if __name__ == "__main__":
    main()
