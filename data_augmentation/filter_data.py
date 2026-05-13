"""
Filter augmented / decomposed problems by checking that the model-supplied
answer matches Gemini's answer (extracted by extract_answer.py).

Globs {input_dir}/{mode}_data_{setting}_{data_type}_*_with_extracted_gemini_answers.json
and writes the survivors to {output_dir}/all_{mode}_data_{setting}_{data_type}_filtered.json.
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_filter import is_equivalent
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--setting", required=True,
                        help="Identifier matching the augment/decompose run, e.g. qwen_0.5b_math_level1to4.")
    parser.add_argument("--data-type", required=True)
    parser.add_argument("--mode", required=True, choices=["augmented", "decomposed"])
    parser.add_argument("--input-dir", default=None,
                        help="Default: ./{mode}_data_gemini_answer")
    parser.add_argument("--output-dir", default=None,
                        help="Default: ./{mode}_data")
    args = parser.parse_args()

    if args.mode == "augmented":
        problem_list_key, problem_text_key = "augmented_problems", "problem"
    else:
        problem_list_key, problem_text_key = "subproblems", "subproblem"

    input_dir = args.input_dir or f"./{args.mode}_data_gemini_answer"
    output_dir = args.output_dir or f"./{args.mode}_data"
    os.makedirs(output_dir, exist_ok=True)

    pattern = os.path.join(
        input_dir,
        f"{args.mode}_data_{args.setting}_{args.data_type}_*_with_extracted_gemini_answers.json",
    )
    files = glob.glob(pattern)

    total = 0
    all_filtered = []
    for path in files:
        with open(path) as f:
            data = json.load(f)
        for item in tqdm(data, desc=os.path.basename(path)):
            for sub in item[problem_list_key]:
                total += 1
                if is_equivalent(sub["answer"], sub["extracted_gemini_answer"]):
                    sub["question"] = sub.pop(problem_text_key)
                    sub.pop("gemini_answer", None)
                    sub.pop("extracted_gemini_answer", None)
                    all_filtered.append(sub)

    out_path = os.path.join(
        output_dir, f"all_{args.mode}_data_{args.setting}_{args.data_type}_filtered.json"
    )
    with open(out_path, "w") as f:
        json.dump(all_filtered, f, indent=2)
    print(f"kept {len(all_filtered)} / {total} ({len(all_filtered) / max(total, 1) * 100:.2f}%) -> {out_path}")


if __name__ == "__main__":
    main()
