"""
Generate similar-problem augmentations for a list of seed problems via the
OpenAI API. For each seed problem, produces N similar problems (default 5)
that test the same skills with different surface details.

Inputs:
  --data-file        JSONL of seed problems. Each line must contain the
                     problem text under "question" and an integer/string id
                     under "id" or item["extra_info"]["index"].
  --prompt-ids-file  JSON list of seed-problem ids to augment. Use e.g. the
                     output of classification/finalize_unlearnable_prompt_id.py.

Outputs:
  {output_dir}/{output_name}_{idx}.json — list of {id, question, augmented_problems}.

Sharding: --idx N --total-batch K processes shard N of K equal-sized splits
(by sorted id) so multiple workers can be launched in parallel.

Requires OPENAI_API_KEY in the environment.
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.openai_utils import OpenaiClient


AUGMENT_PROMPT = """Generate {num_similar} new problems that test the same core skills and reasoning patterns as the problem below. The new problems should have similar difficulty and structure but use different contexts, numbers, or scenarios.

Original problem:
{problem}

Requirements:
- Each new problem must be solvable and well-defined
- The answer must be a mathematical expression or a number
- Vary the surface details (context, numbers, names) while keeping the underlying logic consistent
- Maintain similar complexity and difficulty level
- Ensure solutions are complete, correct, and show clear step-by-step reasoning
- Problems should help the model generalize the required skills

Output format (valid JSON only):
[
    {{
        "problem": "...",
        "solution": "...",
        "answer": "..."
    }},
    {{
        "problem": "...",
        "solution": "...",
        "answer": "..."
    }},
    ...
]

Important:
- Use latex to format the mathematical expressions whenever possible.
- The "solution" field should contain the step-by-step working.
- The "answer" field should contain only the final answer.
- Return only the JSON string, no additional text, explanations, or formatting markers.
"""


def generate_augmentations(sorted_problem_list, model, num_similar):
    batch_prompts = [
        AUGMENT_PROMPT.format(num_similar=num_similar, problem=problem)
        for _, problem in sorted_problem_list
    ]
    print(f"querying {len(batch_prompts)} prompts")

    responses, batch_cost = model.batch_query_threading(
        batch_prompts,
        temperature=1.0,
        max_tokens=50000,
        max_workers=8,
        show_progress=True,
    )
    print(f"batch cost: {batch_cost}")
    output_data = []
    failed_cases = []
    for item, response in zip(sorted_problem_list, responses):
        try:
            output_data.append({
                "id": item[0],
                "question": item[1],
                "augmented_problems": json.loads(response),
            })
        except Exception:
            print(f"Error parsing response for id {item[0]}")
            failed_cases.append(item)
    return output_data, failed_cases, batch_cost


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-file", required=True,
                        help="JSONL of seed problems.")
    parser.add_argument("--prompt-ids-file", required=True,
                        help="JSON list of seed-problem ids to augment.")
    parser.add_argument("--output-dir", default="./augmented_data",
                        help="Directory to write augmented JSONs to.")
    parser.add_argument("--output-name", required=True,
                        help="Base name for output files.")
    parser.add_argument("--idx", type=int, default=0,
                        help="Shard index (0-based).")
    parser.add_argument("--total-batch", type=int, default=1,
                        help="Total number of shards.")
    parser.add_argument("--openai-model", default="gpt-5",
                        help="OpenAI model name to use.")
    parser.add_argument("--num-similar", type=int, default=5,
                        help="Number of similar problems per seed.")
    parser.add_argument("--max-retry", type=int, default=5,
                        help="Max retries on JSON-parse failures per shard.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    save_file = os.path.join(args.output_dir, f"{args.output_name}_{args.idx}.json")

    target_ids = set(json.load(open(args.prompt_ids_file)))

    with open(args.data_file) as f:
        rows = [json.loads(line) for line in f]
    print(f"source problems: {len(rows)}")

    problem_list = {}
    for item in rows:
        item_id = item.get("id")
        if item_id is None:
            item_id = item["extra_info"]["index"]
        if item_id in target_ids:
            problem_list[item_id] = item["question"]
    print(f"selected problems: {len(problem_list)}")

    sorted_problems = sorted(problem_list.items(), key=lambda x: x[0])
    bs = math.ceil(len(sorted_problems) / args.total_batch)
    sorted_problems = sorted_problems[args.idx * bs : (args.idx + 1) * bs]
    print(f"shard {args.idx}/{args.total_batch} size: {len(sorted_problems)}")

    model = OpenaiClient(model=args.openai_model)

    all_output = []
    pending = sorted_problems
    for tried in range(args.max_retry):
        if not pending:
            break
        output, failed, cost = generate_augmentations(pending, model, args.num_similar)
        all_output.extend(output)
        pending = failed
        print(f"--- attempt {tried}: cost={cost}, succeeded={len(output)}, failed={len(failed)} ---")

    print(f"total augmented: {len(all_output)}")
    print(f"total cost: {model.total_cost}")
    with open(save_file, "w") as f:
        json.dump(all_output, f, indent=2, ensure_ascii=False)
    print(f"saved to {save_file}")


if __name__ == "__main__":
    main()
