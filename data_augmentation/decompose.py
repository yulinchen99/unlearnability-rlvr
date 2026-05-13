"""
Decompose seed problems into independent subproblems via the OpenAI API.
For each seed problem, produces a sequence of subproblems whose solutions
collectively solve the original.

Inputs:
  --data-file        JSONL of seed problems. Each line must contain the
                     problem text under "question" and an integer/string id
                     under "id" or item["extra_info"]["index"].
  --prompt-ids-file  JSON list of seed-problem ids to decompose. Use e.g.
                     the output of
                     classification/finalize_unlearnable_prompt_id.py.

Outputs:
  {output_dir}/{output_name}_{idx}.json — list of {id, question, subproblems}.

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


DECOMPOSE_PROMPT = """
Task: Decompose a mathematical problem into independent subproblems whose solutions collectively solve the original problem.

Requirements for subproblems:
1. Independence: Each subproblem must be fully self-contained, including all necessary context and definitions. A reader should be able to solve any single subproblem without seeing the others.
2. Clarity: Each subproblem must be unambiguous and have a unique, well-defined answer.
3. Progression: Subproblems should follow a logical order, building toward the final solution.

Requirements for solutions and answers:
1. Show complete step-by-step reasoning.
2. Use LaTeX formatting for all mathematical expressions (e.g., $n$, $\\frac{{a}}{{b}}$, $\\mod$).
3. Ensure calculations are correct and verifiable.
4. The answer should be a numerical value or a single mathematical expression.

Output format:
Return ONLY a valid JSON array with no additional text, markdown code fences, or explanations before or after.

Structure:
[
    {{
        "subproblem": "<Clear, self-contained problem statement>",
        "solution": "<Step-by-step working with LaTeX formatting>",
        "answer": "<Final numerical or mathematical answer only>"
    }},
    ...
]

Example of an original problem and one of its well-formed subproblems:
Original Problem:
There are $7$ boxes arranged in a row and numbered $1$ through $7$. You have a stack of $2015$ cards, which you place one by one in the boxes. The first card is placed in box $1$, the second in box $2$, and so forth up to the seventh card which is placed in box $7$. You then start working back in the other direction, placing the eighth card in box $6$, the ninth in box $5$, up to the thirteenth card being placed in box $1$. The fourteenth card is then placed in box $2$, and this continues until every card is distributed. What box will the last card be placed in?

Subproblem:
{{
    "subproblem": "In the card distribution pattern described, cards are placed in boxes following the sequence 1,2,3,4,5,6,7,6,5,4,3,2,1,2,3,... (bouncing between box 1 and box 7). How many cards are placed in one complete cycle, where a cycle starts at box 1, goes to box 7, and returns to box 1 (not including the return to box 1)?",
    "solution": "A complete cycle goes: $1 \\to 2 \\to 3 \\to 4 \\to 5 \\to 6 \\to 7 \\to 6 \\to 5 \\to 4 \\to 3 \\to 2$, which is $12$ placements. The next card at box $1$ begins a new cycle.",
    "answer": "12"
}}

IMPORTANT:
- Do not include the original problem statement in your response.
- Each subproblem must be fully self-contained, including all necessary context and definitions. A reader should be able to solve any single subproblem without seeing the others.
- Verify that all arithmetic and modular calculations are correct.

Please decompose the following problem into subproblems:
{problem}
"""


def generate_decompositions(sorted_problem_list, model):
    batch_prompts = [
        DECOMPOSE_PROMPT.format(problem=problem)
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
                "subproblems": json.loads(response),
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
                        help="JSON list of seed-problem ids to decompose.")
    parser.add_argument("--output-dir", default="./decomposed_data",
                        help="Directory to write decomposed JSONs to.")
    parser.add_argument("--output-name", required=True,
                        help="Base name for output files.")
    parser.add_argument("--idx", type=int, default=0,
                        help="Shard index (0-based).")
    parser.add_argument("--total-batch", type=int, default=1,
                        help="Total number of shards.")
    parser.add_argument("--openai-model", default="gpt-5",
                        help="OpenAI model name to use.")
    parser.add_argument("--max-retry", type=int, default=5,
                        help="Max retries on JSON-parse failures per shard.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    save_file = os.path.join(args.output_dir, f"{args.output_name}_{args.idx}.json")
    if os.path.exists(save_file):
        print(f"{save_file} already exists, skipping")
        return

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
        output, failed, cost = generate_decompositions(pending, model)
        all_output.extend(output)
        pending = failed
        print(f"--- attempt {tried}: cost={cost}, succeeded={len(output)}, failed={len(failed)} ---")

    print(f"total decomposed: {len(all_output)}")
    print(f"total cost: {model.total_cost}")
    with open(save_file, "w") as f:
        json.dump(all_output, f, indent=2, ensure_ascii=False)
    print(f"saved to {save_file}")


if __name__ == "__main__":
    main()
