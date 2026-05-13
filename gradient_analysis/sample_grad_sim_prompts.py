"""
Sample N prompts from each example-id group and write a filtered test JSONL
containing only those prompts, to be passed to inference/test.sh as
--test-data for the gradient-analysis rollouts.

Inputs:
  --example-ids-dir / --example-ids-prefix
      Point at the JSONs written by classification/finalize_unlearnable_example_id.py:
      {dir}/{prefix}_{group}.json.
  --groups
      Which groups to sample from. Default: easy learnable unlearnable.
  --n-per-group
      Number of prompt-ids to sample per group (sampled without replacement;
      groups smaller than N use all of their qids).
  --source-test-jsonl
      The parsed test JSONL inference/test.sh writes on first use
      (./data/{data_name}/test_parsed.jsonl).
  --output-jsonl
      Where to write the filtered test JSONL.
"""
import argparse
import json
import os
import random
import sys


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--example-ids-dir", required=True)
    parser.add_argument("--example-ids-prefix", required=True)
    parser.add_argument("--groups", nargs="+",
                        default=["easy", "learnable", "unlearnable"])
    parser.add_argument("--n-per-group", type=int, default=100)
    parser.add_argument("--source-test-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)

    sampled = set()
    for g in args.groups:
        path = os.path.join(args.example_ids_dir, f"{args.example_ids_prefix}_{g}.json")
        if not os.path.exists(path):
            sys.exit(f"missing group file: {path}")
        qs = json.load(open(path))
        picked = random.sample(qs, min(args.n_per_group, len(qs)))
        sampled.update(picked)
        print(f"{g}: sampled {len(picked)} of {len(qs)} (from {path})")

    if not os.path.exists(args.source_test_jsonl):
        sys.exit(f"missing source test JSONL: {args.source_test_jsonl}")

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    kept = 0
    with open(args.source_test_jsonl) as f_in, open(args.output_jsonl, "w") as f_out:
        for line in f_in:
            item = json.loads(line)
            if item.get("id") in sampled:
                f_out.write(line)
                kept += 1

    print(f"wrote {kept} / {len(sampled)} sampled prompts -> {args.output_jsonl}")
    if kept < len(sampled):
        missing = len(sampled) - kept
        print(f"  warning: {missing} sampled qids not found in {args.source_test_jsonl} "
              "(id-type mismatch between example-ids JSON and test JSONL?)")


if __name__ == "__main__":
    main()
