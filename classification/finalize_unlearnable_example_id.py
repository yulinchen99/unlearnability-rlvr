"""
Finalize the unlearnable / learnable / no-reward prompt-ID partitions
directly from per-run pass@1 evaluations.

Inputs per training run:
  --run         NAME=GLOB   pass@1 test-result JSONs from inference/test.sh.
  --reward-stats NAME=PATH  prompt_reward_stats.csv emitted during training.
                            Used as the exact source of no-reward prompts.
Plus:
  --initial-run NAME=GLOB   pass@1 test-result JSONs for the untrained model.

Definitions (matching the paper):
  no_reward    = union over training runs of prompts whose max mean_reward
                 across all logged training steps is 0 (taken straight from
                 each run's prompt_reward_stats.csv). These are treated as
                 verifier failures, not learning failures.
  unlearnable  = intersection over training runs of pass@1<=threshold qids,
                 minus no_reward.
  learnable    = intersection over training runs of per-run learnable sets,
                 where a qid is "learnable in run X" iff it is initially
                 flagged (pass@1<=threshold in --initial-run) and no longer
                 flagged after training (pass@1>threshold in run X).
                 Equivalently: initial_flagged minus the union of
                 training-run flags.

Outputs four JSON files under --output-dir:
  {output_prefix}_unlearnable.json
  {output_prefix}_learnable.json
  {output_prefix}_no_reward.json
  {output_prefix}_easy.json        all qids with initial pass@1 > threshold.

Each --run / --initial-run / --reward-stats argument is NAME=VALUE; for
--run / --initial-run, VALUE is a glob matching one or more result JSONs
(sharded outputs are concatenated); for --reward-stats, VALUE is the path
to a single CSV. NAMEs in --run and --reward-stats must match 1:1.
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict


def load_results(glob_pattern):
    paths = sorted(glob.glob(glob_pattern))
    if not paths:
        sys.exit(f"no result files matched {glob_pattern!r}")
    items = []
    for p in paths:
        with open(p) as f:
            items.extend(json.load(f))
    return items, paths


def per_run_qid_scores(items):
    out = defaultdict(list)
    for item in items:
        out[item["question_id"]].append(int(item["verification"]["score"]))
    return out


def parse_name_value(arg, label):
    if "=" not in arg:
        sys.exit(f"--{label} expects NAME=VALUE, got {arg!r}")
    name, value = arg.split("=", 1)
    if not name or not value:
        sys.exit(f"--{label} expects NAME=VALUE, got {arg!r}")
    return name, value


def load_no_reward_from_csv(path):
    """Prompts whose max mean_reward across all logged training steps is 0."""
    if not os.path.exists(path):
        sys.exit(f"reward-stats CSV not found: {path}")
    max_reward = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row["prompt_id"])
            r = float(row["mean_reward"])
            prev = max_reward.get(pid)
            if prev is None or r > prev:
                max_reward[pid] = r
    return {pid for pid, r in max_reward.items() if r == 0.0}


def evaluate_run(name, pattern, threshold):
    items, paths = load_results(pattern)
    qid_scores = per_run_qid_scores(items)
    n_set = {len(s) for s in qid_scores.values()}
    if len(n_set) > 1:
        print(f"  warning: {name} has uneven sample counts per qid: {sorted(n_set)}")
    all_qids = set(qid_scores)
    flagged = {q for q, s in qid_scores.items() if (sum(s) / len(s)) <= threshold}
    print(f"{name}: {len(items)} rollouts across {len(qid_scores)} qids "
          f"({len(paths)} file(s)); pass@1 <= {threshold}: {len(flagged)}")
    return flagged, all_qids


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", action="append", default=[], required=True,
                        metavar="NAME=GLOB",
                        help="Per-run result-file glob, repeatable.")
    parser.add_argument("--reward-stats", action="append", default=[], required=True,
                        metavar="NAME=PATH",
                        help="Per-run prompt_reward_stats.csv path, repeatable. "
                             "NAMEs must match --run NAMEs 1:1.")
    parser.add_argument("--initial-run", required=True, metavar="NAME=GLOB",
                        help="Result glob for the initial (untrained) model.")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="qids with pass@1 <= threshold are flagged.")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write the three partition JSONs to.")
    parser.add_argument("--output-prefix", required=True,
                        help="Prefix for output filenames, e.g. qwen_0.5b_math_level1to4.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    runs = [parse_name_value(r, "run") for r in args.run]
    reward_stats = dict(parse_name_value(r, "reward-stats") for r in args.reward_stats)
    run_names = {name for name, _ in runs}
    if run_names != set(reward_stats):
        sys.exit(f"--run names {sorted(run_names)} must match --reward-stats names "
                 f"{sorted(reward_stats)} exactly")

    intersect_flagged = None
    union_flagged = set()
    no_reward_union = set()
    for name, pattern in runs:
        flagged, _ = evaluate_run(name, pattern, args.threshold)
        intersect_flagged = flagged if intersect_flagged is None else intersect_flagged & flagged
        union_flagged |= flagged

        nr = load_no_reward_from_csv(reward_stats[name])
        no_reward_union |= nr
        print(f"{name}: no-reward prompts from {reward_stats[name]}: {len(nr)}")

    init_name, init_pattern = parse_name_value(args.initial_run, "initial-run")
    initial_flagged, initial_all = evaluate_run(init_name, init_pattern, args.threshold)

    unlearnable = intersect_flagged - no_reward_union
    learnable = initial_flagged - union_flagged
    easy = initial_all - initial_flagged

    out = {
        "unlearnable": sorted(unlearnable),
        "learnable": sorted(learnable),
        "no_reward": sorted(no_reward_union),
        "easy": sorted(easy),
    }
    for name, qids in out.items():
        path = os.path.join(args.output_dir, f"{args.output_prefix}_{name}.json")
        with open(path, "w") as f:
            json.dump(qids, f)
        print(f"{name}: {len(qids)} -> {path}")

    total = len(initial_all)
    if total > 0:
        print(f"unlearnable ratio: {len(unlearnable) / total:.3f}")
        print(f"learnable ratio:   {len(learnable) / total:.3f}")
        print(f"no-reward ratio:   {len(no_reward_union) / total:.3f}")
        print(f"easy ratio:        {len(easy) / total:.3f}")


if __name__ == "__main__":
    main()
