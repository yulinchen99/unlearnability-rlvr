"""
Concatenate filtered augmented + decomposed problems into a single
test_parsed.jsonl-style file usable by inference/test.sh for rollout
generation, which in turn feeds gradient_analysis/.

Each output line: {"id": "<mode>_<orig_id>_<sub_idx>", "question": ..., "ground_truth": ...}.

Either --augmented-file, --decomposed-file, or both must be provided.
If --valid-original-ids-file is given, only augmentations whose seed id
is in that list are kept.
"""

import argparse
import json
import os


def load_filtered(path):
    if path is None:
        return []
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--augmented-file", default=None,
                        help="all_augmented_data_*_filtered.json output of filter_data.py.")
    parser.add_argument("--decomposed-file", default=None,
                        help="all_decomposed_data_*_filtered.json output of filter_data.py.")
    parser.add_argument("--output-file", required=True,
                        help="Output JSONL path.")
    parser.add_argument("--valid-original-ids-file", default=None,
                        help="Optional JSON list of original prompt ids to keep.")
    args = parser.parse_args()

    if args.augmented_file is None and args.decomposed_file is None:
        parser.error("at least one of --augmented-file or --decomposed-file is required")

    valid_ids = None
    if args.valid_original_ids_file is not None:
        valid_ids = set(json.load(open(args.valid_original_ids_file)))

    output_rows = []
    seeds_with_augmentations = set()
    for mode, path in [("augmented", args.augmented_file), ("decomposed", args.decomposed_file)]:
        for item in load_filtered(path):
            if valid_ids is not None:
                seed_id = int(str(item["id"]).split("_")[0])
                if seed_id not in valid_ids:
                    continue
                seeds_with_augmentations.add(seed_id)
            output_rows.append({
                "id": f"{mode}_{item['id']}",
                "question": item["question"],
                "ground_truth": item["answer"],
            })

    if valid_ids is not None:
        print(f"seeds with at least one survivor: {len(seeds_with_augmentations)} / {len(valid_ids)}")

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(output_rows)} rows -> {args.output_file}")


if __name__ == "__main__":
    main()
