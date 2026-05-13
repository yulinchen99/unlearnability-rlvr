"""
Pairwise cosine similarity over per-prompt gradient files produced by
checkpoint_gradient_analysis.py.

For each unordered pair (a, b) of question_ids, computes cosine similarity
on three flavors of saved gradients (when present):
  - average across all rollouts (file: average_gradients_prompt_<qid>.npz)
  - correct rollouts only       (file: average_gradients_prompt_correct_<qid>.npz)
  - incorrect rollouts only     (file: average_gradients_prompt_incorrect_<qid>.npz)
"""

import argparse
import glob
import itertools
import json
import os
import re
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.grad_utils import flatten_gradient, get_scaled_gradient


_avg_storage = {}
_pos_storage = {}
_neg_storage = {}
_sorted_keys = None


def _get_gradient_from_file(gradient_file, idx, storage, head_idx_set, layer_idx=None):
    """Load + flatten + scale a gradient file. Caches each `idx` only if it's
    a "head" index (i.e., the smaller half of all pairs, so we hit it many times)."""
    global _sorted_keys
    if idx in storage:
        return storage[idx]

    gradient = np.load(gradient_file)
    if _sorted_keys is None:
        _sorted_keys = [k for k in gradient.keys() if "base_layer" not in k and "mlp" not in k]
        if layer_idx is not None:
            _sorted_keys = [k for k in _sorted_keys if any(f"layers.{i}" in k for i in layer_idx)]
        print("sorted keys:", json.dumps(_sorted_keys, indent=2))

    flat = flatten_gradient(gradient, _sorted_keys)
    flat = get_scaled_gradient(flat)
    if idx in head_idx_set:
        storage[idx] = flat
    return flat


def _cosine(grad_a, grad_b):
    if grad_a.dtype != np.float32:
        grad_a = grad_a.astype(np.float32, copy=False)
    if grad_b.dtype != np.float32:
        grad_b = grad_b.astype(np.float32, copy=False)
    dot = np.einsum("i,i->", grad_a, grad_b)
    return float(dot / (np.linalg.norm(grad_a) * np.linalg.norm(grad_b)))


def similarity_for_pair(file_a, file_b, idx_a, idx_b, storage, head_idx_set, layer_idx=None):
    if not (os.path.exists(file_a) and os.path.exists(file_b)):
        return None
    try:
        grad_a = _get_gradient_from_file(file_a, idx_a, storage, head_idx_set, layer_idx)
        grad_b = _get_gradient_from_file(file_b, idx_b, storage, head_idx_set, layer_idx)
        return _cosine(grad_a, grad_b)
    except Exception as e:
        print(f"Error on ({idx_a}, {idx_b}): {e}")
        return None


def collect_qids(data_dir):
    qid_list = []
    for path in glob.glob(os.path.join(data_dir, "average_gradients_prompt_correct_*.npz")):
        m = re.search(r"average_gradients_prompt_correct_(\d+)\.npz$", path)
        if m:
            qid_list.append(int(m.group(1)))
    return sorted(qid_list)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run_name", default="qwen_0.5b_demo",
                        help="Subdirectory under GRAD_ANALYSIS_OUTPUT_DIR.")
    parser.add_argument("--start_idx", type=int, default=None,
                        help="Start index into the (sorted) prompt-pair list.")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="End index into the (sorted) prompt-pair list.")
    parser.add_argument("--normalized", action="store_true",
                        help="Use *_normalized_*.npz files instead of base ones (for correct rollouts only).")
    parser.add_argument("--layer_idx", type=int, nargs="+", default=None,
                        help="If set, only include parameters from the listed transformer layers.")
    args = parser.parse_args()

    output_root = os.environ.get("GRAD_ANALYSIS_OUTPUT_DIR", "./grpo_gradient_analysis_output")
    data_dir = os.path.join(output_root, args.run_name)
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist")
        sys.exit(1)

    save_file = os.path.join(data_dir, f"qid_cosine_similarity_pair_dict_{args.start_idx}_{args.end_idx}_{args.normalized}_layer_{args.layer_idx}.json")
    save_pos_file = os.path.join(data_dir, f"qid_cosine_similarity_pair_dict_pos_{args.start_idx}_{args.end_idx}_{args.normalized}_layer_{args.layer_idx}.json")
    save_neg_file = os.path.join(data_dir, f"qid_cosine_similarity_pair_dict_neg_{args.start_idx}_{args.end_idx}_{args.normalized}_layer_{args.layer_idx}.json")
    if os.path.exists(save_pos_file):
        print(f"Pos file {save_pos_file} already exists, exiting...")
        sys.exit(1)

    all_pair_dict_file = os.path.join(data_dir, "qid_cosine_similarity_prompt_pair_pos_dict_all.json")
    if os.path.exists(all_pair_dict_file):
        with open(all_pair_dict_file) as f:
            visited_pairs = {eval(k) for k in json.load(f).keys()}
    else:
        visited_pairs = set()

    qid_list = collect_qids(data_dir)
    prompt_pairs = sorted(itertools.combinations(qid_list, 2))
    prompt_pairs = [pair for pair in prompt_pairs if pair not in visited_pairs]

    if args.start_idx is not None or args.end_idx is not None:
        prompt_pairs = prompt_pairs[args.start_idx : args.end_idx]
    print(f"Calculating cosine similarity for {len(prompt_pairs)} prompt pairs")

    head_idx_set = {pair[0] for pair in prompt_pairs}
    print(f"unique head indices: {len(head_idx_set)}")

    avgsim, possim, negsim = {}, {}, {}
    for a_idx, b_idx in tqdm(prompt_pairs):
        # average across rollouts
        score = similarity_for_pair(
            f"{data_dir}/average_gradients_prompt_{a_idx}.npz",
            f"{data_dir}/average_gradients_prompt_{b_idx}.npz",
            a_idx, b_idx, _avg_storage, head_idx_set,
        )
        if score is not None:
            avgsim[(a_idx, b_idx)] = score

        # incorrect-only
        score = similarity_for_pair(
            f"{data_dir}/average_gradients_prompt_incorrect_{a_idx}.npz",
            f"{data_dir}/average_gradients_prompt_incorrect_{b_idx}.npz",
            a_idx, b_idx, _neg_storage, head_idx_set,
        )
        if score is not None:
            negsim[(a_idx, b_idx)] = score

        # correct-only
        suffix = "correct_normalized" if args.normalized else "correct"
        score = similarity_for_pair(
            f"{data_dir}/average_gradients_prompt_{suffix}_{a_idx}.npz",
            f"{data_dir}/average_gradients_prompt_{suffix}_{b_idx}.npz",
            a_idx, b_idx, _pos_storage, head_idx_set,
            layer_idx=args.layer_idx,
        )
        if score is not None:
            possim[(a_idx, b_idx)] = score

    for sim_dict, path, label in [
        (avgsim, save_file, "avg"),
        (possim, save_pos_file, "pos"),
        (negsim, save_neg_file, "neg"),
    ]:
        if sim_dict:
            with open(path, "w") as f:
                json.dump({str(k): float(v) for k, v in sim_dict.items()}, f)
            print(f"saved {len(sim_dict)} {label} pair similarities -> {path}")


if __name__ == "__main__":
    main()
