"""
Heatmap of pairwise GRPO-gradient cosine similarity, with prompts ordered by
classification group (e.g. easy / learnable / unlearnable).

Consumes one or more `qid_cosine_similarity_pair_dict_*.json` files written
by prompt_grad_similarity.py (cal_similarity.sh). Each key is the string of
a tuple "(qid_a, qid_b)" and each value is the cosine similarity.

Group membership is read from the {prefix}_{group}.json files written by
classification/finalize_unlearnable_example_id.py.
"""
import argparse
import ast
import glob
import json
import os
import sys

import numpy as np


def expand_globs(patterns):
    paths = []
    for p in patterns:
        matches = sorted(glob.glob(p))
        if not matches:
            sys.exit(f"no files matched {p!r}")
        paths.extend(matches)
    return paths


def load_similarity(paths):
    sim = {}
    for path in paths:
        with open(path) as f:
            d = json.load(f)
        for k, v in d.items():
            a, b = ast.literal_eval(k)
            sim[(min(a, b), max(a, b))] = float(v)
    return sim


def load_groups(dir_, prefix, groups):
    group_of = {}
    for g in groups:
        path = os.path.join(dir_, f"{prefix}_{g}.json")
        if not os.path.exists(path):
            sys.exit(f"missing group file: {path}")
        for qid in json.load(open(path)):
            group_of[qid] = g
    return group_of


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--similarity-files", nargs="+", required=True,
                        help="Paths or globs to qid_cosine_similarity_pair_dict_*.json files.")
    parser.add_argument("--example-ids-dir", required=True,
                        help="Directory containing {prefix}_{group}.json.")
    parser.add_argument("--example-ids-prefix", required=True)
    parser.add_argument("--groups", nargs="+",
                        default=["easy", "learnable", "unlearnable"],
                        help="Group order (left-to-right, top-to-bottom on the heatmap).")
    parser.add_argument("--output", required=True,
                        help="Output image path (e.g. heatmap.png or .pdf).")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--cmap", default="RdBu_r")
    parser.add_argument("--title", default=None)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--figsize", type=float, nargs=2, default=(7.0, 6.0),
                        metavar=("W", "H"))
    args = parser.parse_args()

    paths = expand_globs(args.similarity_files)
    sim = load_similarity(paths)
    group_of = load_groups(args.example_ids_dir, args.example_ids_prefix, args.groups)

    qids_in_sim = {q for pair in sim for q in pair if q in group_of}
    if not qids_in_sim:
        sys.exit("no qids from the similarity files belong to the listed groups")

    group_index = {g: i for i, g in enumerate(args.groups)}
    ordered = sorted(qids_in_sim, key=lambda q: (group_index[group_of[q]], q))
    pos = {q: i for i, q in enumerate(ordered)}
    n = len(ordered)

    mat = np.full((n, n), np.nan, dtype=np.float32)
    np.fill_diagonal(mat, 1.0)
    for (a, b), v in sim.items():
        if a in pos and b in pos:
            mat[pos[a], pos[b]] = v
            mat[pos[b], pos[a]] = v

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    im = ax.imshow(mat, cmap=args.cmap, vmin=args.vmin, vmax=args.vmax,
                   aspect="equal", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("cosine similarity")

    sizes = [sum(1 for q in ordered if group_of[q] == g) for g in args.groups]
    boundaries = np.cumsum(sizes)
    for b in boundaries[:-1]:
        ax.axhline(b - 0.5, color="black", linewidth=0.8)
        ax.axvline(b - 0.5, color="black", linewidth=0.8)

    centers = []
    prev = 0
    for b in boundaries:
        centers.append((prev + b - 1) / 2)
        prev = b
    ax.set_xticks(centers)
    ax.set_xticklabels(args.groups)
    ax.set_yticks(centers)
    ax.set_yticklabels(args.groups, rotation=90, va="center")
    ax.tick_params(length=0)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

    if args.title:
        ax.set_title(args.title)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {args.output}  ({n} prompts: " +
          ", ".join(f"{g}={s}" for g, s in zip(args.groups, sizes)) + ")")


if __name__ == "__main__":
    main()
