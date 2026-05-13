"""Utilities for extracting and verifying boxed answers from LLM responses.

Imported by extract_answer.py and filter_data.py.
"""

from math_verify import parse, verify


def last_boxed_only_string(string):
    """Return the contents of the last \\boxed{...} (or \\fbox{...}) in `string`.

    Returns None if no balanced boxed expression is found.
    """
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

    ans = string[left_brace_idx + 1 : right_brace_idx].strip()
    if ans and ans[0] == "{" and ans[-1] == "}":
        ans = ans[1:-1]
    return ans.replace("\n", "")


def is_equivalent(answer1, answer2):
    """Mathematical equivalence check via the `math_verify` package."""
    answer1 = str(answer1)
    answer2 = str(answer2)
    if not answer1.startswith("$"):
        answer1 = "$" + answer1 + "$"
    if not answer2.startswith("$"):
        answer2 = "$" + answer2 + "$"
    return verify(parse(answer1), parse(answer2))
