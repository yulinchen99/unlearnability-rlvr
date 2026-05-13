INSTRUCT_PROMPT = "\n\nPlease put final answer within \\boxed{{}}."

import json
import sys
import os
import glob
from tqdm import tqdm
from google import genai
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def generate_answer_with_gemini(client, subproblem_text, max_retries=3):
    """
    Generate answer for a subproblem using Gemini API
    """
    prompt = subproblem_text + INSTRUCT_PROMPT
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
            )
            # print(response)
            # print(response.usage_metadata)
            # print(response.usage_metadata.prompt_token_count)



            # print(response.prompt_token_count)
            # print(response.usage_metadata.thoughts_token_count)
            return response.text
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                print(f"Error generating answer after {max_retries} attempts: {e}")
                return None

def process_subproblem(args):
    """
    Process a single subproblem and generate answer
    """
    client, subproblem = args
    subproblem_text = subproblem.get(PROBLEM_TEXT_KEY, "")
    if not subproblem_text:
        return subproblem
    
    gemini_answer = generate_answer_with_gemini(client, subproblem_text)
    result = subproblem.copy()
    result["gemini_answer"] = gemini_answer
    return result

def process_file(file_path, api_key, max_workers=10):
    """
    Process a single augmented data file and generate answers for all subproblems
    """
    print(f"Processing file: {file_path}")
    
    # Initialize Gemini client
    client = genai.Client(api_key=api_key)
    
    # Load data
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Collect all subproblems to process
    all_subproblems = []
    item_indices = []
    subproblem_indices = []
    
    for item_idx, item in enumerate(data):
        if PROBLEM_LIST_KEY in item:
            for sub_idx, subproblem in enumerate(item[PROBLEM_LIST_KEY]):
                # Skip if already has gemini_answer
                if "gemini_answer" not in subproblem:
                    all_subproblems.append((client, subproblem))
                    item_indices.append(item_idx)
                    subproblem_indices.append(sub_idx)
        # break
    
    if not all_subproblems:
        print(f"No subproblems to process in {file_path}")
        return data
    
    print(f"Found {len(all_subproblems)} subproblems to process")
    
    # Process subproblems in parallel
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(process_subproblem, args): (item_idx, sub_idx)
            for args, item_idx, sub_idx in zip(all_subproblems, item_indices, subproblem_indices)
        }
        
        for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc=f"Processing {os.path.basename(file_path)}"):
            item_idx, sub_idx = future_to_idx[future]
            try:
                result = future.result()
                if (item_idx, sub_idx) not in results:
                    results[(item_idx, sub_idx)] = result
            except Exception as e:
                print(f"Error processing subproblem at item {item_idx}, subproblem {sub_idx}: {e}")
                # Keep original subproblem if processing fails
                results[(item_idx, sub_idx)] = data[item_idx][PROBLEM_LIST_KEY][sub_idx].copy()
                results[(item_idx, sub_idx)]["gemini_answer"] = None
                results[(item_idx, sub_idx)]["gemini_error"] = str(e)
                
    
    # Update data with generated answers
    for (item_idx, sub_idx), result in results.items():
        data[item_idx][PROBLEM_LIST_KEY][sub_idx] = result
        # add id
        data[item_idx][PROBLEM_LIST_KEY][sub_idx]["id"] = f"{data[item_idx]['id']}_{sub_idx}"


    
    return data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__ or "Generate answers via Gemini for augmented/decomposed problems.")
    parser.add_argument("--setting", required=True,
                        help="Identifier matching the augment/decompose run, e.g. qwen_0.5b_math_level1to4.")
    parser.add_argument("--data-type", required=True,
                        help="learnable / unlearnable / etc.")
    parser.add_argument("--mode", required=True, choices=["augmented", "decomposed"])
    parser.add_argument("--input-dir", default=None,
                        help="Default: ./{mode}_data")
    parser.add_argument("--output-dir", default=None,
                        help="Default: ./{mode}_data_gemini_answer")
    parser.add_argument("--max-workers", type=int, default=10)
    args = parser.parse_args()
    if args.mode == "augmented":
        PROBLEM_LIST_KEY = "augmented_problems"
        PROBLEM_TEXT_KEY = "problem"
    else:
        PROBLEM_LIST_KEY = "subproblems"
        PROBLEM_TEXT_KEY = "subproblem"

    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        print("Error: GEMINI_API_KEY not set. Please set it as an environment variable.")
        sys.exit(1)

    input_dir = args.input_dir or f"./{args.mode}_data"
    output_dir = args.output_dir or f"./{args.mode}_data_gemini_answer"
    os.makedirs(output_dir, exist_ok=True)

    pattern = os.path.join(input_dir, f"{args.mode}_data_{args.setting}_{args.data_type}_*.json")
    file_paths = sorted(glob.glob(pattern))
    print(f"Found {len(file_paths)} files to process")
    if not file_paths:
        print(f"No files found matching pattern: {pattern}")
        sys.exit(1)

    for file_path in file_paths:
        output_path = os.path.join(
            output_dir,
            os.path.basename(file_path).replace(".json", "_with_gemini_answers.json"),
        )
        if os.path.exists(output_path):
            print(f"Skipping {file_path} because {output_path} already exists")
            continue
        try:
            updated_data = process_file(file_path, gemini_api_key, max_workers=args.max_workers)
            with open(output_path, "w") as f:
                json.dump(updated_data, f, indent=2, ensure_ascii=False)
            print(f"Saved results to: {output_path}")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    print("All files processed!")
