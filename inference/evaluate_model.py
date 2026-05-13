import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import torch
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
import numpy as np
from math_verifier import MathResponseVerifier
from collections import defaultdict
from datetime import datetime
import random
from datasets import load_dataset, Dataset
from common.parser import parse_question, parse_ground_truth
import pandas as pd
from utils import evaluate_mc

def seed_everything(seed: int = 0):
    """Seed everything for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def lower_keys(example):
    new_example = {}
    for key, value in example.items():
        if key != key.lower():
            new_key = key.lower()
            new_example[new_key] = value
        else:
            new_example[key] = value
    return new_example

def load_jsonl_as_list(file):
    with open(file, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def load_jsonl(file):
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()

def load_test_data(data_path: str) -> List[Dict[str, Any]]:
    """Load test data from a JSON file."""
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)

DATA_ROOT = os.environ.get("DATA_ROOT", "./data")

def load_hf_data(data_name, split, data_dir="./data", max_problems=None):
    data_name_save = data_name
    if  max_problems is not None:
        data_name_save = f"{data_name}_sample_{max_problems}"
    data_file = f"{data_dir}/{data_name_save}/{split}_parsed.jsonl"
    print("path to look for:", data_file)
    if os.path.exists(data_file):
        examples = list(load_jsonl(data_file))
        return examples
    
    data_file = f"{data_dir}/{data_name_save}/{split}.jsonl"
    if os.path.exists(data_file):
        examples = list(load_jsonl(data_file))
    else:
        if data_name == "math_train":
            dataset = load_jsonl(f"{DATA_ROOT}/math/train.jsonl")
        elif data_name in ["simplelr_qwen_level1to4", "simplelr_qwen_level3to5"]:
            dataset = pd.read_parquet(f"{DATA_ROOT}/SimpleRL-Zoo-Data/{data_name}/{split}.parquet")
            dataset = dataset.to_dict(orient="records")
        elif data_name == "aime25":
            dataset = load_dataset("math-ai/aime25")["test"]
        elif data_name == "hendrycks_math":
            dataset = load_dataset("parquet", data_files=f"{DATA_ROOT}/hendrycks_math/{split}.parquet")["train"]
        elif data_name == "deepscaler":
            dataset = load_dataset("parquet", data_files=f"{DATA_ROOT}/deepscaler/{split}.parquet")["train"]
        elif data_name == "math_perturb_simple":
            dataset = load_jsonl(f"{DATA_ROOT}/math_perturb/math_perturb_simple.jsonl")
            # filter dataset by original_split == split
            dataset = [example for example in dataset if example["original_split"] == split]
            # convert each value in each item to str
            dataset = [
                {k: str(v) for k, v in item.items()}
                for item in dataset
            ]
        elif data_name in ["SCP-116K-cleaned-sampled-5k", "DAPO-Math-17k-sampled-5k"]:
            dataset = load_dataset("parquet", data_files=f"{DATA_ROOT}/{data_name}/{split}.parquet")["train"]
        elif data_name == "math_500":
            dataset = load_jsonl(f"{DATA_ROOT}/math_500/test.jsonl")
        elif "simplelr" in data_name:
            test_data_path = f"{DATA_ROOT}/SimpleRL-Zoo-Data/{data_name}/train.parquet"
            dataset = load_dataset("parquet", data_files={"train": test_data_path})["train"]
        else:
            # try loading the dataset from the data_dir
            print("Loading dataset from:", f"{DATA_ROOT}/{data_name}; split:", split)
            dataset = load_dataset(f"{DATA_ROOT}/{data_name}", split=split)
            # raise NotImplementedError(data_name)

        examples = list(dataset)
        if data_name == "simplelr_qwen_level1to4_sample_2k":
            # sample by ids
            target_ids = list(range(100)) + list(range(200, 300)) + list(range(600, 700))
            examples = [example for example in examples if example["id"] in target_ids]
        examples = [lower_keys(example) for example in examples]
        dataset = Dataset.from_list(examples)
        os.makedirs(f"{data_dir}/{data_name_save}", exist_ok=True)
        dataset.to_json(data_file)
    
    def get_potential_id(example):
        if "uid" in example:
            return example["uid"]
        if "extra_info" in example and "index" in example["extra_info"]:
            return example["extra_info"]["index"]
        return None
    

    # add 'idx' in the first column
    if get_potential_id(examples[0]) is None:
        examples = [{"id": i, **example} for i, example in enumerate(examples)]
    else:
        examples = [{"id": get_potential_id(example), **example} for example in examples]

    # dedepulicate & sort
    examples = list(sorted(examples, key=lambda x: x["id"]))
    if max_problems is not None:
        examples = random.sample(examples, min(max_problems, len(examples)))

    for example in tqdm(examples, desc="Parsing questions and ground truths"):
        example["question"] = parse_question(example, data_name)
        _, example["ground_truth"] = parse_ground_truth(example, data_name)
    os.makedirs(f"{data_dir}/{data_name_save}", exist_ok=True)
    data_file = f"{data_dir}/{data_name_save}/{split}_parsed.jsonl"

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            # Handle other non-serializable numpy types, if needed
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return super().default(obj)

    with open(data_file, "w") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False, cls=NumpyEncoder) + "\n")
    return examples

def calculate_pass_at_k(n_samples: int, problem_results: Dict[str, List[float]], k: int) -> float:
    """
    Calculate probability of solving a problem with k attempts.
    
    Args:
        n_samples: Total number of samples
        n_correct: Number of correct samples
        k: Number of attempts allowed
    
    Returns:
        Probability of solving the problem within k attempts
    """
    if n_samples < k:
        raise ValueError(f"n_samples ({n_samples}) must be greater than or equal to k ({k})")

    def estimate_pass_at_k(n_samples: int, problem_results: Dict[str, List[float]], k: int) -> float:
        assert len(problem_results) == n_samples
        n_correct = sum(problem_results)
        
        # Calculate probability using combination formula
        n = n_samples
        c = n_correct
        if c == 0:
            return 0.0
        
        # Calculate 1 - P(all k samples are wrong)
        return 1.0 - np.prod([(n - c - i)/(n - i) for i in range(k)])
    
    incorrect_q_id = []
    correct = 0
    total = 0
    for q_id, results in problem_results.items():
        score = estimate_pass_at_k(n_samples, results, k)
        correct += score
        total += 1
        if score == 0:
            incorrect_q_id.append(q_id)
    
    return correct / total, incorrect_q_id

def load_verifier(test_data_path):
    if any(pattern in test_data_path for pattern in ["gpqa_diamond", "mmlupro", "webinstruct"]):
        print("Using evaluate_mc for verifier")
        return MathResponseVerifier(evaluate_func=evaluate_mc)
    return MathResponseVerifier()


def evaluate_pass_at_k(generator,
                      test_data_path: str, 
                      prompt_type: str,
                      num_samples: int = 10,
                      max_new_tokens: int = 512,
                      max_problems: int = None,
                      beam_search=False,
                      temperature: float = 0.7,
                      top_p: float = 0.95) -> Dict[str, float]:
    """
    Evaluate pass@k metrics for a given model on test data.
    
    Args:
        model_path: Path to the model
        test_data_path: Path to test data JSON file
        num_samples: Number of samples to generate per problem
        k_values: List of k values to evaluate
        batch_size: Batch size for generation
        max_new_tokens: Maximum number of new tokens to generate
    
    Returns:
        Dictionary containing pass@k metrics for each k
    """

    # Load test data
    if test_data_path.endswith(".json"):
        test_data = load_test_data(test_data_path)
    elif test_data_path.endswith(".jsonl"):
        test_data = list(load_jsonl(test_data_path))
        for item in test_data:
            if "ground_truth" not in item:
                item["ground_truth"] = item["gt_answer"]
    elif "@" in test_data_path:
        test_data_path, split = test_data_path.split("@")
        test_data = load_hf_data(test_data_path, split=split,  max_problems=max_problems)
    else:
        # data_name
        print("default split is test")
        test_data = load_hf_data(test_data_path, split="test",  max_problems=max_problems)
        
    verifier = load_verifier(test_data_path)

    print(f"Loaded {len(test_data)} test problems")
    print(test_data[0])
    
    if max_problems is not None:
        # random sample max_problems problems
        if len(test_data) > max_problems:
            test_data = list(random.sample(test_data, max_problems))
        else:
            print(f"Warning: max_problems ({max_problems}) is greater or equal to than the number of test problems ({len(test_data)}), do nothing")

    # strip potential embedded template in 'question'
    def strip_template(question):
        if question.startswith("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user"):
            question = question[len("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user"):].strip()
        if question.endswith("<|im_end|>\n<|im_start|>assistant"):
            question = question[:-len("<|im_end|>\n<|im_start|>assistant")].strip()
        if question.endswith("Please reason step by step, and put your final answer within \\boxed{}."):
            question = question[:-len("Please reason step by step, and put your final answer within \\boxed{}.")].strip()
        return question

    for item in test_data:
        item["question"] = strip_template(item["question"])

    all_responses = generator.generate_responses(
            questions=test_data,
            question_type="math",
            prompt_type=prompt_type,
            temperature=temperature,
            top_p=top_p,
            max_length=max_new_tokens,
            num_samples=num_samples,
            beam_search=beam_search
        )

    verified_responses = verifier.verify_responses(all_responses)

    problem_results = defaultdict(list)
    for item in verified_responses:
        problem_results[item["question_id"]].append(int(item["verification"]["score"]))

    results = cal_score(num_samples, problem_results)

    return results, verified_responses

def cal_score(num_samples, problem_results):
    # Calculate pass@k for each k
    results = {}

    # calculate k_values automatically
    k_values = [1]
    power = 1
    while 2**power <= num_samples:  # Add powers of 2 up to max_len
        k_values.append(2**power)
        power += 1

    for k in k_values:
        pass_at_k, _ = calculate_pass_at_k(num_samples, problem_results, k)
        results[f"pass@{k}"] = pass_at_k
    results["support"] = len(problem_results)
    return results

def cal_score_from_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    problem_results = defaultdict(list)
    for item in data:
        problem_results[item["question_id"]].append(int(item["verification"]["score"]))
    results = cal_score(len(problem_results[list(   problem_results.keys())[0]]), problem_results)
    return results

def auto_save_results(filepath, output_file):
    if os.path.exists(output_file):
        results_list = load_jsonl_as_list(output_file)
    else:
        results_list = []
    outfile_set = set([item["output_file"] for item in results_list])
    if filepath in outfile_set:
        print(f"Skipping {filepath} because it already exists")
        return
    with open(output_file, "a+") as f:
        f.write(json.dumps(filepath, ensure_ascii=False) + "\n")
    return 0

def main():
    parser = argparse.ArgumentParser(description="Evaluate model's pass@k metrics")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    parser.add_argument("--tokenizer-path", type=str, required=True, help="Path to the tokenizer")

    parser.add_argument("--test-data", type=str, required=True, help="Path to test data JSON file")
    parser.add_argument("--prompt-type", type=str, required=True, help="Prompt type")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples per problem")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum number of new tokens to generate")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--use-sglang", action="store_true", help="Use sglang")
    parser.add_argument("--max-problems", type=int, default=None, help="Maximum number of problems to evaluate")
    parser.add_argument("--beam-search", action="store_true")
    parser.add_argument("--save-path", type=str, default="./results", help="Path to save the generated responses")
    parser.add_argument("--model-name", type=str, default=None, help="Path to save the generated responses")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p")
    parser.add_argument("--gpu-memory-util", type=float, default=0.75, help="GPU memory utilization")
    parser.add_argument("--lora", type=str, default=None, help="Path to LoRA adapter directory (enables LoRA inference)")

    args = parser.parse_args()

    if args.model_name is None:
        args.model_name = args.model_path.split("/")[-1]

    os.makedirs(args.save_path, exist_ok=True)

    seed_everything(args.seed)

    model_path = args.model_path
    model_loaded = False

    test_data_list = args.test_data.split(",")
    for test_data in test_data_list:
        test_data = test_data.strip()
        if test_data:
            print(f"Evaluating {test_data}")
            if test_data.endswith(".jsonl") or test_data.endswith(".json"):
                result_output_file = os.path.join(args.save_path, f"{args.model_name}_{args.temperature}_{args.num_samples}_{args.max_new_tokens}_{args.max_problems}_{args.seed}_{args.prompt_type}.json")
            else:
                result_output_file = os.path.join(args.save_path, f"{args.model_name}_{test_data}_{args.temperature}_{args.num_samples}_{args.max_new_tokens}_{args.max_problems}_{args.seed}_{args.prompt_type}.json")
            if os.path.exists(result_output_file):
                print(f"Skipping {test_data} because it already exists")
                auto_save_results(result_output_file, os.path.join(args.save_path, "test_result.jsonl"))
                continue
            if not model_loaded:
                print(f"Loading model from {model_path}")
                if args.use_sglang:
                    from gen_utils_sglang import ResponseGenerator as ResponseGeneratorSGLang
                    generator = ResponseGeneratorSGLang(model_path, args.tokenizer_path or model_path)
                else:
                    from gen_utils import ResponseGenerator
                    generator = ResponseGenerator(model_path, args.tokenizer_path or model_path, gpu_memory_util=args.gpu_memory_util, lora_dir=args.lora)
                model_loaded = True

            results, verified_responses = evaluate_pass_at_k(
                generator=generator,
                test_data_path=test_data,
                prompt_type=args.prompt_type,
                num_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                max_problems=args.max_problems,
                beam_search=args.beam_search,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            with open(result_output_file, "w") as f:
                json.dump(verified_responses, f, indent=2)

            print(results)
            output_file = os.path.join(args.save_path, "test_result.jsonl")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            current_item = {
                "model_path": args.model_path,
                "test_data": test_data,
                "prompt_type": args.prompt_type,
                "num_samples": args.num_samples,
                "max_problems": args.max_problems,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "seed": args.seed,
                "output_file": result_output_file,
                "results": results
            }
            with open(output_file, "a+") as f:
                f.write(json.dumps(current_item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main() 