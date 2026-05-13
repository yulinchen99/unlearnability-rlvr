import json
import pandas as pd
import os
from collections import defaultdict
import numpy as np
import string

def get_model(text):
    if "Qwen-2.5-0.5B-SimpleRL-Zoo" in text:
        return None
    elif "Qwen2.5-0.5B" in text or "replicate-qwen-0.5B" in text or  "easy_explore" in text:
        return "Qwen2.5-0.5B"
    elif "Qwen2.5-1.5B" in text:
        return "Qwen2.5-1.5B"
    elif "Qwen2.5-3B" in text:
        return "Qwen2.5-3B"
    elif "Qwen2.5-7B" in text:
        return "Qwen2.5-7B"
    elif "Qwen2.5-14B" in text:
        return "Qwen2.5-14B"
    elif "Qwen2.5-32B" in text:
        return "Qwen2.5-32B"
    else:
        # print("default to qwen-0.5b")
        return "unknown"
        # raise ValueError(f"Unknown model in text: {text}")

def get_setting(model_path):
    text = model_path
    if "entloss_weight" in text:
        if "instance_power"  in text:
            return "entloss_weight_instance_power"
        elif "instance_linear" in text:
            return "entloss_weight_instance_linear"        
    elif "divsample" in text or "downsamplecorrect" in text:
        if "downsamplecorrect0.5" in text:
            return "downsample_correct_0.5"
        elif "downsamplecorrect0.1" in text:
            return "downsample_correct_0.1"
        else:
            return "downsample_unknown"
    elif "data_ablate" in text:
        return "data_ablate"
    elif "loss_ablate" in text:
        return "loss_ablate"
    elif "highent_tokenmask" in text:
        if "entcoef0.1" in text:
            return "highent_tokenmask_entcoef0.1"
        else:
            return "highent_tokenmask_unknown"
    elif "replicate" in text or "baseline" in text:
        # print(text)
        return "baseline"
    elif "easy_explore" in text:
        if "baseline" in text:
            return "easy_explore_baseline"
        elif "easy_focus" in text:
            return "easy_explore_easyfocus"
        else:
            return "easy_explore_unknown"
    elif "cot_quality" in text:
        return "cot_quality"
    else:
        print(text)
        return None
        
def get_data(text):
    if "0.2k_easy_0.8k_hard" in text:
        return "0.2k_easy_0.8k_hard"
    else:
        return "math"
        
def get_step(text, output_file=None):
    if output_file is not None:
        if "global_step" in output_file:
            return int(output_file.split("global_step_")[-1].split("_")[0].strip())
        elif "step" in output_file:
            return int(output_file.split("step_")[-1].split("_")[0].strip())
    if "global_step" in text:
        return int(text.split("/")[-3].split("_")[-1])
    else:
        return 0

def get_temperature(item):
    if "temperature" in item:
        return float(item["temperature"])
    text = item["output_file"]
    if text is None or "global_step" not in text:
        return 0.7
    _, text_later = text.split("global_step")
    if "1.0" in text_later:
        return 1.0
    elif "0.5" in text_later:
        return 0.5
    elif "0.0" in text_later:
        return 0.0
    else:
        #default
        return 0.7

def get_lr(text):
    if "1e-7" in text:
        return "1e-7"
    elif "1e-6" in text:
        return "1e-6"
    else:
        return "5e-7"

def load_json(file_path, old_version=False):
    """
    Load a JSON file and convert to dataframe.
    
    :param file_path: Path to the JSON file.
    :return: Parsed JSON content as a Python dictionary.
    """
    csv_file = file_path.replace(".json", ".csv")
    # if os.path.exists(csv_file): 
        # return pd.read_csv(csv_file)
    # print("file_path:", file_path)
    if file_path.endswith(".jsonl"):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = [json.loads(line) for line in file]
    else:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    dict_data = {
        "model": [],
        "data": [],
        "prompt": [],
        "pass_k":[],
        "score": [],
        "step":[],
        "temperature": [],
        "setting": []
    }
    for item in data:
        # print(item)
        # setting_tmp = get_setting(item["model_path"])
        # if setting_tmp is None:
        #     continue
        # print("here:", item)
        model_name = get_model(item["model_path"])
        step = get_step(item["model_path"], output_file = item["output_file"])
        if item["output_file"] and not old_version:
            setting = item["output_file"].split("/")[-1].split("_step")[0].strip()
        else:
            setting = get_data(item["model_path"]) + "_" + get_setting(item["model_path"]) + "_" + get_lr(item["model_path"])
        temperature = get_temperature(item)

        if temperature > 0.0 and item["num_samples"] not in [128, 256]:
            # print("invalid:", item)
            continue
            

        for k, v in item["results"].items():
            if k == "support":
                continue
            dict_data["temperature"].append(temperature)
            dict_data["model"].append(model_name)
            dict_data["data"].append(item.get("test_data", None))
            dict_data["prompt"].append(item["prompt_type"])
            dict_data["setting"].append(setting)
            dict_data["step"].append(step)
            k = int(k.split("@")[-1])
            dict_data["pass_k"].append(k)
            dict_data["score"].append(v)
    df = pd.DataFrame(dict_data)
    df.to_csv(file_path.replace(".json", ".csv"), index=False)
    print(f"Data saved to {file_path.replace('.json', '.csv')}")
    return df

def load_qid2score(data_path):
    if data_path.endswith(".json"):
        item_list = json.load(open(data_path))
    elif data_path.endswith(".jsonl"):
        item_list = [json.loads(line) for line in open(data_path)]
    else:
        raise ValueError(f"Unknown file type: {data_path}")
    qid2score = defaultdict(list)
    for item in item_list:
        qid2score[item["question_id"]].append(item["verification"]["score"])
    return qid2score
        
def cosine_similarity_stable(grad_a, grad_b):
    max_a_value = np.max(np.abs(grad_a))
    max_b_value = np.max(np.abs(grad_b))
    if max_a_value > 1:
        grad_a = grad_a / max_a_value
    if max_b_value > 1:
        grad_b = grad_b / max_b_value
    return np.dot(grad_a, grad_b) / (np.linalg.norm(grad_a) * np.linalg.norm(grad_b))

def get_scaled_gradient(gradient):
    max_value = np.max(np.abs(gradient))
    if max_value > 1:
        gradient = gradient / max_value
    return gradient

def flatten_gradient(gradient, sorted_keys):
    flattened_gradient = []
    for key in sorted_keys:
        flattened_gradient.append(gradient[key].flatten())
    return np.concatenate(flattened_gradient)

def preprocess_text(text):
    """Lowercases and removes punctuation from the text."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def get_word_ngrams(text: str, n: int):
    text = preprocess_text(text)
    words = text.split()
    return {tuple(words[i:i+n]) for i in range(len(words) - n + 1)}

def calculate_pairwise_ngram_similarity(string_1, string_2, n: int) -> float:
    """
    Calculate pairwise n-gram similarity.
    """
    # Jaccard similarity for n-gram overlap: intersection / union
    # assert len(ngrams_list) == 2, "Need at least two strings to calculate pairwise similarity"
    # ngrams1 = ngrams_list[0]
    # ngrams2 = ngrams_list[1]
    ngrams1 = get_word_ngrams(string_1, n)
    ngrams2 = get_word_ngrams(string_2, n)
    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    if union > 0:
        similarity = intersection / union
        return similarity
    else:
        return None