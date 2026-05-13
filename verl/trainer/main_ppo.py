# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
print("beginning of main_ppo.py")
from verl import DataProto
import torch
from verl.utils.reward_score import gsm8k, math
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.reward_score import kk
# from verl.utils.reward_score import simplelr_math
# from verl.utils.reward_score import deepseek_r1
from verl.utils.reward_score import hf_math_verify
from verl.utils.reward_score import string_match
# import asyncio
import concurrent.futures
from typing import List, Dict, Any
# import functools
# import time
import logging
logging.info("finished importing")

def _default_compute_score(data_source, solution_str, ground_truth):
    """
    Default compute score function.
    
    Args:
        data_source: Source of the data
        solution_str: Model's solution/response
        ground_truth: Ground truth answer
        problem_text: Original problem text (unused, kept for compatibility)
    """
    # Get base score from existing scoring functions
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ['lighteval/MATH', 'DigitalLearningGmbH/MATH-lighteval']:
        return math.compute_score(solution_str, ground_truth)
    elif "kk" in data_source:
        return kk.compute_score(solution_str, ground_truth)
    elif "simplelr" in data_source or "deepscaler" in data_source or data_source == "math" or data_source == "math_dapo":
        return hf_math_verify.compute_score(solution_str, ground_truth)
    elif "deepseek_r1" in data_source:
        return deepseek_r1.compute_score(solution_str, ground_truth)
    elif "zebralogic" in data_source:
        return string_match.compute_score(solution_str, ground_truth)
    else:
        raise NotImplementedError(f"Unknown data source: {data_source}")


class RewardManager():
    """The reward manager with parallel processing capabilities.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, max_workers=128, parallel_strategy='simple') -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.max_workers = max_workers  # number of parallel workers for scoring
        self.parallel_strategy = parallel_strategy  # 'simple' or 'none'
        
        # Validate parallel strategy
        if parallel_strategy not in ['simple', 'none']:
            raise ValueError(f"parallel_strategy must be one of ['simple', 'none'], got {parallel_strategy}")

    def _prepare_batch_data(self, data: DataProto) -> List[Dict[str, Any]]:
        """Prepare batch data for parallel processing."""
        batch_items = []
        
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            
            # Get problem text if available, with fallback
            # problem_text = None
            # if 'problem_text' in data_item.non_tensor_batch:
            #     problem_text = data_item.non_tensor_batch['problem_text']
            # elif 'prompt' in data_item.non_tensor_batch:
            #     problem_text = data_item.non_tensor_batch['prompt']
            # elif 'question' in data_item.non_tensor_batch:
            #     problem_text = data_item.non_tensor_batch['question']
            # # If none of the above, we'll use the decoded prompt as fallback
            # else:
            #     problem_text = self.tokenizer.decode(valid_prompt_ids)

            batch_items.append({
                'index': i,
                'data_source': data_source,
                'solution_str': sequences_str,
                'ground_truth': ground_truth,
                'valid_response_length': valid_response_length,
                'sequences_str': sequences_str,
                # 'problem_text': problem_text
            })
        
        return batch_items

    def _compute_score_sequential(self, batch_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sequential scoring for compatibility (original behavior)."""
        results = []
        
        for item in batch_items:
            try:
                score_dict = self.compute_score(
                    data_source=item['data_source'],
                    solution_str=item['solution_str'],
                    ground_truth=item['ground_truth'],
                    # problem_text=item['problem_text'] # Pass problem_text
                )
                results.append({
                    'index': item['index'],
                    'score_dict': score_dict,
                    'valid_response_length': item['valid_response_length'],
                    'sequences_str': item['sequences_str'],
                    'data_source': item['data_source']
                })
            except Exception as exc:
                print(f'Sequential item {item["index"]} generated an exception: {exc}')
                results.append({
                    'index': item['index'],
                    'score_dict': {'score': 0.0, 'correctness': False},
                    'valid_response_length': item['valid_response_length'],
                    'sequences_str': item['sequences_str'],
                    'data_source': item['data_source']
                })
        
        return results

    def _compute_score_parallel(self, batch_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compute scores for multiple items in parallel using ThreadPoolExecutor."""
        results = []
        
        # Group items by data_source to optimize batch processing
        data_source_groups = {}
        for item in batch_items:
            data_source = item['data_source']
            if data_source not in data_source_groups:
                data_source_groups[data_source] = []
            data_source_groups[data_source].append(item)
        
        # Process each data source group in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_item = {}
            for data_source, items in data_source_groups.items():
                for item in items:
                    future = executor.submit(
                        self.compute_score,
                        data_source=item['data_source'],
                        solution_str=item['solution_str'],
                        ground_truth=item['ground_truth'],
                        # problem_text=item['problem_text'] # Pass problem_text
                    )
                    future_to_item[future] = item
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    score_dict = future.result()
                    results.append({
                        'index': item['index'],
                        'score_dict': score_dict,
                        'valid_response_length': item['valid_response_length'],
                        'sequences_str': item['sequences_str'],
                        'data_source': item['data_source']
                    })
                except Exception as exc:
                    print(f'Item {item["index"]} generated an exception: {exc}')
                    # Fallback to default score on error
                    results.append({
                        'index': item['index'],
                        'score_dict': {'score': 0.0, 'correctness': False},
                        'valid_response_length': item['valid_response_length'],
                        'sequences_str': item['sequences_str'],
                        'data_source': item['data_source']
                    })
        
        # Sort results by original index to maintain order
        results.sort(key=lambda x: x['index'])
        return results



    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        correctness_tensor = torch.zeros(len(data), dtype=torch.float32)
        already_print_data_sources = {}

        # Prepare batch data for parallel processing
        batch_items = self._prepare_batch_data(data)
        
        # Compute scores using selected strategy
        if self.parallel_strategy == 'none':
            # Fallback to sequential processing (original behavior)
            results = self._compute_score_sequential(batch_items)
        elif self.parallel_strategy == 'simple':
            results = self._compute_score_parallel(batch_items)
        elif self.parallel_strategy == 'optimized':
            # Use simple parallel strategy instead of optimized
            results = self._compute_score_parallel(batch_items)
        else:
            raise ValueError(f"Unknown parallel strategy: {self.parallel_strategy}")

        # Process results and update tensors
        for result in results:
            i = result['index']
            score_dict = result['score_dict']
            valid_response_length = result['valid_response_length']
            sequences_str = result['sequences_str']
            data_source = result['data_source']
            
            reward_tensor[i, valid_response_length - 1] = score_dict['score']
            correctness_tensor[i] = score_dict['correctness']

            # Handle printing for examination
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        return {"reward_tensor": reward_tensor, "correctness_tensor": correctness_tensor}


print("before import ray")
import ray
import hydra
import sys

# print("before register hydra")
# @hydra.main(config_path='config', config_name='ppo_trainer', 
# version_base=None)
# def main(config):
#     print("before run_ppo")
#     run_ppo(config)
# print("before initialize hydra")

def main():
    print("inside main")
    # Use hydra.initialize and compose with command line overrides
    with hydra.initialize(config_path='config', version_base=None):
        # Parse command line arguments (skip script name)
        overrides = sys.argv[1:] if len(sys.argv) > 1 else []
        print(f"Command line overrides: {overrides}")
        config = hydra.compose(config_name='ppo_trainer', overrides=overrides)
        print("before run_ppo")
        run_ppo(config)

print("before define run_ppo")
def run_ppo(config, compute_score=None):
    print("inside run_ppo")
    if not ray.is_initialized():
        print("Initializing Ray")
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config, compute_score))
    # print("Ray initialized, about to call main_task.remote()")
    # print(f"Ray cluster status: {ray.cluster_resources()}")
    # print(f"Ray nodes: {ray.nodes()}")
    
    # # Check if there are any available resources
    # resources = ray.cluster_resources()
    # if 'CPU' not in resources or resources['CPU'] == 0:
    #     print("WARNING: No CPU resources available in Ray cluster")
    # if 'GPU' in resources and resources['GPU'] == 0:
    #     print("WARNING: No GPU resources available in Ray cluster")
    
    # # Check Ray dashboard URL
    # print(f"Ray dashboard available at: http://127.0.0.1:8265")
    
    # # Submit the remote task
    # print("Submitting remote task...")
    # task_ref = main_task.remote(config, compute_score)
    # print(f"Task submitted with ref: {task_ref}")
    
    # # Wait for the result with timeout
    # print("Waiting for task result...")
    # try:
    #     result = ray.get(task_ref, timeout=300)  # 5 minute timeout
    #     print("Task completed successfully")
    # except ray.exceptions.GetTimeoutError:
    #     print("ERROR: Task timed out after 5 minutes")
    #     print(f"Task status: {ray.get_runtime_context().get_worker_id()}")
    #     raise
    # except Exception as e:
    #     print(f"ERROR: Task failed with exception: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     raise

print("before define main_task")
@ray.remote
def main_task(config, compute_score=None):
    print("=== MAIN_TASK STARTED ===")
    logging.info("Starting main task")
    print("=== LOGGING INFO PRINTED ===")
    from verl.utils.fs import copy_local_path_from_hdfs
    # from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, compute_score=compute_score)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1, compute_score=compute_score)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    logging.info("before building trainer")
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    logging.info("after building trainer")
    logging.info("before init workers")
    trainer.init_workers()
    logging.info("after init workers")
    logging.info("before fit")
    trainer.fit()


if __name__ == '__main__':
    print("before main")
    main()
