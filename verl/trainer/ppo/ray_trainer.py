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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface

7/25: add prompt filter (borrowed from DAPO code https://github.com/volcengine/verl/blob/main/recipe/dapo/dapo_ray_trainer.py)
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from tensordict import TensorDict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto, DataProtoItem
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
import wandb
import time
import random
import json
import inspect
from collections import defaultdict


WorkerType = Type[Worker]

def dataprotoitem_to_dataproto(item: DataProtoItem) -> DataProto:
    """Convert a DataProtoItem to a DataProto object"""
    return DataProto.from_dict(
        tensors=item.batch,  # TensorDict is already in correct format
        non_tensors=item.non_tensor_batch,  # Dict is already in correct format 
        meta_info=item.meta_info
    )


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6

class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """
    GAE = 'gae'
    GRPO = 'grpo'
    REINFORCE_PLUS_PLUS = 'reinforce_plus_plus'
    REINFORCE_PLUS_PLUS_BASELINE = 'reinforce_plus_plus_baseline'
    REMAX = 'remax'
    RLOO = 'rloo'


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, prompt_reward_data=None):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        # advantages, returns, prompt_reward_data = 
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index,
                                                                        prompt_reward_data=prompt_reward_data)
        # print("advantages.size():", advantages.size())
        # print("advantages:", advantages)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
        # data.non_tensor_batch['prompt_reward_data'] = prompt_reward_data
        # {id2mean: {}, id2std: {}, id2n: {}}
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        if key == "actor/token_logprob":
            all_token_logprob = np.concatenate(val)
            metrics[key] = all_token_logprob
        elif key == "actor/pg_clipfrac_by_prompt":
            # For per-prompt data, we need to merge dictionaries from different micro-batches
            if isinstance(val, list) and len(val) > 0:
                # Merge all per-prompt dictionaries
                merged_data = {}
                for micro_batch_data in val:
                    if isinstance(micro_batch_data, dict):
                        merged_data.update(micro_batch_data)
                metrics[key] = merged_data
            # If it's already a single dict, keep it as is
        else:
            metrics[key] = np.mean(val)
    return metrics

def compute_response_mask(data: DataProto):
    responses = data.batch['responses']
    response_length = responses.size(1)
    attention_mask = data.batch['attention_mask']
    return attention_mask[:, -response_length:]

def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )
    
    
def compute_response_metrics(batch):
    max_response_length = batch.batch['responses'].shape[-1]
    response_info = _compute_response_info(batch)
    response_length = response_info['response_length']
    
    metrics = {
        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
    }
    return metrics  


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)


    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'filtered_response_length/mean':
            torch.mean(response_length).detach().item(),
        'filtered_response_length/max':
            torch.max(response_length).detach().item(),
        'filtered_response_length/min':
            torch.min(response_length).detach().item(),
        'filtered_response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_prompt_reward_stats(batch: DataProto):
    """Compute per-prompt mean/std/n over sample rewards.
    Rewards are stored as token-level scores with reward on the last token per sample.
    """
    # response_info = _compute_response_info(batch)
    sample_rewards = torch.sum(batch.batch['token_level_scores'], dim=-1).cpu().tolist()
    uids = batch.non_tensor_batch['uid']
    with torch.no_grad():
        id2values = {}
        for idx, uid in enumerate(uids):
            if uid not in id2values:
                id2values[uid] = []
            id2values[uid].append(sample_rewards[idx])

        id2mean = {}
        id2std = {}
        id2n = {}
        for uid, values in id2values.items():
            arr = np.array(values, dtype=float)
            id2mean[uid] = torch.tensor(float(arr.mean()))
            id2std[uid] = float(arr.std(ddof=0)) if arr.size > 1 else 0.0
            id2n[uid] = int(arr.size)

    return {"id2mean": id2mean, "id2std": id2std, "id2n": id2n}


def compute_timing_metrics(batch, timing_raw):
    """
    Compute timing metrics for various training operations.
    
    Returns:
        - timing_s/{operation}: Total time in seconds for each operation
        - timing_per_token_ms/{operation}: Time per token in milliseconds for token-level operations
        - timing_per_batch_ms/{operation}: Time per batch sample in milliseconds for batch-level operations
    
    Operations tracked:
        - gen: Sequence generation time
        - reward_model: Reward model inference time (if enabled)
        - reward_function: Reward function verification time
        - old_log_prob: Log probability computation time
        - ref: Reference policy computation time
        - values: Critic value computation time
        - update_critic: Critic update time
        - update_actor: Actor update time
        - testing: Validation time
        - save_checkpoint: Checkpoint saving time
    """
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }
    
    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
        # Add per-batch timing for special sections that don't operate per-token
        # **{
        #     f'timing_per_batch_ms/{name}': timing_raw[name] * 1000 / batch.batch.batch_size[0] for name in special_timing_sections if name in timing_raw
        # },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last

class RolloutBuffer:
    def __init__(self, max_use=3):
        self.buffer = None # DataProto
        self.buffer_size = 0
        self.prompt_id2buffer = defaultdict(list) # dict[str, Tuple [(index, gen_steps)]]
        self.idx2count = defaultdict(int)
        self.max_use = max_use
    
    
    def append(self, batch: DataProto, prompt_stats: dict):
        # TODO rn I keep all positive rollouts, it may cause memory issue
        keep_idx = []
        for i in range(len(batch.batch['correctness'])):
            if batch.batch['correctness'][i] == 1:
                #  and prompt_stats[batch.non_tensor_batch['uid'][i]]['mean'] < 0.5:
                keep_idx.append(i)
                prompt_id = batch.non_tensor_batch['uid'][i]
                # append new index for this prompt_id
                new_idx = self.buffer_size
                self.prompt_id2buffer[prompt_id].append(new_idx)
                self.buffer_size += 1

                # keep only the most recent 5 rollouts per prompt_id
                if len(self.prompt_id2buffer[prompt_id]) > 5:
                    # remove the oldest index from this prompt_id's buffer view
                    old_idx = self.prompt_id2buffer[prompt_id].pop(0)
                    # clean up usage counter for the retired index
                    if old_idx in self.idx2count:
                        del self.idx2count[old_idx]
        batch_to_keep = batch[keep_idx]
        if self.buffer is None:
            self.buffer = batch_to_keep
        else:
            self.buffer = DataProto.concat([self.buffer, batch_to_keep])

        # compact underlying buffer so that it only stores rollouts that are still referenced
        self._compact_buffer()
        print(f"[RolloutBuffer] buffer size after append: {self.buffer_size}")
        return batch_to_keep


    def sample(self, prompt_id: str, n_samples: int):
        if not self.prompt_id2buffer.get(prompt_id, []):
            print(f"[RolloutBuffer] no positive rollouts in buffer for prompt_id: {prompt_id}")
            return None
        # sample n_samples
        sampled_idx = list(random.sample(self.prompt_id2buffer[prompt_id], n_samples))
        print(f"[RolloutBuffer] sampled idx: {sampled_idx}")
        # convert protoitem to dataproto
        sampled_batch = self.buffer[sampled_idx]
        print(f"[RolloutBuffer] type of sampled_batch: {type(sampled_batch)}")
        # update idx2count
        for idx in sampled_idx:
            self.idx2count[idx] += 1
            if self.idx2count[idx] >= self.max_use:
                print(f"[RolloutBuffer] used for {self.idx2count[idx]} times, idx={idx} retired")
                self.prompt_id2buffer[prompt_id].remove(idx)
        print(f"[RolloutBuffer] sampled rollouts: {len(sampled_batch)} for prompt_id: {prompt_id}")
        return sampled_batch


    def _compact_buffer(self):
        """
        Rebuild self.buffer so that it only contains rollouts still referenced
        by prompt_id2buffer. This both bounds memory and keeps indices dense.
        """
        if self.buffer is None:
            return

        # all indices that are still in use
        active_indices = sorted({idx for idx_list in self.prompt_id2buffer.values() for idx in idx_list})
        if not active_indices:
            # nothing is referenced anymore; reset everything
            self.buffer = None
            self.buffer_size = 0
            self.idx2count.clear()
            self.prompt_id2buffer.clear()
            return

        # build compacted buffer
        new_buffer = self.buffer[active_indices]
        # map from old index to new, dense index
        old2new = {old_idx: new_idx for new_idx, old_idx in enumerate(active_indices)}

        # remap prompt_id2buffer to the new dense indices
        for prompt_id, idx_list in list(self.prompt_id2buffer.items()):
            new_list = [old2new[idx] for idx in idx_list if idx in old2new]
            if new_list:
                self.prompt_id2buffer[prompt_id] = new_list
            else:
                # remove prompt_ids that no longer have any rollouts
                del self.prompt_id2buffer[prompt_id]

        # remap idx2count to the new dense indices
        new_idx2count = defaultdict(int)
        for old_idx, count in self.idx2count.items():
            if old_idx in old2new:
                new_idx2count[old2new[old_idx]] = count
        self.idx2count = new_idx2count

        self.buffer = new_buffer
        self.buffer_size = len(active_indices)
        assert self.buffer_size == len(self.buffer), f"buffer size mismatch: {self.buffer_size} != {len(self.buffer)}"


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        if self.config.algorithm.adv_estimator == 'gae':
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader()
        self.rollout_buffer = RolloutBuffer()
    
    

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.micro_batch_size' or "
                                 f"'{name}.micro_batch_size_per_gpu'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(f"[{name}] You have set both '{name}.micro_batch_size' AND "
                                 f"'{name}.micro_batch_size_per_gpu'. Please remove '{name}.micro_batch_size' "
                                 f"because only '*_micro_batch_size_per_gpu' is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu,
                                     "reward_model")

        # Actor
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           drop_last=True,
                                           collate_fn=collate_fn,
                                           sampler=sampler)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=len(self.val_dataset),
                                         shuffle=True,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _validate(self):
        correctness_lst = []
        reward_tensor_lst = []
        data_source_lst = []
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            # test_batch = test_batch.to('cuda')

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }
            test_gen_batch.non_tensor_batch['n'] = np.array([1] * len(test_gen_batch.batch['input_ids']),dtype=object)

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            # TODO: missing "n" in test_gen_batch_padded
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            # print('validation generation end')

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            # for certain reward function (e.g. sandbox), the generation can overlap with reward
            # check if  return_dict is an argument in the reward_fn
            if 'return_dict' in inspect.signature(self.val_reward_fn).parameters:
                reward_tensor_dict = self.val_reward_fn(test_batch, return_dict=True)
            else:
                reward_tensor_dict = self.val_reward_fn(test_batch)
            # reward_tensor_dict = self.val_reward_fn(test_batch)
            
            reward_tensor = reward_tensor_dict['reward_tensor'] 
            correctness = reward_tensor_dict['correctness_tensor']
            
            reward_tensor_lst.append(reward_tensor)
            correctness_lst.append(correctness)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        correctness_tensor = torch.cat(correctness_lst, dim=0).cpu()
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}
        data_source_correctness = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
                data_source_correctness[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())
            data_source_correctness[data_source].append(correctness_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

        for data_source, correctnesses in data_source_correctness.items():
            metric_dict[f'val/test_correctness/{data_source}'] = np.mean(correctnesses)

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, remove_previous_ckpt=self.config.trainer.remove_previous_ckpt)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, remove_previous_ckpt=self.config.trainer.remove_previous_ckpt)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        import dill
        torch.save(self.train_dataloader, dataloader_local_path, pickle_module=dill)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))
        
        # gen_step save
        gen_step_local_path = os.path.join(self.config.trainer.default_local_dir, 'gen_step.txt')
        with open(gen_step_local_path, 'w') as f:
            f.write(str(self.gen_steps))
            
    # # 新增过滤方法（需在类中定义）
    # def _filter_batch(self, batch, mask: np.ndarray) -> DataProto:
    #     """根据布尔掩码过滤批次数据"""
    #     mask_tensor = torch.from_numpy(mask).to(batch.batch.device)
    
    #     # 过滤张量数据
    #     filtered_tensors = {
    #         k: v[mask_tensor] for k, v in batch.batch.items()
    #     }
    
    #     # 过滤非张量数据（如果有）
    #     filtered_non_tensors = {
    #         k: [x for x, m in zip(v, mask) if m]
    #         for k, v in batch.non_tensor_batch.items()
    #     }
    
    #     return DataProto(
    #         batch=TensorDict(filtered_tensors, batch_size=mask.sum()),
    #         non_tensor_batch=filtered_non_tensors,
    #         meta_info=batch.meta_info
    #     )
    def _filter_batch(self, batch, mask) -> DataProto:
        """根据布尔掩码过滤批次数据"""
        mask_tensor = mask.to(batch.batch.device)
        # mask_tensor = torch.from_numpy(mask).to(batch.batch.device)
    
        # 过滤张量数据
        filtered_tensors = {
            k: v[mask_tensor == 1] for k, v in batch.batch.items()
        }
    
        # ==== 修复点：保持 non_tensor_batch 为 NumPy 数组 ====
        filtered_non_tensors = {
            k: v[mask.numpy() == 1]  # 直接使用 NumPy 布尔索引（保持数组类型）
            for k, v in batch.non_tensor_batch.items()
        }
    
        return DataProto(
            batch=TensorDict(filtered_tensors, batch_size=int(mask_tensor.sum().item())),
            non_tensor_batch=filtered_non_tensors,
            meta_info=batch.meta_info
        )


    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if not (self.config.trainer.resume_from_path and global_step_folder is not None):
                assert isinstance(self.config.trainer.resume_mode, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_mode, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])
        # gen_step load
        gen_step_local_path = os.path.join(self.config.trainer.default_local_dir, 'gen_step.txt')
        with open(gen_step_local_path, 'r') as f:
            self.gen_steps = int(f.read())

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path)

        # load dataloader,
        # TODO: from remote not implemented yet
        if self.config.trainer.resume_data_state:
            dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
            self.train_dataloader = torch.load(dataloader_local_path, weights_only=False)
            if isinstance(self.train_dataloader.dataset, RLHFDataset):
                self.train_dataloader.dataset.resume_dataset_state()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)
    
    def _filter_std_0(self, new_batch: DataProto, prompt_reward_data: dict):
        if not self.config.data.keep_all_negative:
            prompt_ids_to_keep = [pid for pid, std in prompt_reward_data['id2std'].items() if std > 0]
            rollout_idx_to_keep = []
            for idx, prompt_id in enumerate(new_batch.non_tensor_batch['uid']):
                if prompt_id in prompt_ids_to_keep:
                    rollout_idx_to_keep.append(idx)
            new_batch = new_batch[rollout_idx_to_keep]
            print("[BATCH] batch length after filter for std > 0: ", len(new_batch))
        else:
            # filter all positive but keep all negative
            prompt_ids_to_keep = [pid for pid, std in prompt_reward_data['id2std'].items() if std > 0 or prompt_reward_data['id2mean'][pid] == 0.0]
            rollout_idx_to_keep = []
            for idx, prompt_id in enumerate(new_batch.non_tensor_batch['uid']):
                if prompt_id in prompt_ids_to_keep:
                    rollout_idx_to_keep.append(idx)
            new_batch = new_batch[rollout_idx_to_keep]
            print("[BATCH] batch length after filter for std > 0 or mean == 0.0: ", len(new_batch))
        return new_batch

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        os.makedirs(self.config.trainer.default_local_dir, exist_ok=True)

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        self.gen_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        from tqdm import tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.gen_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1

        batch = None
        batch_remain = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        stats_table = wandb.Table(columns=["step", "prompt_id", "mean_reward", "std_reward", "n", "processed"])
        clipfrac_stats_table = wandb.Table(columns=["step", "prompt_id", "correct_clipfrac", "incorrect_clipfrac", "correct_count", "incorrect_count"])
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1

                gen_batch = new_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                is_last_step = self.gen_steps >= self.total_training_steps

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    # TODO: handle uid
                    if new_batch.non_tensor_batch['prompt_id'][0] is not None:
                        new_batch.non_tensor_batch['uid'] = new_batch.non_tensor_batch['prompt_id']
                    else:
                        new_batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(new_batch.batch))],
                                                             dtype=object)
                    new_batch.batch['instance_id'] = torch.tensor([i for i in range(len(new_batch.batch))])
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n if self.config.actor_rollout_ref.rollout.oversample_n == -1 else self.config.actor_rollout_ref.rollout.oversample_n, interleave=True)
                    # print("print batch here:", batch)
                    print("gen batch size:", len(gen_batch_output))
                    print("new batch size:", len(new_batch))
                    new_batch = new_batch.union(gen_batch_output)

                    

                    with _timer('adv', timing_raw):
                        # Step 1: compute scores via reward model (if enabled), then verifier-only reward function
                        if self.use_rm:
                            with _timer('reward_model', timing_raw):
                                reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                                new_batch = new_batch.union(reward_tensor)

                        with _timer('reward_function', timing_raw):
                            reward_tensor_dict: dict = self.reward_fn(new_batch)
                            new_batch.batch['token_level_scores'] = reward_tensor_dict['reward_tensor']
                            new_batch.batch['correctness'] = reward_tensor_dict['correctness_tensor']
                        
                        


                        # Step 2: compute per-prompt stats (mean/std/n)
                        prompt_reward_data = compute_prompt_reward_stats(new_batch)
                        for prompt_id in prompt_reward_data["id2mean"]:
                            stats_table.add_data(
                                self.global_steps,
                                prompt_id,
                                prompt_reward_data["id2mean"][prompt_id].item(),
                                prompt_reward_data["id2std"][prompt_id],
                                prompt_reward_data["id2n"][prompt_id],
                                False
                            )
                        metrics.update({"actor/prompt_reward_stats": stats_table})

                        # =====================================
                        # store in rollout buffer
                        # =====================================
                        if self.config.data.replay:
                            self.rollout_buffer.append(new_batch, prompt_reward_data)
                        
                        # ===========================
                        # replay
                        # ===========================
                        if self.config.data.replay:
                            # replay for prompt without positive rollouts, randomly replace existing negative rollouts
                            # arg: replay_sample_n
                            # 1. collect all prompt_id with mean == 0.0
                            prompt_id_to_replay = []
                            for prompt_id in prompt_reward_data["id2mean"]:
                                if prompt_reward_data["id2mean"][prompt_id] == 0.0:
                                    prompt_id_to_replay.append(prompt_id)
                            # 2. randomly replace existing negative rollouts
                            for prompt_id in prompt_id_to_replay:
                                sampled_batch = self.rollout_buffer.sample(prompt_id, self.config.data.replay_n)
                                if sampled_batch is None:
                                    continue
                                # get index of negative rollouts
                                negative_idx = []
                                for i in range(len(new_batch)):
                                    if new_batch.non_tensor_batch['uid'][i] == prompt_id:
                                        negative_idx.append(i)
                                selected_negative_idx = np.random.choice(negative_idx, size=self.config.data.replay_n, replace=False)
                                for i, idx in enumerate(selected_negative_idx):
                                    new_batch[idx] = sampled_batch[i]
                            
                            # recompute prompt reward
                            prompt_reward_data = compute_prompt_reward_stats(new_batch)

                                
                        
                        # Step 4: then filter prompts with std > 0, after this step, all prompts should have at least one positive rollout and at least one negative rollout
                        new_batch = self._filter_std_0(new_batch, prompt_reward_data)

                        #     print("[BATCH] batch length after filter for std > 0: ", len(new_batch))
                            
                        # ===============================================
                        # Step 3.2: Downsample and make sure the same number of rollout appears for each designated prompt
                        # ===============================================
                        if hasattr(self.config.actor_rollout_ref.rollout, 'oversample_n') and self.config.actor_rollout_ref.rollout.oversample_n > self.config.actor_rollout_ref.rollout.n:
                            print("=== trigger downsampling ===")
                            print("oversample_n:", self.config.actor_rollout_ref.rollout.oversample_n)
                            print("n:", self.config.actor_rollout_ref.rollout.n)
                            print("=== trigger downsampling ===")
                            new_batch = self._downsample(new_batch, n_rollout=self.config.actor_rollout_ref.rollout.n)
                            print("[BATCH] batch length after downsample: ", len(new_batch))

                            if not self.config.data.keep_old_advantage_before_downsample:
                                prompt_reward_data = compute_prompt_reward_stats(new_batch)# do filter again
                                # new_batch = self._filter_std_0(new_batch, prompt_reward_data)
                                # prompt_ids_to_keep = [pid for pid, std in prompt_reward_data['id2std'].items() if std > 0]
                                # num_prompt_in_batch += len(prompt_ids_to_keep)
                                # rollout_idx_to_keep = []
                                # for idx, prompt_id in enumerate(new_batch.non_tensor_batch['uid']):
                                #     if prompt_id in prompt_ids_to_keep:
                                #         rollout_idx_to_keep.append(idx)
                                # new_batch = new_batch[rollout_idx_to_keep]
                                # print("[BATCH] batch length after filter again: ", len(new_batch))

                            for prompt_id in prompt_reward_data["id2mean"]:
                                stats_table.add_data(
                                    self.global_steps,
                                    prompt_id,
                                    prompt_reward_data["id2mean"][prompt_id].item(),
                                    prompt_reward_data["id2std"][prompt_id],
                                    prompt_reward_data["id2n"][prompt_id],
                                    True
                                )
                            metrics.update({"actor/prompt_reward_stats": stats_table})

                                
                        

                        

                        # Step 5: compute old_log_probs (needed for diversity filtering)
                        # any rollout-level discard should be done after this step due to distributed log_prob computation. the rollout_bsz must be dividable by the number of gpus.
                        with _timer('old_log_prob', timing_raw):
                            output = self.actor_rollout_wg.compute_log_prob(new_batch)
                            new_batch = new_batch.union(output)
                        
                        if "last_token_hidden_states" in new_batch.batch:
                            _ = new_batch.batch.pop("last_token_hidden_states")

                        # Step 7: set rewards with KL penalty if configured, then compute advantages on the filtered set
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(new_batch, kl_ctrl=self.kl_ctrl, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                        new_batch = compute_advantage(new_batch,
                                                      adv_estimator=self.config.algorithm.adv_estimator,
                                                      gamma=self.config.algorithm.gamma,
                                                      lam=self.config.algorithm.lam,
                                                      num_repeat=self.config.actor_rollout_ref.rollout.n,
                                                      prompt_reward_data=prompt_reward_data)

                        print(f"new_batch length: {len(new_batch)}")
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        # check batch size match
                        current_bsz = len(batch)
                        rollout_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n

                        if current_bsz < rollout_bsz:
                            print(f"{current_bsz=} < {rollout_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                progress_bar.update(1)
                                self.gen_steps += 1
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            batch_remain = batch[rollout_bsz:] # TODO: save remainig batch does not waste rollout but may cause off-policy issue. in DAPO original implememntation, it is not saved.
                            batch = batch[:rollout_bsz]

                    
                    
                    


                    response_info = _compute_response_info(batch)
                    response_lengths = response_info['response_length']
                    max_len = batch.batch['responses'].shape[-1]  # 获取当前批次的最大响应长度
                    # ===== 核心修改：按样本整体屏蔽 =====
                    # 生成样本级掩码（True表示需要屏蔽）
                    sample_mask = (response_lengths >= max_len)  # [batch_size]
                    

                    adjusted_attention_mask = batch.batch['attention_mask'].clone()
                    for i, mask in enumerate(sample_mask):
                        if mask:
                            adjusted_attention_mask[i, -max_len:] = 0  # 将响应部分掩码置零
                    
                    metrics.update(compute_response_metrics(batch=batch))
                    
                    if self.config.trainer.remove_clip:
                        batch.batch['attention_mask'] = adjusted_attention_mask
                    


                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    

                    self._balance_batch(batch, metrics=metrics)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        if self.global_steps % self.config.trainer.save_freq == 0:
                            actor_output_metrics['actor/token_logprob'] = wandb.Histogram(actor_output_metrics['actor/token_logprob'])
                        else:
                            actor_output_metrics.pop('actor/token_logprob')
                        
                        # Collect per-prompt clipping fraction data
                        if 'actor/pg_clipfrac_by_prompt' in actor_output_metrics:
                            pg_clipfrac_by_prompt = actor_output_metrics.pop('actor/pg_clipfrac_by_prompt')
                            if pg_clipfrac_by_prompt is not None:
                                for prompt_id, clipfrac_data in pg_clipfrac_by_prompt.items():
                                    clipfrac_stats_table.add_data(
                                        self.global_steps,
                                        prompt_id,
                                        clipfrac_data['correct_clipfrac'],
                                        clipfrac_data['incorrect_clipfrac'],
                                        clipfrac_data['correct_count'],
                                        clipfrac_data['incorrect_count']
                                    )
                                metrics.update({"actor/clipfrac_stats": clipfrac_stats_table})
                                print(f"Added {len(pg_clipfrac_by_prompt)} prompts to clipfrac_stats_table")
                        
                        metrics.update(actor_output_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()
                


                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                
                # Add efficiency metrics for reward operations
                if 'timing_s/step' in metrics:
                    step_time = metrics['timing_s/step']
                    reward_ops_time = 0
                    if 'timing_s/reward_function' in metrics:
                        reward_ops_time += metrics['timing_s/reward_function']
                    if 'timing_s/reward_model' in metrics:
                        reward_ops_time += metrics['timing_s/reward_model']

                    if step_time > 0:
                        metrics['efficiency/reward_ops_time_ratio'] = reward_ops_time / step_time
                        metrics['efficiency/reward_ops_time_percentage'] = (reward_ops_time / step_time) * 100

                # Log timing summary for debugging
                if self.global_steps % 10 == 0:  # Log every 10 steps to avoid spam
                    timing_summary = {k: v for k, v in metrics.items() if k.startswith('timing_s/')}
                    print(f"Step {self.global_steps} Timing Summary: {timing_summary}")

                    # Log specific timing for reward-related operations
                    reward_timing = {}
                    if 'timing_s/reward_function' in metrics:
                        reward_timing['reward_function'] = metrics['timing_s/reward_function']
                    if 'timing_s/reward_model' in metrics:
                        reward_timing['reward_model'] = metrics['timing_s/reward_model']

                    if reward_timing:
                        print(f"Step {self.global_steps} Reward Operations Timing: {reward_timing}")

                # TODO: make a canonical logger that supports various backend
                metrics["train/num_gen_batches"] = num_gen_batches

                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0
                if batch_remain is not None and len(batch_remain) > 0:
                    batch = batch_remain
                    batch_remain = None
                logger.log(data=metrics, step=self.global_steps)

                # save stats to csv
                table = metrics["actor/prompt_reward_stats"]
                stats_df = table.get_dataframe()

                os.makedirs(self.config.trainer.default_local_dir, exist_ok=True)
                output_file = os.path.join(self.config.trainer.default_local_dir, "prompt_reward_stats.csv")
                if not os.path.exists(output_file):
                    stats_df.to_csv(output_file, index=False)
                else:
                    with open(output_file, "a+") as f:
                        # write  each row to csv
                        for index, row in stats_df.iterrows():
                            f.write(f"{row['step']},{row['prompt_id']},{row['mean_reward']},{row['std_reward']},{row['n']},{row['processed']}\n")

                # save clipfrac stats to csv
                if "actor/clipfrac_stats" in metrics:
                    clipfrac_table = metrics["actor/clipfrac_stats"]
                    clipfrac_df = clipfrac_table.get_dataframe()

                    clipfrac_output_file = os.path.join(self.config.trainer.default_local_dir, "prompt_clipfrac_stats.csv")
                    if not os.path.exists(clipfrac_output_file):
                        clipfrac_df.to_csv(clipfrac_output_file, index=False)
                        print(f"Created new clipfrac stats file: {clipfrac_output_file}")
                    else:
                        with open(clipfrac_output_file, "a+") as f:
                            # write each row to csv
                            for index, row in clipfrac_df.iterrows():
                                f.write(f"{row['step']},{row['prompt_id']},{row['correct_clipfrac']},{row['incorrect_clipfrac']},{row['correct_count']},{row['incorrect_count']}\n")
                        print(f"Appended {len(clipfrac_df)} clipfrac records to: {clipfrac_output_file}")
                else:
                    print(f"Step {self.global_steps}: No clipfrac stats found in metrics")



                stats_table = wandb.Table(columns=["step", "prompt_id", "mean_reward", "std_reward", "n", "processed"])
                clipfrac_stats_table = wandb.Table(columns=["step", "prompt_id", "correct_clipfrac", "incorrect_clipfrac", "correct_count", "incorrect_count"])

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1

                # if self.global_steps >= self.total_training_steps:

                #     # perform validation after training
                #     if self.val_reward_fn is not None:
                #         val_metrics = self._validate()
                #         pprint(f'Final validation metrics: {val_metrics}')
                #         logger.log(data=val_metrics, step=self.global_steps)
                #     return
    
    
    def _downsample(self, batch: DataProto, n_rollout: int):
        """
        Downsample the batch so each uid keeps at most n_rollout rollouts, sampled uniformly at random.
        """
        uid2idx = defaultdict(list)
        for idx, uid in enumerate(batch.non_tensor_batch['uid']):
            uid2idx[uid].append(idx)

        keep_idx = []
        for uid, idx_list in uid2idx.items():
            idx_tensor = torch.tensor(idx_list)
            idx_tensor = idx_tensor[torch.randperm(len(idx_tensor))]
            idx_tensor = idx_tensor[:n_rollout]
            keep_idx.extend(idx_tensor.tolist())
        keep_idx = torch.tensor(keep_idx).sort()[0]
        return batch[keep_idx]
        
        
