# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(config):
    if config.critic.kl_ctrl.type == 'fixed':
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == 'adaptive':
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       horizon=config.critic.kl_ctrl.horizon)
    else:
        raise ValueError('Unknown kl_ctrl type')

    return kl_ctrl


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
# def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
#                                    eos_mask: torch.Tensor,
#                                    index: torch.Tensor,
#                                    epsilon: float = 1e-6):
#     """
#     Compute advantage for GRPO, operating only on Outcome reward 
#     (with only one scalar reward for each response).
#     Args:
#         token_level_rewards: `(torch.Tensor)`
#             shape: (bs, response_length)
#         eos_mask: `(torch.Tensor)`
#             shape: (bs, response_length)
    
#     Returns:
#         advantages: `(torch.Tensor)`
#             shape: (bs, response_length)
#         Returns: `(torch.Tensor)`
#             shape: (bs, response_length)
#     """
#     response_length = token_level_rewards.shape[-1]
#     scores = token_level_rewards.sum(dim=-1)

#     id2score = defaultdict(list)
#     id2mean = {}
#     id2std = {}

#     with torch.no_grad():
#         bsz = scores.shape[0]
#         for i in range(bsz):
#             id2score[index[i]].append(scores[i])
#         for idx in id2score:
#             if len(id2score[idx]) == 1:
#                 id2mean[idx] = torch.tensor(0.0)
#                 id2std[idx] = torch.tensor(1.0)
#             elif len(id2score[idx]) > 1:
#                 id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
#                 id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
#             else:
#                 raise ValueError(f"no score in prompt index: {idx}")
#         for i in range(bsz):
#             scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
#         scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

#     return scores, scores


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio

def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.
    Args:
        loss_mat: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_agg_mode: (str) choices: "token-mean" / "seq-mean-token-sum" / "seq-mean-token-mean"
            "token-mean" is the default behavior
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss

def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange, low_clip_ratio, high_clip_ratio, 
                        correctness=None, instance_id=None, total_token_count=None):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        correctness: `(torch.Tensor)`, optional
            shape: (bs,) - binary tensor indicating correctness of each rollout
        instance_id: `(torch.Tensor)`, optional
            shape: (bs,) - instance IDs for grouping rollouts by prompt
            
    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped
        pg_clipfrac_by_prompt: `dict`, optional
            Dictionary containing clipping fractions per prompt for correct/incorrect rollouts

    """
    masked_mean = verl_F.masked_mean
    # masked_mean = verl_F.masked_mean_drgrpo




    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = masked_mean(-negative_approx_kl, eos_mask, total_token_count=total_token_count)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - low_clip_ratio, 1.0 + high_clip_ratio)

    pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), eos_mask, total_token_count=total_token_count)
    pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask, total_token_count=total_token_count)
    
    # Calculate per-prompt clipping fractions if correctness and instance_id are provided
    pg_clipfrac_by_prompt = None
    if correctness is not None and instance_id is not None:
        pg_clipfrac_by_prompt = compute_per_prompt_clipfrac(
            pg_losses, pg_losses2, eos_mask, correctness, instance_id
        )
    
    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_by_prompt


def compute_per_prompt_clipfrac(pg_losses, pg_losses2, eos_mask, correctness, instance_id):
    """Compute clipping fractions per prompt for correct and incorrect rollouts separately.
    
    Args:
        pg_losses: `(torch.Tensor)`
            shape: (bs, response_length) - unclipped policy gradient losses
        pg_losses2: `(torch.Tensor)`
            shape: (bs, response_length) - clipped policy gradient losses
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length) - mask for valid tokens
        correctness: `(torch.Tensor)`
            shape: (bs,) - binary tensor indicating correctness of each rollout
        instance_id: `(torch.Tensor)`
            shape: (bs,) - instance IDs for grouping rollouts by prompt
            
    Returns:
        dict: Dictionary with structure:
            {
                prompt_id: {
                    'correct_clipfrac': float,
                    'incorrect_clipfrac': float,
                    'correct_count': int,
                    'incorrect_count': int
                }
            }
    """
    # Compute which tokens are clipped
    is_clipped = torch.gt(pg_losses2, pg_losses).float()  # (bs, response_length)
    
    # Get unique instance IDs
    unique_instance_ids = torch.unique(instance_id)
    
    result = {}
    
    for prompt_id in unique_instance_ids:
        # Find rollouts for this prompt
        prompt_mask = (instance_id == prompt_id)
        
        if prompt_mask.sum() == 0:
            continue
            
        # Separate correct and incorrect rollouts for this prompt
        correct_mask = prompt_mask & (correctness == 1)
        incorrect_mask = prompt_mask & (correctness == 0)
        
        prompt_result = {
            'correct_clipfrac': 0.0,
            'incorrect_clipfrac': 0.0,
            'correct_count': 0,
            'incorrect_count': 0
        }
        
        # Calculate clipping fraction for correct rollouts
        if correct_mask.sum() > 0:
            correct_eos_mask = eos_mask[correct_mask]
            correct_is_clipped = is_clipped[correct_mask]
            
            if correct_eos_mask.sum() > 0:
                correct_clipfrac = verl_F.masked_mean(correct_is_clipped, correct_eos_mask)
                prompt_result['correct_clipfrac'] = correct_clipfrac.item()
                prompt_result['correct_count'] = correct_mask.sum().item()
        
        # Calculate clipping fraction for incorrect rollouts
        if incorrect_mask.sum() > 0:
            incorrect_eos_mask = eos_mask[incorrect_mask]
            incorrect_is_clipped = is_clipped[incorrect_mask]
            
            if incorrect_eos_mask.sum() > 0:
                incorrect_clipfrac = verl_F.masked_mean(incorrect_is_clipped, incorrect_eos_mask)
                prompt_result['incorrect_clipfrac'] = incorrect_clipfrac.item()
                prompt_result['incorrect_count'] = incorrect_mask.sum().item()
        
        result[prompt_id.item()] = prompt_result
    
    return result


def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == 'low_var_kl':
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6,
                                   prompt_reward_data=None):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    id2n = {}
    bsz = scores.shape[0]
    if prompt_reward_data is None:
        with torch.no_grad():
            for i in range(bsz):
                id2score[index[i]].append(scores[i])
            for idx in id2score:
                id2n[idx] = len(id2score[idx])
                if len(id2score[idx]) == 1:
                    id2mean[idx] = torch.tensor(0.0)
                    id2std[idx] = torch.tensor(1.0)
                elif len(id2score[idx]) > 1:
                    id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                    id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
                else:
                    raise ValueError(f"no score in prompt index: {idx}")
    else:
        id2mean = prompt_reward_data["id2mean"]
        id2std = prompt_reward_data["id2std"]
        id2n = prompt_reward_data["id2n"]

    for i in range(bsz):
        if id2std[index[i]] == 0.0:
            assert id2mean[index[i]] == 0.0, f"id2mean[index[i]] is not 0.0 when id2std[index[i]] is 0.0. there must be something wrong with filtering."
            scores[i] = 0.0
        else:
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
    scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores
    # , {"id2mean": id2mean, "id2std": id2std, "id2n": id2n}


class InstanceLinearEntropyController:
    """
    Adaptive entropy loss coefficient controller based on average score per instance
    larger score corresponds to greater entropy loss
    Details:
    en_coef = max(beta * pass@1_score, min_en_coef)

    """

    def __init__(self, beta, min_en_coef):
        self.id2coef = {}
        self.beta = beta
        self.min_coef = min_en_coef

    
    def get_entropy_coef(self, instance_idx: torch.Tensor):
        device = instance_idx.device
        instance_idx = instance_idx.cpu().tolist()
        return torch.tensor([self.id2coef[idx] for idx in instance_idx], device=device)

    def update(self, index: torch.Tensor, correctness_tensor: torch.Tensor):
        index = index.cpu().tolist()
        self.id2coef = {}
        id2score = defaultdict(list)
        bsz = len(correctness_tensor)
        for i in range(bsz):
            id2score[index[i]].append(correctness_tensor[i])
        for idx in id2score:
            id2score[idx] = torch.mean(torch.tensor(id2score[idx])).item()
        for idx in id2score:
            self.id2coef[idx] = max(self.min_coef, self.beta * id2score[idx])
        return self.id2coef
    
class InstancePowerEntropyController:
    """
    Adaptive entropy loss coefficient controller based on average score per instance
    larger score corresponds to greater entropy loss
    
    Details:

    """
    def __init__(self, beta, min_en_coef, power=2):
        self.id2coef = {}
        self.beta = beta
        self.min_coef = min_en_coef
        self.power = power
    
    def get_entropy_coef(self, instance_idx: torch.Tensor):
        device = instance_idx.device
        instance_idx = instance_idx.cpu().tolist()
        return torch.tensor([self.id2coef[idx] for idx in instance_idx], device=device)
    
    def update(self, index: torch.Tensor, correctness_tensor: torch.Tensor):
        index = index.cpu().tolist()
        self.id2coef = {}
        id2score = defaultdict(list)
        bsz = len(correctness_tensor)
        for i in range(bsz):
            id2score[index[i]].append(correctness_tensor[i])
        for idx in id2score:
            id2score[idx] = torch.mean(torch.tensor(id2score[idx])).item()
        for idx in id2score:
            self.id2coef[idx] = max(self.min_coef, self.beta * id2score[idx] ** self.power)
    


class FixedEntropyController:
    """Fixed KL controller."""

    def __init__(self, en_coef):
        self.value = en_coef

    def update(self, index: torch.Tensor, correctness_tensor: torch.Tensor):
        pass

    def get_entropy_coef(self, instance_idx):
        return self.value


def get_entropy_controller(config):
    if config.en_ctrl.type == 'fixed':
        en_ctrl = FixedEntropyController(en_coef=config.en_ctrl.en_coef)
    elif config.en_ctrl.type == 'instance_linear':
        en_ctrl = InstanceLinearEntropyController(beta=config.en_ctrl.beta,
                                                  min_en_coef=config.en_ctrl.min_en_coef)
    elif config.en_ctrl.type == 'instance_power':
        en_ctrl = InstancePowerEntropyController(beta=config.en_ctrl.beta,
                                                  min_en_coef=config.en_ctrl.min_en_coef)
    else:
        raise ValueError('Unknown en_ctrl type')

    return en_ctrl