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
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis
import wandb

__all__ = ['DataParallelPPOActor']


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)
        # self.init_entropy_controller()
    
    def init_entropy_controller(self):
        self.en_ctrl = core_algos.get_entropy_controller(self.config)
    
    def _forward_micro_batch_w_hidden_states(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            last_token_hidden_states: # (bs, hidden_size)
        """
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    raise NotImplementedError("ULYSSES SP is not supported for this function")
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           use_cache=False,
                                           output_hidden_states=True)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # save memory
                last_layer_hidden_states = output.hidden_states[-1].cpu()

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    raise NotImplementedError("ULYSSES SP is not supported for this function")
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                # print("before pad log_probs.size():", log_probs.size())
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)
                # print("after pad full_log_probs.size():", full_log_probs.size())
                # pad in cpu to save memory
                # TODO
                # print("before pad last_layer_hidden_states.size():", last_layer_hidden_states.size())
                full_last_layer_hidden_states = pad_input(hidden_states=last_layer_hidden_states.squeeze(0).cpu(),
                                                          indices=indices.cpu(),
                                                          batch=batch_size,
                                                          seqlen=seqlen)
                # print("full_last_layer_hidden_states.size():", full_last_layer_hidden_states.size())

                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)

                # take last non-zero index in attention_mask
                seq_len = attention_mask.size(1)
                # print("attention_mask.size():", attention_mask.size())
                # print("attention_mask:", attention_mask)
                indices = torch.arange(seq_len, device=attention_mask.device).expand_as(attention_mask)
                # print("indices.size():", indices.size())
                # print("indices:", indices)
                last_indices = torch.where(attention_mask==1, indices, -1).max(dim=1)[0]
                # print("last_indices.size():", last_indices.size())
                # print("last_indices:", last_indices)

                # full_last_layer_hidden_states = full_last_layer_hidden_states.squeeze(0)
                # print("full_last_layer_hidden_states.size():", full_last_layer_hidden_states.size())
                batch_indices = torch.arange(full_last_layer_hidden_states.size(0))
                last_token_hidden_states = full_last_layer_hidden_states[batch_indices, last_indices.cpu()]  # (bsz, response_length)
                # put it back to gpu
                last_token_hidden_states = last_token_hidden_states.to(log_probs.device)

            else:  # not using rmpad and no ulysses sp
                raise NotImplementedError("ULYSSES SP is not supported for this function")
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs, last_token_hidden_states

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        # self.actor_optimizer.step()
        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm): 
            print(f"WARN: grad_norm is not finite: {grad_norm}, skip the update")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        last_token_hidden_states_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                _, log_probs, last_token_hidden_states = self._forward_micro_batch_w_hidden_states(micro_batch, temperature=temperature)
                # print("last_token_hidden_states.size():", last_token_hidden_states.size())
                log_probs_lst.append(log_probs)
                last_token_hidden_states_lst.append(last_token_hidden_states)
        log_probs = torch.concat(log_probs_lst, dim=0)
        last_token_hidden_states = torch.concat(last_token_hidden_states_lst, dim=0) # (bs, hs)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            last_token_hidden_states = last_token_hidden_states[revert_indices]

        return log_probs, last_token_hidden_states

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages','instance_id']
        #####
        if "loss_mask" in data.batch.keys():
            select_keys.append("loss_mask")
        #####
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        # Add correctness for per-prompt clipping analysis
        if "correctness" in data.batch.keys():
            select_keys.append("correctness")
        batch = data.select(batch_keys=select_keys).batch
        # uid = data.non_tensor_batch['uid']
        uid = batch['instance_id']

        # update entropy controller
        # self.en_ctrl.update(uid, data.batch['correctness'])
        # print("id2coef:", self.en_ctrl.id2coef)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

            self.actor_optimizer.zero_grad()

            # valid_batches = []
            # for data in micro_batches:
            #     attention_mask = data["loss_mask"]
            #     if attention_mask.sum().item() > 0:
            #         # valid_batches.append(data.cuda())
            #         valid_batches.append(data)

            #     else:
            #         print("no effective token, skip this micro-batch")
                    
            # print(f"valid_batches: {len(valid_batches)}")
            # del micro_batches
            # torch.cuda.empty_cache()
            # # free cpu memory
            # import gc
            # gc.collect()


            for j, data in enumerate(micro_batches):
                if "loss_mask" in data.keys() and data["loss_mask"].sum().item() == 0:
                    print("no effective token, skip this micro-batch")
                    continue
                # print(f"process {j}th micro-batch")

                # data = data.cuda()  # actor device is cpu when using offload
                responses = data['responses']
                response_length = responses.size(1)
                total_token_count = response_length * responses.size(0) # total number of tokens in this micro-batch
                #####
                # assert "loss_mask" in data.keys()
                attention_mask = data['attention_mask'] if "loss_mask" not in data.keys() else data['loss_mask']
                # attention_mask = data["loss_mask"]
                #####
                # if attention_mask.sum().item() == 0:
                #     print("no effective token, skip this micro-batch")
                #     # no effective token, skip this micro-batch
                #     torch.cuda.empty_cache()
                #     continue
                response_mask = attention_mask[:, -response_length:]
                old_log_prob = data['old_log_probs']
                advantages = data['advantages']
                micro_batch_uid = data['instance_id']
                # micro_batch_uid = data.non_tensor_batch['uid']

                clip_ratio = self.config.clip_ratio
                low_clip_ratio = self.config.low_clip_ratio
                high_clip_ratio = self.config.high_clip_ratio
                # compute policy loss
                # entropy_coeff = self.config.entropy_coeff
                entropy_coeff = self.en_ctrl.get_entropy_coef(micro_batch_uid)
                # entropy_coeff = self.config.entropy_coeff

                # all return: (bsz, response_length)
                entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

                if response_mask.sum() == 0:
                    print("no effective token after masking lowprob token, skip this micro-batch")
                    del log_prob, entropy, old_log_prob, advantages, response_mask, micro_batch_uid, data
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    continue

                # check no nan element in entropy and log_prob
                # print("entropy.size():", entropy.size())
                # print("entropy:", entropy)
                skip_flag = False
                if torch.isnan(entropy).any():
                    print("entropy contains nan!!")
                    skip_flag = True
                
                if torch.isnan(log_prob).any():
                    print("log_prob contains nan!!")
                    skip_flag = True
                
                if skip_flag:
                    del log_prob, entropy, old_log_prob, advantages, response_mask, micro_batch_uid, data
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    continue
                
                assert not torch.isnan(entropy).any(), "entropy contains nan"
                # print("log_prob.size():", log_prob.size())
                # print("log_prob:", log_prob)
                assert not torch.isnan(log_prob).any(), "log_prob contains nan"

                # Get correctness and instance_id for per-prompt analysis
                correctness = data.get('correctness', None) if 'correctness' in data.keys() else None
                instance_id = data['instance_id']
                
                pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_by_prompt = core_algos.compute_policy_loss(
                    old_log_prob=old_log_prob,
                    log_prob=log_prob,
                    advantages=advantages,
                    eos_mask=response_mask,
                    cliprange=clip_ratio,
                    low_clip_ratio=low_clip_ratio,
                    high_clip_ratio=high_clip_ratio,
                    correctness=correctness,
                    instance_id=instance_id,
                    total_token_count=total_token_count
                )
                # compute entropy loss from entropy
                entropy_loss = verl_F.masked_mean(entropy, response_mask)
                assert not torch.isnan(entropy_loss).any(), "entropy_loss contains nan"

                # compute policy loss
                # policy_loss = pg_loss - entropy_loss * entropy_coeff

                if isinstance(entropy_coeff, torch.Tensor):
                    weighted_entropy_loss = verl_F.masked_mean(entropy * entropy_coeff.to(entropy.device)[:, None], response_mask)
                else:
                    weighted_entropy_loss = verl_F.masked_mean(entropy * entropy_coeff, response_mask)

                # entropy_loss = verl_F.masked_mean(entropy, response_mask)

                # # print(f"HEREHERE entropy_coeff: {entropy_coeff}" )
                policy_loss = pg_loss - weighted_entropy_loss

                if self.config.use_kl_loss:
                    ref_log_prob = data['ref_log_prob']
                    # compute kl loss
                    kld = core_algos.kl_penalty(logprob=log_prob,
                                                ref_logprob=ref_log_prob,
                                                kl_penalty=self.config.kl_loss_type)
                    kl_loss = masked_mean(kld, response_mask)

                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    metrics['actor/kl_loss'] = kl_loss.detach().item()
                    metrics['actor/kl_coef'] = self.config.kl_loss_coef

                if self.config.use_dynamic_bsz:
                    # relative to the dynamic bsz
                    loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                else:
                    loss = policy_loss / self.gradient_accumulation
                loss.backward()

                data = {
                    'actor/entropy_loss': entropy_loss.detach().item(),
                    'actor/pg_loss': pg_loss.detach().item(),
                    'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                    'actor/ppo_kl': ppo_kl.detach().item(),
                    'actor/loss': loss.detach().item(),
                    'actor/token_logprob': log_prob[response_mask==1].flatten().detach().cpu().numpy(),
                }
                
                # Add per-prompt clipping fractions if available
                if pg_clipfrac_by_prompt is not None:
                    # Calculate average clipping fractions across prompts
                    correct_clipfracs = [v['correct_clipfrac'] for v in pg_clipfrac_by_prompt.values() if v['correct_count'] > 0]
                    incorrect_clipfracs = [v['incorrect_clipfrac'] for v in pg_clipfrac_by_prompt.values() if v['incorrect_count'] > 0]
                    
                    if correct_clipfracs:
                        data['actor/pg_clipfrac_correct_avg'] = np.mean(correct_clipfracs)
                        data['actor/pg_clipfrac_correct_std'] = np.std(correct_clipfracs)
                    
                    if incorrect_clipfracs:
                        data['actor/pg_clipfrac_incorrect_avg'] = np.mean(incorrect_clipfracs)
                        data['actor/pg_clipfrac_incorrect_std'] = np.std(incorrect_clipfracs)
                    
                    # Store detailed per-prompt data for analysis
                    data['actor/pg_clipfrac_by_prompt'] = pg_clipfrac_by_prompt
                append_to_dict(metrics, data)

            grad_norm = self._optimizer_step()
            data = {'actor/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
