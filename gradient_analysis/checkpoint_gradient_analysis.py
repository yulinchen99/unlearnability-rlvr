#!/usr/bin/env python3
"""
GRPO Gradient Analysis for Intermediate Checkpoint Rollouts

This script analyzes GRPO (Group Relative Policy Optimization) gradients from rollouts 
generated at an intermediate checkpoint. It calculates:
1. Average GRPO gradient across all rollouts
2. GRPO gradient for each individual prompt (group-based ranking)
3. Separate GRPO gradients from positive and negative rollouts

GRPO Loss: -log(softmax(rewards)) * logprobs
This implements group-based relative ranking where responses are ranked within each prompt group.

PERFORMANCE OPTIMIZATIONS:
- Batched gradient computation: Processes multiple rollouts in a single forward pass
- Optimized batching: Groups similar-length sequences to minimize padding
- Memory management: Automatic GPU cache clearing and fallback to individual computation
- Configurable batch size: Adjust based on available GPU memory

Usage:
    python checkpoint_gradient_analysis.py --rollouts_file <path_to_rollouts.json> --model_path <path_to_checkpoint> [--batch_size 8]
"""

import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from verl.utils.torch_functional import logprobs_from_logits, entropy_from_logits

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def gather_from_labels(data, label):
    """Gather the label from data. The value in label should be [0, vocab_size)

    Args:
        data: (..., vocab_size)
        label (torch.IntTensor) : (...,)

    Returns:

    """

    output = torch.gather(data, -1, label.unsqueeze(-1)).squeeze(-1)
    return output

QWEN_PROMPT = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
LLAMA_PROMPT = """Question:\n{question}\nAnswer:\nLet's think step by step."""

def get_prompt(question: str, prompt_type: str = "qwen"):
    if prompt_type == "qwen":
        return QWEN_PROMPT.format(question=question)
    elif prompt_type == "llama":
        return LLAMA_PROMPT.format(question=question)
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")


class CheckpointGradientAnalyzer:
    def __init__(self, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 lora: bool = False, lora_r: int = 16, lora_alpha: int = 32,
                 prompt_type: Optional[str] = None):
        """
        Initialize the GRPO gradient analyzer with a model checkpoint.

        Args:
            model_path: Path to the model checkpoint
            device: Device to run computations on
            lora: If True, wrap the base model with a LoRA adapter
            lora_r / lora_alpha: LoRA rank / alpha (only when lora=True)
            prompt_type: "qwen" or "llama". If None, infer from model_path.
        """
        self.device = device
        self.model_path = model_path
        self.print_example = 0

        logger.info(f"Loading model from {model_path}")
        if lora:
            # Fix LoRA init seed so the adapter weights are reproducible across runs.
            SEED = 42
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED)
            os.environ["PYTHONHASHSEED"] = str(SEED)
            from peft import LoraConfig, TaskType, get_peft_model
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(base_model, peft_config)
            self.model.print_trainable_parameters()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if prompt_type is not None:
            self.prompt_type = prompt_type
        elif "qwen" in model_path.lower() or "instruct" in model_path.lower():
            self.prompt_type = "qwen"
        elif "llama" in model_path.lower() or "octothinker" in model_path.lower():
            self.prompt_type = "llama"
        else:
            raise ValueError(
                f"Could not infer prompt_type from model path {model_path!r}; "
                "pass --prompt-type qwen or --prompt-type llama explicitly."
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def load_rollouts(self, rollouts_files: List[str]) -> List[Dict]:
        """Load rollouts from one or more JSON / JSONL files."""
        logger.info(f"Loading rollouts from {rollouts_files}")
        all_rollouts = []
        for file in rollouts_files:
            if file.endswith(".jsonl"):
                with open(file, 'r') as f:
                    rollouts = [json.loads(line) for line in f]
            else:
                with open(file, 'r') as f:
                    rollouts = json.load(f)
            all_rollouts.extend(rollouts)
        return all_rollouts
    
    def group_rollouts_by_prompt(self, rollouts: List[Dict]) -> Dict[int, List[Dict]]:
        """
        Group rollouts by question_id (prompt).
        
        Args:
            rollouts: List of rollout dictionaries
            
        Returns:
            Dictionary mapping question_id to list of rollouts for that prompt
        """
        grouped = defaultdict(list)
        for rollout in rollouts:
            question_id = rollout['question_id']
            grouped[question_id].append(rollout)
        return dict(grouped)
    
    def get_group_rewards(self, grouped_rollouts: Dict[int, List[Dict]]):
        new_grouped_rollouts = defaultdict(list)
        for question_id, rollouts in grouped_rollouts.items():
            rewards = [rollout['verification']['score'] for rollout in rollouts]
            avg_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            if std_reward == 0:
                continue
            new_grouped_rollouts[question_id] = rollouts
            for rollout in rollouts:
                rollout['verification']['score'] = (rollout['verification']['score'] - avg_reward) / (std_reward + 1e-6)
        return new_grouped_rollouts
    
    def sample_rollouts(self, grouped_rollouts: Dict[int, List[Dict]], num_rollouts: int):
        # sample num_rollouts rollouts for each question_id for correct and incorrect rollouts
        # first group by rollout["verification"]["is_correct"]
        sampled_rollouts = defaultdict(list)
        for question_id, rollouts in grouped_rollouts.items():
            correct_rollouts = [rollout for rollout in rollouts if rollout['verification']['is_correct']]
            incorrect_rollouts = [rollout for rollout in rollouts if not rollout['verification']['is_correct']]
            # do random sampling
            correct_rollouts = random.sample(correct_rollouts, min(num_rollouts, len(correct_rollouts)))
            incorrect_rollouts = random.sample(incorrect_rollouts, min(num_rollouts, len(incorrect_rollouts)))
            if not correct_rollouts or not incorrect_rollouts:
                continue
            sampled_rollouts[question_id] = correct_rollouts + incorrect_rollouts
        return sampled_rollouts
    
    def tokenize(self, grouped_rollouts: Dict[int, List[Dict]]):
        grouped_tokenized_rollouts = defaultdict(list)
        print_ = False
        for question_id, rollouts in grouped_rollouts.items():
            for rollout in rollouts:
                prompt = get_prompt(rollout['question'], self.prompt_type)
                # add chat template if tokenizer has chat template
                if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
                    prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_special_tokens=False, tokenize=False)
                if not print_:
                    print("========== Example Prompt ==========")
                    print(prompt)
                    print("=====================================")
                    print_ = True

                response = rollout['response'] + self.tokenizer.eos_token
                # get input_ids (prompt + response) and loss_mask (mask the prompt part)    
                full_text = prompt + response

                if self.print_example < 3:
                    print("=== Example Input ===")
                    print(full_text)
                    print("===================")
                    self.print_example += 1

                input_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
                prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
                loss_mask = [1] * len(input_ids)
                loss_mask = torch.LongTensor(loss_mask)
                loss_mask[:len(prompt_ids)] = 0
                input_ids = torch.LongTensor(input_ids)
                attention_mask = torch.LongTensor([1] * len(input_ids))


                grouped_tokenized_rollouts[question_id].append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'loss_mask': loss_mask,
                    "reward": rollout['verification']['score'],
                    "correct": rollout['verification']['is_correct'],
                    # 'prompt_length': len(prompt),
                    # 'length': len(input_ids)
                })
        return grouped_tokenized_rollouts

    
    
    
    def pad_sequences(self, sequences: List[torch.Tensor], pad_value: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad sequences to the same length.
        
        Args:
            sequences: List of 1D tensors to pad
            pad_value: Value to use for padding
            
        Returns:
            Tuple of (padded_sequences, attention_mask)
        """
        max_len = max(seq.size(0) for seq in sequences)
        batch_size = len(sequences)
        
        padded = torch.full((batch_size, max_len), pad_value, dtype=sequences[0].dtype)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        
        for i, seq in enumerate(sequences):
            seq_len = seq.size(0)
            padded[i, :seq_len] = seq
            attention_mask[i, :seq_len] = 1
            
        return padded, attention_mask
    
    def create_optimized_batches(self, rollouts: List[Dict], batch_size: int) -> List[List[Dict]]:
        """
        Create batches optimized for similar sequence lengths to minimize padding.
        
        Args:
            rollouts: List of rollout dictionaries
            batch_size: Maximum batch size
            
        Returns:
            List of batches, each containing rollout dictionaries
        """
        # Sort rollouts by sequence length to group similar lengths together
        sorted_rollouts = sorted(rollouts, key=lambda x: len(x['input_ids']))
        
        batches = []
        current_batch = []
        
        for rollout in sorted_rollouts:
            current_batch.append(rollout)
            
            # If batch is full or adding another would create too much padding, finalize it
            if len(current_batch) >= batch_size:
                batches.append(current_batch)
                current_batch = []
        
        # Add remaining rollouts as final batch
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def estimate_memory_usage(self, batch_size: int, max_seq_len: int) -> float:
        """
        Estimate GPU memory usage for a given batch size and sequence length.
        
        Args:
            batch_size: Number of sequences in batch
            max_seq_len: Maximum sequence length in batch
            
        Returns:
            Estimated memory usage in GB
        """
        # Rough estimation based on model parameters and batch size
        # This is a simplified calculation - actual usage may vary
        vocab_size = self.tokenizer.vocab_size
        hidden_size = getattr(self.model.config, 'hidden_size', 768)
        num_layers = getattr(self.model.config, 'num_hidden_layers', 12)
        
        # Memory for embeddings: batch_size * seq_len * hidden_size * 4 bytes (float32)
        embedding_memory = batch_size * max_seq_len * hidden_size * 4 / (1024**3)
        
        # Memory for attention: batch_size * num_layers * seq_len^2 * 4 bytes
        attention_memory = batch_size * num_layers * (max_seq_len ** 2) * 4 / (1024**3)
        
        # Memory for gradients: roughly 2x forward pass memory
        total_memory = (embedding_memory + attention_memory) * 3  # 3x for gradients
        
        return total_memory
    
    def suggest_batch_size(self, rollouts: List[Dict], max_memory_gb: float = 8.0) -> int:
        """
        Suggest an optimal batch size based on available memory and rollout lengths.
        
        Args:
            rollouts: List of rollout dictionaries
            max_memory_gb: Maximum available GPU memory in GB
            
        Returns:
            Suggested batch size
        """
        if not rollouts:
            return 1
        
        # Get sequence length statistics
        lengths = [len(rollout['input_ids']) for rollout in rollouts]
        avg_length = np.mean(lengths)
        max_length = max(lengths)
        
        # Start with a reasonable batch size and adjust
        suggested_batch_size = 2
        
        # Check if this batch size fits in memory
        estimated_memory = self.estimate_memory_usage(suggested_batch_size, max_length)
        
        if estimated_memory > max_memory_gb:
            # Reduce batch size
            suggested_batch_size = max(1, int(suggested_batch_size * max_memory_gb / estimated_memory))
        
        return suggested_batch_size
    

    def compute_gradients_batch_simplified(self, batch_items: List[Dict], top_entropy_token: bool = False) -> List[Dict[str, torch.Tensor]]:
        """
        Compute gradients for the LM_HEAD ONLY for a batch of rollouts.
        This method uses direct calculation instead of torch.backward().

        NOT WORKING, DO NOT USE
        """
        batch_size = len(batch_items)
        if batch_size == 0:
            return []
        
        # Extract data from batch
        rewards = torch.tensor([item['reward'] for item in batch_items], dtype=torch.float32, device=self.device)
        input_ids_list = [item['input_ids'] for item in batch_items]
        loss_masks_list = [item['loss_mask'] for item in batch_items]
        
        # Pad sequences
        input_ids_padded, attention_mask_padded = self.pad_sequences(input_ids_list, pad_value=self.tokenizer.pad_token_id)
        loss_mask_padded, _ = self.pad_sequences(loss_masks_list, pad_value=0)
        
        # Move to device
        input_ids_padded = input_ids_padded.to(self.device)
        attention_mask_padded = attention_mask_padded.to(self.device)
        loss_mask_padded = loss_mask_padded.bool().to(self.device)
        
        # Create labels
        labels = input_ids_padded.clone()
        labels[~loss_mask_padded] = -100
        
        # Forward pass - SHARED COMPUTATION
        # We MUST get hidden_states to compute weight gradients
        with torch.no_grad(): # No grad needed for forward pass if we do manual backward
            outputs = self.model(
                input_ids=input_ids_padded,
                attention_mask=attention_mask_padded,
                output_hidden_states=True  # <-- CRITICAL CHANGE
            )
            
        logits = outputs.logits  # [B, S, V]
        # Get the input to the LM head (last hidden state)
        last_hidden_states = outputs.hidden_states[-1] # [B, S, H]
        
        batch_gradients = []
        
        for i in range(batch_size):
            rollout_reward = rewards[i]
            
            # --- 1. Get all potential response tokens ---
            # We align logits (output), hidden_states (input), and labels (target)
            # Logits/Hidden predict next token, so we shift labels by 1
            response_logits_all = logits[i, :-1]             # [S-1, V]
            response_hidden_all = last_hidden_states[i, :-1] # [S-1, H]
            response_labels_all = labels[i, 1:]            # [S-1]
            
            # Get the loss mask for these tokens
            current_mask = loss_mask_padded[i, 1:]         # [S-1]
            
            # --- 2. Apply initial mask ---
            response_logits = response_logits_all[current_mask]
            response_hidden = response_hidden_all[current_mask]
            response_labels = response_labels_all[current_mask]
            
            # --- 3. Apply top_entropy_token filtering (if enabled) ---
            if top_entropy_token:
                seq_len = response_logits.shape[0]
                if seq_len > 0:
                    response_entropy = entropy_from_logits(response_logits) # [N]
                    
                    # Keep top 20%
                    k_to_keep = max(1, int(seq_len * 0.2)) # Keep at least 1
                    
                    if seq_len > k_to_keep:
                        top_k_indices = torch.topk(response_entropy, k=k_to_keep).indices
                        
                        # Filter all three tensors again
                        response_logits = response_logits[top_k_indices]
                        response_hidden = response_hidden[top_k_indices]
                        response_labels = response_labels[top_k_indices]

            # --- 4. Check if any tokens are left ---
            N_final = response_labels.shape[0]
            if N_final == 0:
                print(f"No response tokens, skip this rollout")
                continue

            # --- 5. Manual Gradient Calculation ---
            
            # We need to re-attach logits to the graph for this item
            #
            # We can't do this easily without re-running the forward pass for this item.
            # A-HA! The ratio *is* the part that needs grads.
            # `ratio = torch.exp(response_logprobs - response_logprobs.detach())`
            # `response_logprobs` is what needs to be "live"
            
            # Let's get the "live" logits and logprobs for this item's tokens
            # live_response_logits = self.model.lm_head(response_hidden) # [N_final, V]
            live_response_logits = response_logits
            live_response_logprobs = logprobs_from_logits(
                live_response_logits.unsqueeze(0), 
                response_labels.unsqueeze(0)
            ) # [1, N_final]

            # Get detached logprobs (from the no_grad() forward pass)
            with torch.no_grad():
                detached_logprobs = logprobs_from_logits(
                    response_logits.unsqueeze(0), 
                    response_labels.unsqueeze(0)
                ) # [1, N_final]

            ratio = torch.exp(live_response_logprobs - detached_logprobs) # [1, N_final]
            
            # Compute loss
            # rollout_loss = (-rollout_reward * ratio).mean()
            
            # Manually compute the upstream gradient (dLoss / dLogits)
            
            # dLoss / d(logp) = (-reward / N) * ratio
            grad_loss_wrt_logprob = (-rollout_reward / N_final) * ratio.squeeze(0) # [N_final]
            
            # d(logp) / d(logits) = (one_hot_labels - probs)
            probs = F.softmax(live_response_logits, dim=-1) # [N_final, V]
            vocab_size = probs.shape[-1]
            
            targets_one_hot = F.one_hot(response_labels, num_classes=vocab_size).to(probs.dtype) # [N_final, V]
            
            grad_logprob_wrt_logits = targets_one_hot - probs # [N_final, V]
            
            # Upstream gradient: dLoss / dLogits
            # (N_final,) -> (N_final, 1) * (N_final, V) -> (N_final, V)
            grad_L = grad_loss_wrt_logprob.unsqueeze(-1) * grad_logprob_wrt_logits
            
            # Now, get gradients for the LM head
            # Assumes lm_head is a standard nn.Linear(hidden_size, vocab_size)
            # Its weight shape is [vocab_size, hidden_size]
            
            # dLoss / dWeight = (dLoss / dLogits).T @ Hidden_Input
            # (V, N_final) @ (N_final, H) -> (V, H)
            # convert both to float16
            grad_W_lm = grad_L.t() @ response_hidden.to(grad_L.dtype)
            
            # dLoss / dBias = sum(dLoss / dLogits)
            # (N_final, V) -> sum(dim=0) -> (V,)
            # grad_b_lm = grad_L.sum(dim=0)
            
            # Store gradients
            rollout_gradients = {
                # Use the correct names for your model parameters
                'lm_head.weight': grad_W_lm.cpu().detach(),
                # 'lm_head.bias': grad_b_lm.cpu().detach()
            }
            batch_gradients.append(rollout_gradients)

        return batch_gradients
    
    def compute_mle_gradients_batch(self, batch_items: List[Dict]) -> List[Dict[str, torch.Tensor]]:
        """
        Compute gradients for a batch of rollouts using MLE loss. (regardless of whether the rollout is correct or incorrect)
        This method processes multiple rollouts in a single forward pass for efficiency.
        Args:
            batch_items: List of rollout dictionaries
        Returns:
            List of gradient dictionaries, one per rollout
        """
        batch_size = len(batch_items)
        if batch_size == 0:
            return []
        
        # Extract data from batch
        input_ids_list = [item['input_ids'] for item in batch_items]
        loss_masks_list = [item['loss_mask'] for item in batch_items]
        attention_masks_list = [item['attention_mask'] for item in batch_items]
        
        # Pad sequences
        input_ids_padded, attention_mask_padded = self.pad_sequences(input_ids_list, pad_value=self.tokenizer.pad_token_id)
        loss_mask_padded, _ = self.pad_sequences(loss_masks_list, pad_value=0)
        
        # Move to device
        input_ids_padded = input_ids_padded.to(self.device)
        attention_mask_padded = attention_mask_padded.to(self.device)
        loss_mask_padded = loss_mask_padded.bool().to(self.device)
        
        # Create labels (only compute loss on response part)
        labels = input_ids_padded.clone()
        labels[~loss_mask_padded] = -100  # Ignore question part in loss
        
        # Forward pass - shared computation
        self.model.zero_grad()
        outputs = self.model(
            input_ids=input_ids_padded,
            attention_mask=attention_mask_padded,
            labels=labels
        )
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]       

        
        
        # Compute individual gradients using a more efficient approach
        batch_gradients = []
        
        for i in range(batch_size):
            # Get the loss mask for this specific rollout
            rollout_loss_mask = loss_mask_padded[i]  # [seq_len]
            rollout_labels = labels[i]  # [seq_len]
            
            # Get logits for response tokens only (exclude last token for logits)
            response_logits = logits[i, :-1]  # [seq_len-1, vocab_size]
            response_labels = rollout_labels[1:]  # [seq_len-1]
            response_loss_mask = rollout_loss_mask[1:]  # [seq_len-1]
            # print(f"response logits shape: {response_logits.size()}")
            response_logits = response_logits[response_loss_mask]
            response_labels = response_labels[response_loss_mask]
            response_loss_mask = response_loss_mask[response_loss_mask]
            # print(f"response logits shape after loss mask: {response_logits.size()}")
           
            
            if response_loss_mask.sum() == 0:
                # No response tokens, create zero gradients
                # gradients = {}
                # for name, param in self.model.named_parameters():
                #     gradients[name] = torch.zeros_like(param).cpu()
                # batch_gradients.append(gradients)
                print(f"No response tokens, skip this rollout")
                continue


            response_logprobs = logprobs_from_logits(response_logits[response_loss_mask].unsqueeze(0), response_labels[response_loss_mask].unsqueeze(0))
            
            # Compute loss for this rollout
            rollout_loss = -response_logprobs.mean()
            
            # Compute gradients for this specific rollout
            # We need to do individual backward passes for each rollout
            # This is still more efficient than separate forward passes
            rollout_gradients = {}
            
            # Clear gradients first
            self.model.zero_grad()
            
            # Backward pass for this specific rollout
            rollout_loss.backward(retain_graph=True)
            
            # Extract gradients
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    rollout_gradients[name] = param.grad.clone().cpu()
                else:
                    rollout_gradients[name] = torch.zeros_like(param).cpu()
            
            batch_gradients.append(rollout_gradients)
        
        return batch_gradients
    
    def compute_gradients_batch(self, batch_items: List[Dict], top_entropy_token: bool = False) -> List[Dict[str, torch.Tensor]]:
        """
        Compute gradients for a batch of rollouts using GRPO-style loss.
        This method processes multiple rollouts in a single forward pass for efficiency.
        
        Args:
            batch_items: List of rollout dictionaries
            
        Returns:
            List of gradient dictionaries, one per rollout
        """
        batch_size = len(batch_items)
        if batch_size == 0:
            return []
        
        # Extract data from batch
        rewards = torch.tensor([item['reward'] for item in batch_items], dtype=torch.float32, device=self.device)
        input_ids_list = [item['input_ids'] for item in batch_items]
        loss_masks_list = [item['loss_mask'] for item in batch_items]
        attention_masks_list = [item['attention_mask'] for item in batch_items]
        
        # Pad sequences
        input_ids_padded, attention_mask_padded = self.pad_sequences(input_ids_list, pad_value=self.tokenizer.pad_token_id)
        loss_mask_padded, _ = self.pad_sequences(loss_masks_list, pad_value=0)
        
        # Move to device
        input_ids_padded = input_ids_padded.to(self.device)
        attention_mask_padded = attention_mask_padded.to(self.device)
        loss_mask_padded = loss_mask_padded.bool().to(self.device)
        
        # Create labels (only compute loss on response part)
        labels = input_ids_padded.clone()
        labels[~loss_mask_padded] = -100  # Ignore question part in loss
        
        # Forward pass - shared computation
        self.model.zero_grad()
        outputs = self.model(
            input_ids=input_ids_padded,
            attention_mask=attention_mask_padded,
            labels=labels
        )
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        
        
        # Compute individual gradients using a more efficient approach
        batch_gradients = []
        
        for i in range(batch_size):
            # Get the loss mask for this specific rollout
            rollout_loss_mask = loss_mask_padded[i]  # [seq_len]
            rollout_labels = labels[i]  # [seq_len]
            rollout_reward = rewards[i]  # scalar
            
            # Get logits for response tokens only (exclude last token for logits)
            response_logits = logits[i, :-1]  # [seq_len-1, vocab_size]
            response_labels = rollout_labels[1:]  # [seq_len-1]
            response_loss_mask = rollout_loss_mask[1:]  # [seq_len-1]
            # print(f"response logits shape: {response_logits.size()}")
            response_logits = response_logits[response_loss_mask]
            response_labels = response_labels[response_loss_mask]
            response_loss_mask = response_loss_mask[response_loss_mask]
            # print(f"response logits shape after loss mask: {response_logits.size()}")


            if top_entropy_token:
                response_entropy = entropy_from_logits(response_logits)  # [seq_len]
                # update loss mask to only include top entropy tokens for each sequence
                # top 20% entropy tokens
                seq_len = response_loss_mask.sum().item()
                tomask_entropy_tokens = torch.argsort(response_entropy)[:int(seq_len * 0.8)]
                response_loss_mask[tomask_entropy_tokens] = False
            
            if response_loss_mask.sum() == 0:
                # No response tokens, create zero gradients
                # gradients = {}
                # for name, param in self.model.named_parameters():
                #     gradients[name] = torch.zeros_like(param).cpu()
                # batch_gradients.append(gradients)
                print(f"No response tokens, skip this rollout")
                continue
            # print(f"Valid #Response tokens after loss mask: {response_loss_mask.sum()}", response_loss_mask[:10])

            # Gather logits for the actual response tokens
            # response_logits_gathered = gather_from_labels(
            #     response_logits[response_loss_mask].unsqueeze(0), 
            #     response_labels[response_loss_mask].unsqueeze(0)
            # )  # [1, num_response_tokens]

            response_logprobs = logprobs_from_logits(response_logits[response_loss_mask].unsqueeze(0), response_labels[response_loss_mask].unsqueeze(0))
            ratio = torch.exp(response_logprobs - response_logprobs.detach())
            
            # Compute loss for this rollout
            rollout_loss = (-rollout_reward * ratio).mean()
            
            # Compute gradients for this specific rollout
            # We need to do individual backward passes for each rollout
            # This is still more efficient than separate forward passes
            rollout_gradients = {}
            
            # Clear gradients first
            self.model.zero_grad()
            
            # Backward pass for this specific rollout
            rollout_loss.backward(retain_graph=True)
            
            # Extract gradients
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    rollout_gradients[name] = param.grad.clone().cpu()
            
            batch_gradients.append(rollout_gradients)
        
        return batch_gradients
    
    def save_gradients(self, gradients: Dict[str, torch.Tensor], output_path: str):
        """Save gradients to file."""
        # Convert to CPU and numpy for saving
        # convert to float16
        gradients_cpu = {name: grad.cpu().to(torch.float16).numpy() for name, grad in gradients.items()}

        # calculate norm
        # flatten the gradient into a 1D array  
        # flat_grad = np.concatenate([grad.flatten() for grad in gradients_cpu.values()])
        # norm = np.linalg.norm(flat_grad)
        
        with open(output_path, 'wb') as f:
            np.savez_compressed(f, **gradients_cpu)
        
        logger.info(f"Saved gradients to {output_path}")
        return None
        # return norm
    
    def analyze_gradient_magnitudes(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Analyze gradient magnitudes by layer."""
        magnitudes = {}
        for name, grad in gradients.items():
            magnitudes[name] = grad.norm().item()
        return magnitudes
    
    def setup_before_analysis(self):
        self.prompt_avg_gradients = {}
        self.prompt_correct_avg_gradients = {}
        self.prompt_incorrect_avg_gradients = {}

        # normalized
        self.prompt_avg_gradients_normalized = {}
        self.prompt_correct_avg_gradients_normalized = {}
        self.prompt_incorrect_avg_gradients_normalized = {}


        self.prompt_avg_gradients_squared = {}
        self.prompt_correct_avg_gradients_squared = {}
        self.prompt_incorrect_avg_gradients_squared = {}

        # normalized squared
        self.prompt_avg_gradients_normalized_squared = {}
        self.prompt_correct_avg_gradients_normalized_squared = {}
        self.prompt_incorrect_avg_gradients_normalized_squared = {}

        self.prompt_correct_total = defaultdict(int)
        self.prompt_incorrect_total = defaultdict(int)
        self.prompt_total = defaultdict(int)
        self.all_avg_gradients = None
        self.all_avg_gradients_squared = 0.0
        self.all_total = 0
    
    def get_squared(self, gradients: Dict[str, torch.Tensor]):
        total = 0.0
        for name, grad in gradients.items():
            total += grad.square().sum().item()
        return total
    
    def get_normalized(self, gradients: Dict[str, torch.Tensor], norm: float=None):
        # return normalized gradient
        # get norm first
        if norm is None:
            norm = torch.norm(torch.stack([grad.flatten() for grad in gradients.values()])).item()
        return {name: grad / norm for name, grad in gradients.items()}


    
    def update_avg_gradients_squared(self, squared_value: float):
        self.all_avg_gradients_squared += squared_value
    
    def update_prompt_avg_gradients_squared(self, question_id: int, squared_value: float):
        if question_id not in self.prompt_avg_gradients_squared:
            self.prompt_avg_gradients_squared[question_id] = squared_value
        else:
            self.prompt_avg_gradients_squared[question_id] += squared_value
    
    def update_prompt_incorrect_avg_gradients_squared(self, question_id: int, squared_value: float):
        if question_id not in self.prompt_incorrect_avg_gradients_squared:
            self.prompt_incorrect_avg_gradients_squared[question_id] = squared_value
        else:
            self.prompt_incorrect_avg_gradients_squared[question_id] += squared_value
    
    def update_prompt_correct_avg_gradients_squared(self, question_id: int, squared_value: float):
        if question_id not in self.prompt_correct_avg_gradients_squared:
            self.prompt_correct_avg_gradients_squared[question_id] = squared_value
        else:
            self.prompt_correct_avg_gradients_squared[question_id] += squared_value
    
    # normalized
    def update_prompt_avg_gradients_normalized(self, question_id: int, normalized_gradients: Dict[str, torch.Tensor]):
        if question_id not in self.prompt_avg_gradients_normalized:
            self.prompt_avg_gradients_normalized[question_id] = normalized_gradients
        else:
            for param_name in self.prompt_avg_gradients_normalized[question_id].keys():
                self.prompt_avg_gradients_normalized[question_id][param_name] += normalized_gradients[param_name]
    
    def update_prompt_correct_avg_gradients_normalized(self, question_id: int, normalized_gradients: Dict[str, torch.Tensor]):
        if question_id not in self.prompt_correct_avg_gradients_normalized:
            self.prompt_correct_avg_gradients_normalized[question_id] = normalized_gradients
        else:
            for param_name in self.prompt_correct_avg_gradients_normalized[question_id].keys():
                self.prompt_correct_avg_gradients_normalized[question_id][param_name] += normalized_gradients[param_name]
    
    def update_prompt_incorrect_avg_gradients_normalized(self, question_id: int, normalized_gradients: Dict[str, torch.Tensor]):
        if question_id not in self.prompt_incorrect_avg_gradients_normalized:
            self.prompt_incorrect_avg_gradients_normalized[question_id] = normalized_gradients
        else:
            for param_name in self.prompt_incorrect_avg_gradients_normalized[question_id].keys():
                self.prompt_incorrect_avg_gradients_normalized[question_id][param_name] += normalized_gradients[param_name]
    
    # normalized squared
    def update_prompt_avg_gradients_normalized_squared(self, question_id: int, normalized_square_value: Dict[str, torch.Tensor]):
        if question_id not in self.prompt_avg_gradients_normalized_squared:
            self.prompt_avg_gradients_normalized_squared[question_id] = normalized_square_value
        else:
            self.prompt_avg_gradients_normalized_squared[question_id] += normalized_square_value
    
    def update_prompt_correct_avg_gradients_normalized_squared(self, question_id: int, normalized_square_value: Dict[str, torch.Tensor]):
        if question_id not in self.prompt_correct_avg_gradients_normalized_squared:
            self.prompt_correct_avg_gradients_normalized_squared[question_id] = normalized_square_value
        else:
            self.prompt_correct_avg_gradients_normalized_squared[question_id] += normalized_square_value
    
    def update_prompt_incorrect_avg_gradients_normalized_squared(self, question_id: int, normalized_square_value: Dict[str, torch.Tensor]):
        if question_id not in self.prompt_incorrect_avg_gradients_normalized_squared:
            self.prompt_incorrect_avg_gradients_normalized_squared[question_id] = normalized_square_value
        else:
            self.prompt_incorrect_avg_gradients_normalized_squared[question_id] += normalized_square_value


    #######################
    def update_avg_gradients(self, gradients: Dict[str, torch.Tensor]):
        if self.all_avg_gradients is None:
            self.all_avg_gradients = gradients
        else:
            for param_name in self.all_avg_gradients.keys():
                self.all_avg_gradients[param_name] += gradients[param_name]
        # self.all_total += 1
    
    def update_prompt_avg_gradients(self, question_id: int, gradients: Dict[str, torch.Tensor]):
        if question_id not in self.prompt_avg_gradients:
            self.prompt_avg_gradients[question_id] = gradients
        else:
            for param_name in self.prompt_avg_gradients[question_id].keys():
                self.prompt_avg_gradients[question_id][param_name] += gradients[param_name]
        # self.prompt_total[question_id] += 1
    
    def update_prompt_incorrect_avg_gradients(self, question_id: int, gradients: Dict[str, torch.Tensor]):
        if question_id not in self.prompt_incorrect_avg_gradients:
            self.prompt_incorrect_avg_gradients[question_id] = gradients
        else:
            for param_name in self.prompt_incorrect_avg_gradients[question_id].keys():
                self.prompt_incorrect_avg_gradients[question_id][param_name] += gradients[param_name]
        # self.prompt_incorrect_total[question_id] += 1
    
    def update_prompt_correct_avg_gradients(self, question_id: int, gradients: Dict[str, torch.Tensor]):
        if question_id not in self.prompt_correct_avg_gradients:
            self.prompt_correct_avg_gradients[question_id] = gradients
        else:
            for param_name in self.prompt_correct_avg_gradients[question_id].keys():
                self.prompt_correct_avg_gradients[question_id][param_name] += gradients[param_name]
        # self.prompt_correct_total[question_id] += 1
    
    def avg_gradients(self):
        if self.all_avg_gradients is not None:
            for param_name in self.all_avg_gradients.keys():
                self.all_avg_gradients[param_name] /= self.all_total
        for question_id in self.prompt_avg_gradients.keys():
            for param_name in self.prompt_avg_gradients[question_id].keys():
                self.prompt_avg_gradients[question_id][param_name] /= self.prompt_total[question_id]
        for question_id in self.prompt_correct_avg_gradients.keys():
            for param_name in self.prompt_correct_avg_gradients[question_id].keys():
                self.prompt_correct_avg_gradients[question_id][param_name] /= self.prompt_correct_total[question_id]
        for question_id in self.prompt_incorrect_avg_gradients.keys():
            for param_name in self.prompt_incorrect_avg_gradients[question_id].keys():
                self.prompt_incorrect_avg_gradients[question_id][param_name] /= self.prompt_incorrect_total[question_id]

        for question_id in self.prompt_avg_gradients_normalized.keys():
            for param_name in self.prompt_avg_gradients_normalized[question_id].keys():
                self.prompt_avg_gradients_normalized[question_id][param_name] /= self.prompt_total[question_id]
        for question_id in self.prompt_correct_avg_gradients_normalized.keys():
            for param_name in self.prompt_correct_avg_gradients_normalized[question_id].keys():
                self.prompt_correct_avg_gradients_normalized[question_id][param_name] /= self.prompt_correct_total[question_id]
        for question_id in self.prompt_incorrect_avg_gradients_normalized.keys():
            for param_name in self.prompt_incorrect_avg_gradients_normalized[question_id].keys():
                self.prompt_incorrect_avg_gradients_normalized[question_id][param_name] /= self.prompt_incorrect_total[question_id]
    
    def avg_gradients_squared(self):
        self.all_avg_gradients_squared /= self.all_total
        for question_id in self.prompt_avg_gradients_squared.keys():
            self.prompt_avg_gradients_squared[question_id] /= self.prompt_total[question_id]
        for question_id in self.prompt_correct_avg_gradients_squared.keys():
            self.prompt_correct_avg_gradients_squared[question_id] /= self.prompt_correct_total[question_id]
        for question_id in self.prompt_incorrect_avg_gradients_squared.keys():
            self.prompt_incorrect_avg_gradients_squared[question_id] /= self.prompt_incorrect_total[question_id]


        for question_id in self.prompt_avg_gradients_normalized_squared.keys():
            self.prompt_avg_gradients_normalized_squared[question_id] /= self.prompt_total[question_id]
        for question_id in self.prompt_correct_avg_gradients_normalized_squared.keys():
            self.prompt_correct_avg_gradients_normalized_squared[question_id] /= self.prompt_correct_total[question_id]
        for question_id in self.prompt_incorrect_avg_gradients_normalized_squared.keys():
            self.prompt_incorrect_avg_gradients_normalized_squared[question_id] /= self.prompt_incorrect_total[question_id]
    
    def avg_individual_prompt_gradient(self, qid_list: List[int]):
        for qid in qid_list:
            if qid in self.prompt_avg_gradients:
                for param_name in self.prompt_avg_gradients[qid].keys():
                    self.prompt_avg_gradients[qid][param_name] /= self.prompt_total[qid]
            if qid in self.prompt_correct_avg_gradients:
                for param_name in self.prompt_correct_avg_gradients[qid].keys():
                    self.prompt_correct_avg_gradients[qid][param_name] /= self.prompt_correct_total[qid]
            if qid in self.prompt_incorrect_avg_gradients:
                for param_name in self.prompt_incorrect_avg_gradients[qid].keys():
                    self.prompt_incorrect_avg_gradients[qid][param_name] /= self.prompt_incorrect_total[qid]
            
            if qid in self.prompt_avg_gradients_normalized:
                for param_name in self.prompt_avg_gradients_normalized[qid].keys():
                    self.prompt_avg_gradients_normalized[qid][param_name] /= self.prompt_total[qid]
            if qid in self.prompt_correct_avg_gradients_normalized:
                for param_name in self.prompt_correct_avg_gradients_normalized[qid].keys():
                    self.prompt_correct_avg_gradients_normalized[qid][param_name] /= self.prompt_correct_total[qid]
            if qid in self.prompt_incorrect_avg_gradients_normalized:
                for param_name in self.prompt_incorrect_avg_gradients_normalized[qid].keys():
                    self.prompt_incorrect_avg_gradients_normalized[qid][param_name] /= self.prompt_incorrect_total[qid]
        # # avg squared
        # for question_id in self.prompt_avg_gradients_squared.keys():
        #     self.prompt_avg_gradients_squared[question_id] /= self.prompt_total[question_id]
        # for question_id in self.prompt_correct_avg_gradients_squared.keys():
        #     self.prompt_correct_avg_gradients_squared[question_id] /= self.prompt_correct_total[question_id]
        # for question_id in self.prompt_incorrect_avg_gradients_squared.keys():
        #     self.prompt_incorrect_avg_gradients_squared[question_id] /= self.prompt_incorrect_total[question_id]
    
    def run_full_analysis(self, args, rollouts_files: List[str], output_dir: str = "./gradient_analysis_output", top_entropy_token: bool = False, type="both", start_idx=-1, end_idx=-1):
        """Run complete gradient analysis."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Load rollouts
        rollouts = self.load_rollouts(rollouts_files)
        
        # Group by prompt
        print("Grouping rollouts by prompt")
        grouped_rollouts = self.group_rollouts_by_prompt(rollouts)
        question_ids = list(grouped_rollouts.keys())
        question_ids = list(sorted(question_ids))
        if args.id_file is None:
            # default to math train set 0.5B typed ids
            # sample first 100 ids from each group
            # question_ids = question_ids[:100] + question_ids[200:300] + question_ids[400:500] + question_ids[600:700]
            # print(f"Sampled {len(question_ids)} questions (first 100 from each group)")
            pass
        else:
            question_ids = json.load(open(args.id_file))
        # print(f"Processing {len(question_ids)} questions")
        question_ids = [i for i in question_ids if not os.path.exists(os.path.join(output_dir, f"average_gradients_prompt_correct_{i}.npz"))]
        if start_idx != -1:
            question_ids = question_ids[start_idx:end_idx]
        # if end_idx != -1:
        #     question_ids = question_ids[:end_idx]
        grouped_rollouts = {k: v for k, v in grouped_rollouts.items() if k in question_ids}
        print(f"Processing {len(grouped_rollouts)} prompts")

        print("Getting group rewards")
        grouped_rollouts = self.get_group_rewards(grouped_rollouts)
        if args.pos_only:
            # retain all positive rollout
            print("Removing all negative rollouts")
            new_grouped_rollouts = {}
            for k, v in grouped_rollouts.items():
                new_rollouts = [i for i in v if i["verification"]["is_correct"]]
                if len(new_rollouts) > 20:
                    new_rollouts = list(random.sample(new_rollouts, 20))
                if len(new_rollouts) > 0:
                    new_grouped_rollouts[k] = new_rollouts
            
            grouped_rollouts = new_grouped_rollouts
            print(f"Retained {len(grouped_rollouts)} prompts with positive rollouts")
        else:
            print("Sampling rollouts")
            grouped_rollouts = self.sample_rollouts(grouped_rollouts, 100)
        print("Tokenizing rollouts")
        grouped_rollouts = self.tokenize(grouped_rollouts)

        # Process rollouts in batches for efficiency
        batch_size = getattr(self, 'batch_size', 2)  # Use instance variable or default
        
        # If batch_size is not explicitly set, suggest optimal size
        if not hasattr(self, 'batch_size') or self.batch_size is None:
            all_rollouts = []
            for rollouts_list in grouped_rollouts.values():
                all_rollouts.extend(rollouts_list)
            max_memory = getattr(self, 'max_memory_gb', 8.0)
            suggested_batch_size = self.suggest_batch_size(all_rollouts, max_memory)
            print(f"Suggested batch size: {suggested_batch_size}")
            batch_size = suggested_batch_size
            torch.cuda.empty_cache()
            del all_rollouts
        
        print(f"Using batch size: {batch_size}")
        # cnt = 0
        total_qid = 0
        # qid_list = []
        qid2grad_norm = {}
        for question_id, rollouts in tqdm(grouped_rollouts.items(), desc="Computing gradients"):
            # Create optimized batches for this question's rollouts
            batches = self.create_optimized_batches(rollouts, batch_size)
            
            # Process each batch
            for batch_rollouts in batches:
                
                # try:
                # Compute gradients for the batch
                if args.simplified:
                    batch_gradients = self.compute_gradients_batch_simplified(batch_rollouts, top_entropy_token)
                elif args.mle:
                    batch_gradients = self.compute_mle_gradients_batch(batch_rollouts)
                else:
                    batch_gradients = self.compute_gradients_batch(batch_rollouts, top_entropy_token)
                
                # Update running averages for each rollout in the batch
                for j, rollout in enumerate(batch_rollouts):
                    gradients = batch_gradients[j]
                    if args.lora:
                        gradients = {k: v for k, v in gradients.items() if "base_layer" not in k and "mlp" not in k}

                    squared_value = self.get_squared(gradients)
                    # norm = np.sqrt(squared_value)
                    # normalized_gradients = self.get_normalized(gradients, norm)
                    # normalized_squared_value = self.get_squared(normalized_gradients)
                    if type == "both" or type == "mean":
                        if not args.pos_only:
                            # self.update_avg_gradients(gradients)
                            self.update_prompt_avg_gradients(question_id, gradients)
                            # self.update_prompt_avg_gradients_normalized(question_id, normalized_gradients)
                        if rollout['correct']:
                            self.update_prompt_correct_avg_gradients(question_id, gradients)
                            # self.update_prompt_correct_avg_gradients_normalized(question_id, normalized_gradients)
                        else:
                            self.update_prompt_incorrect_avg_gradients(question_id, gradients)
                            # self.update_prompt_incorrect_avg_gradients_normalized(question_id, normalized_gradients)
                    if type == "both" or type == "squared":
                        if not args.pos_only:
                            # self.update_avg_gradients_squared(squared_value)
                            self.update_prompt_avg_gradients_squared(question_id, squared_value)
                            # self.update_prompt_avg_gradients_normalized_squared(question_id, normalized_squared_value)
                        if rollout['correct']:
                            self.update_prompt_correct_avg_gradients_squared(question_id, squared_value)
                            # self.update_prompt_correct_avg_gradients_normalized_squared(question_id, normalized_squared_value)
                        else:
                            self.update_prompt_incorrect_avg_gradients_squared(question_id, squared_value)
                            # self.update_prompt_incorrect_avg_gradients_normalized_squared(question_id, normalized_squared_value)
                    
                    self.all_total += 1
                    self.prompt_total[question_id] += 1
                    if rollout['correct']:
                        self.prompt_correct_total[question_id] += 1
                    else:
                        self.prompt_incorrect_total[question_id] += 1
                
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            if rollouts:
                total_qid += 1
                

        print("Averaging gradients")
        if type == "both" or type == "mean":
            print("Averaging gradients")
            self.avg_gradients()

        if type == "both" or type == "squared":
            print("Averaging squared gradients")
            self.avg_gradients_squared()

        if type == "both" or type == "mean":
            print("Saving gradients")
            # self.save_gradients(self.all_avg_gradients, os.path.join(output_dir, "average_gradients_all.npz"))
            
            from concurrent.futures import ThreadPoolExecutor
            
            def save_gradient_file(args):
                gradients, filepath = args
                self.save_gradients(gradients, filepath)
            
            save_tasks = []
            
            # Collect all save tasks
            for question_id, gradients in self.prompt_correct_avg_gradients.items():
                save_tasks.append((gradients, os.path.join(output_dir, f"average_gradients_prompt_correct_{question_id}.npz")))

            # for question_id, gradients in self.prompt_correct_avg_gradients_normalized.items():
            #     save_tasks.append((gradients, os.path.join(output_dir, f"average_gradients_prompt_correct_normalized_{question_id}.npz")))

            if not args.pos_only:
                for question_id, gradients in self.prompt_avg_gradients.items():
                    save_tasks.append((gradients, os.path.join(output_dir, f"average_gradients_prompt_{question_id}.npz")))
                
                for question_id, gradients in self.prompt_incorrect_avg_gradients.items():
                    save_tasks.append((gradients, os.path.join(output_dir, f"average_gradients_prompt_incorrect_{question_id}.npz")))
                    
                # for question_id, gradients in self.prompt_avg_gradients_normalized.items():
                #     save_tasks.append((gradients, os.path.join(output_dir, f"average_gradients_prompt_normalized_{question_id}.npz")))
                
                
                # for question_id, gradients in self.prompt_incorrect_avg_gradients_normalized.items():
                #     save_tasks.append((gradients, os.path.join(output_dir, f"average_gradients_prompt_incorrect_normalized_{question_id}.npz")))

            print(f"Saving {len(save_tasks)} gradient files in parallel")
            # Use thread pool to parallelize saving
            # set max_workers to the number of cores
            max_workers = os.cpu_count()
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(executor.map(save_gradient_file, save_tasks))
            print("Finished saving all gradient files")
        
        if type == "both" or type == "squared":
            print("Saving squared gradients")
            data_to_save = {
                # "avg_squared": self.all_avg_gradients_squared,
                "prompt_avg_squared": self.prompt_avg_gradients_squared,
                "prompt_correct_avg_squared": self.prompt_correct_avg_gradients_squared,
                "prompt_incorrect_avg_squared": self.prompt_incorrect_avg_gradients_squared,
                # "prompt_avg_normalized_squared": self.prompt_avg_gradients_normalized_squared,
                # "prompt_correct_avg_normalized_squared": self.prompt_correct_avg_gradients_normalized_squared,
                # "prompt_incorrect_avg_normalized_squared": self.prompt_incorrect_avg_gradients_normalized_squared,
            }
            with open(os.path.join(output_dir, f"average_gradients_squared_{start_idx}_{end_idx}.npz"), 'wb') as f:
                np.savez_compressed(f, **data_to_save)
            # load and print
            # data_to_load = np.load(os.path.join(output_dir, "average_gradients_squared_all.npz"), allow_pickle=True)
            # print(data_to_load)
            # print(data_to_load["avg_squared"])
            # print(data_to_load["prompt_avg_squared"])
            # print(data_to_load["prompt_correct_avg_squared"])
            # print(data_to_load["prompt_incorrect_avg_squared"])

        
        # Save per-prompt results
        per_prompt_summary = {}
        for question_id in self.prompt_total.keys():
            per_prompt_summary[question_id] = {
                'num_rollouts': self.prompt_total[question_id],
                'num_correct': self.prompt_correct_total[question_id],
                'num_incorrect': self.prompt_incorrect_total[question_id]
            }
        
        output_file_name = os.path.join(output_dir, "per_prompt_summary.json")

        if os.path.exists(output_file_name):
            old_per_prompt_summary = json.load(open(output_file_name, 'r'))
        else:
            old_per_prompt_summary = {}
        old_per_prompt_summary.update(per_prompt_summary)
        
        with open(output_file_name, 'w') as f:
            json.dump(old_per_prompt_summary, f, indent=2)
        
        # save qid2grad_norm
        # grad_norm_file = os.path.join(output_dir, "qid2grad_norm.json")
        # if os.path.exists(grad_norm_file):
        #     old_grad_norm = json.load(open(grad_norm_file, 'r'))
        # else:
        #     old_grad_norm = {}
        # old_grad_norm.update(qid2grad_norm)
        # with open(grad_norm_file, 'w') as f:
        #     json.dump(old_grad_norm, f, ensure_ascii=False, indent=2)
        


def main():
    parser = argparse.ArgumentParser(description="Analyze GRPO gradients from checkpoint rollouts")
    parser.add_argument(
        "--rollouts_file", 
        type=str, 
        nargs='+', 
        required=True,
        help="Path(s) to one or more rollouts JSON files, e.g.: --rollouts_file file1.json file2.json file3.json"
    )
    # Example usage:
    #   python checkpoint_gradient_analysis.py --rollouts_file file1.json file2.json --model_path ... [other args]
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./grpo_gradient_analysis_output",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cuda/cpu/auto)")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size for gradient computation (auto-detect if not specified)")
    parser.add_argument("--max_memory_gb", type=float, default=140.0,
                       help="Maximum GPU memory to use in GB (for auto batch size detection)")
    parser.add_argument("--top_entropy_token", action="store_true",help="only get gradient for topentropy tokens in one rollout")
    parser.add_argument("--type", type=str, default="both",
                       help="Type of gradient to compute (mean/squared/both)")
    parser.add_argument("--start_idx", type=int, default=-1,
                       help="Start index of prompts to process")
    parser.add_argument("--end_idx", type=int, default=-1,
                       help="End index of prompts to process")
    parser.add_argument("--simplified", action="store_true", help="Use simplified gradient computation")
    parser.add_argument("--lora", action="store_true", help="Use LoRA model")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank (only used when --lora is set)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (only used when --lora is set)")
    parser.add_argument("--prompt_type", type=str, default=None, choices=[None, "qwen", "llama"],
                       help="Prompt template. If unset, inferred from model_path.")
    parser.add_argument("--mle", action="store_true", help="Use MLE gradient computation")
    parser.add_argument("--id_file", type=str, default=None,
                       help="Name of the index file to process")
    parser.add_argument("--pos_only", action="store_true", help="Only process positive rollouts")
    
    
    args = parser.parse_args()

    # check consistency
    if args.simplified and args.mle:
        raise ValueError("Cannot use both simplified and MLE gradient computation at the same time")
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Initialize analyzer
    analyzer = CheckpointGradientAnalyzer(
        args.model_path,
        device=device,
        lora=args.lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        prompt_type=args.prompt_type,
    )
    analyzer.batch_size = args.batch_size  # Set batch size
    analyzer.max_memory_gb = args.max_memory_gb  # Set memory limit
    analyzer.setup_before_analysis()
    
    # Run analysis
    analyzer.run_full_analysis(args, args.rollouts_file, args.output_dir, args.top_entropy_token, args.type, args.start_idx, args.end_idx)


if __name__ == "__main__":
    main()
